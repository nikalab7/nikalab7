"""Orchestration: origin batches, labels and the walk-forward runner.

This module is where the leakage guarantees become mechanical rather than
aspirational. Everything an origin sees is fetched through
:meth:`MarketSource.hourly_panel` and :meth:`MarketSource.daily_panel` bounded
at that origin, and labels are produced by a separate call that is only made
once the exit session has been observed.

The runner drives the portfolio session by session rather than trade by trade,
because overlapping holds, the entry-at-next-open convention and the drawdown
pause are all properties of the session sequence.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .actions import ActionLedger, HoldingAccount, HoldingCosts, label_net_return
from .calendar_spec import ExchangeCalendar
from .config import DesignConfig
from .decision import DecisionModel, OriginRow, fit_decision_model
from .evaluation import DailySeries, TradeRecord
from .features import (
    DailyFeatures,
    assemble_feature_row,
    build_chronos_task,
    build_daily_features,
    forecast_features,
    recommended_panel_bars,
)
from .forecaster import ForecastResult, Forecaster
from .holdout import HoldoutGuard
from .market import BarPanel, DailyPanel
from .policy import AlertCandidate, OpenPositionView, PolicyEngine, PolicyOutcome
from .portfolio import EntryRequest, ReferencePortfolio
from .protocol import label_available_at
from .quality import EligibilityResult, evaluate_eligibility
from .universe import WatchlistManifest
from .variants import MomentumCandidate, rank_momentum_candidates, variant_spec

__all__ = [
    "PipelineError",
    "MarketSource",
    "FixtureMarketSource",
    "OriginScore",
    "OriginBatch",
    "ResearchPipeline",
    "BacktestResult",
    "WalkForwardRunner",
]


class PipelineError(RuntimeError):
    """Raised when an origin cannot be processed."""


class MarketSource(Protocol):
    """Origin-bounded access to market data."""

    def hourly_panel(
        self, symbol: str, origin_session: dt.date, count: int
    ) -> BarPanel:  # pragma: no cover
        ...

    def daily_panel(self, symbol: str) -> DailyPanel:  # pragma: no cover
        ...

    def action_ledger(self, symbol: str) -> ActionLedger:  # pragma: no cover
        ...


@dataclass
class FixtureMarketSource:
    """Market source backed by a :class:`~chronos2_signal.fixtures.SyntheticMarket`."""

    market: object

    def hourly_panel(self, symbol: str, origin_session: dt.date, count: int) -> BarPanel:
        return self.market.panel_ending_at(symbol, origin_session, count)  # type: ignore[attr-defined]

    def daily_panel(self, symbol: str) -> DailyPanel:
        return self.market.daily(symbol)  # type: ignore[attr-defined]

    def action_ledger(self, symbol: str) -> ActionLedger:
        return self.market.action_ledger(symbol)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class OriginScore:
    """One symbol's state at one origin, before the policy is applied."""

    symbol: str
    issuer_id: str
    sector: str
    eligibility: EligibilityResult
    daily: DailyFeatures | None
    features: Mapping[str, float] | None
    sigma_2d: float
    forecast: ForecastResult | None = None
    forecast_cache_key: str | None = None
    skip_reason: str | None = None

    @property
    def scorable(self) -> bool:
        return self.eligibility.eligible and self.features is not None and self.skip_reason is None


@dataclass(frozen=True)
class OriginBatch:
    """Everything one origin produced for one variant."""

    origin_session: dt.date
    variant: str
    scores: tuple[OriginScore, ...]
    eligible_count: int
    notes: tuple[str, ...] = ()

    def scorable(self) -> tuple[OriginScore, ...]:
        return tuple(score for score in self.scores if score.scorable)

    def rows(self) -> list[OriginRow]:
        return [
            OriginRow(
                origin_session=self.origin_session,
                symbol=score.symbol,
                features=dict(score.features or {}),
                sigma_2d=score.sigma_2d,
                eligible_count=self.eligible_count,
            )
            for score in self.scorable()
        ]


@dataclass
class ResearchPipeline:
    """Builds origin batches and labels under the registered configuration."""

    config: DesignConfig
    calendar: ExchangeCalendar
    source: MarketSource
    watchlist: WatchlistManifest
    forecaster: Forecaster | None = None
    guard: HoldoutGuard = field(default_factory=HoldoutGuard.open)
    market_proxy: str | None = None

    def __post_init__(self) -> None:
        if self.market_proxy is None:
            self.market_proxy = self.config.runtime.market_proxy

    # -- batches ----------------------------------------------------------- #

    def build_batch(
        self,
        origin_session: dt.date,
        variant: str,
        *,
        suppressed_sectors: frozenset[str] = frozenset(),
    ) -> OriginBatch:
        """Evaluate eligibility, features and forecasts for one origin.

        Forecasts are generated for **every** eligible row, weak ones included.
        Keeping only the successful or alerted rows would bias both the fit and
        the calibration.
        """
        spec = variant_spec(variant)
        notes: list[str] = []
        context = spec.context_length or self.config.model.context_length
        # Two independent requirements: the common eligibility mask is defined
        # over a fixed number of scheduled bars, and the matching-slot volume
        # channel needs a further twenty sessions before the context's left
        # edge. Every variant loads the larger of the two so that all of them
        # share one mask.
        panel_bars = max(
            recommended_panel_bars(context),
            self.config.universe.common_hourly_history_bars,
        )

        market_daily = self.source.daily_panel(self.market_proxy or "SPY")
        market_hourly = self._panel(self.market_proxy or "SPY", origin_session, panel_bars)
        if market_hourly is None:
            return OriginBatch(
                origin_session=origin_session,
                variant=variant,
                scores=(),
                eligible_count=0,
                notes=("market proxy panel unavailable: whole scan suppressed",),
            )

        sector_daily: dict[str, DailyPanel] = {}
        sector_hourly: dict[str, BarPanel | None] = {}
        for sector, etf in self.watchlist.sector_etfs().items():
            sector_daily[sector] = self.source.daily_panel(etf)
            sector_hourly[sector] = self._panel(etf, origin_session, panel_bars)
            if sector_hourly[sector] is None:
                notes.append(f"sector {sector}: ETF panel unavailable, sector suppressed")

        # Eligibility first, so that the peer set used for breadth contains only
        # eligible members -- breadth is watchlist breadth, not market breadth.
        eligibility: dict[str, EligibilityResult] = {}
        panels: dict[str, BarPanel] = {}
        for member in self.watchlist.members:
            panel = self._panel(member.symbol, origin_session, panel_bars)
            sector_panel = sector_hourly.get(member.sector)
            if panel is None or sector_panel is None:
                eligibility[member.symbol] = EligibilityResult(
                    symbol=member.symbol,
                    origin_session=origin_session,
                    eligible=False,
                    failures=("panel: required hourly history unavailable at this origin",),
                )
                continue
            panels[member.symbol] = panel
            eligibility[member.symbol] = evaluate_eligibility(
                symbol=member.symbol,
                origin_session=origin_session,
                config=self.config,
                hourly=panel,
                daily=self.source.daily_panel(member.symbol),
                market_hourly=market_hourly,
                sector_hourly=sector_panel,
                action_ledger=self.source.action_ledger(member.symbol),
                issuer_id=member.issuer_id,
                sector_suppressed=(
                    member.sector in suppressed_sectors
                    or sector_hourly.get(member.sector) is None
                ),
            )

        eligible_symbols = [
            symbol for symbol, result in eligibility.items() if result.eligible
        ]
        peers_by_sector: dict[str, dict[str, DailyPanel]] = {}
        for member in self.watchlist.members:
            if member.symbol not in eligible_symbols:
                continue
            peers_by_sector.setdefault(member.sector, {})[member.symbol] = (
                self.source.daily_panel(member.symbol)
            )

        scores: list[OriginScore] = []
        tasks = []
        task_owners = []
        for member in self.watchlist.members:
            result = eligibility[member.symbol]
            if not result.eligible:
                scores.append(
                    OriginScore(
                        symbol=member.symbol,
                        issuer_id=member.issuer_id,
                        sector=member.sector,
                        eligibility=result,
                        daily=None,
                        features=None,
                        sigma_2d=float("nan"),
                    )
                )
                continue

            daily_features = build_daily_features(
                symbol=member.symbol,
                origin_session=origin_session,
                stock_daily=self.source.daily_panel(member.symbol),
                market_daily=market_daily,
                sector_daily=sector_daily[member.sector],
                peer_daily=peers_by_sector.get(member.sector, {}),
                config_sigma_window=self.config.decision.sigma_daily_window_sessions,
                config_sigma_floor=self.config.decision.sigma_2d_floor,
            )
            if not daily_features.valid:
                scores.append(
                    OriginScore(
                        symbol=member.symbol,
                        issuer_id=member.issuer_id,
                        sector=member.sector,
                        eligibility=result,
                        daily=daily_features,
                        features=None,
                        sigma_2d=daily_features.sigma_2d,
                        skip_reason=(
                            "required daily features invalid: "
                            + ", ".join(daily_features.invalid[:3])
                        ),
                    )
                )
                continue

            if not spec.uses_forecast:
                scores.append(
                    OriginScore(
                        symbol=member.symbol,
                        issuer_id=member.issuer_id,
                        sector=member.sector,
                        eligibility=result,
                        daily=daily_features,
                        features=assemble_feature_row(daily=daily_features, forecast=None),
                        sigma_2d=daily_features.sigma_2d,
                    )
                )
                continue

            try:
                task = build_chronos_task(
                    symbol=member.symbol,
                    origin_session=origin_session,
                    calendar=self.calendar,
                    stock=panels[member.symbol],
                    market=market_hourly,
                    sector=sector_hourly[member.sector],  # type: ignore[arg-type]
                    context_length=context,
                    horizon_sessions=self.config.model.horizon_sessions,
                    include_covariates=spec.include_covariates,
                )
            except Exception as exc:
                scores.append(
                    OriginScore(
                        symbol=member.symbol,
                        issuer_id=member.issuer_id,
                        sector=member.sector,
                        eligibility=result,
                        daily=daily_features,
                        features=None,
                        sigma_2d=daily_features.sigma_2d,
                        skip_reason=f"task build failed: {exc}",
                    )
                )
                continue
            tasks.append(task)
            task_owners.append((member, result, daily_features))

        if tasks:
            if self.forecaster is None:
                raise PipelineError(
                    f"variant {variant} needs forecasts but no forecaster is configured; "
                    "the pipeline does not fabricate them"
                )
            forecasts = self.forecaster.predict(tasks)
            for (member, result, daily_features), task, forecast in zip(
                task_owners, tasks, forecasts, strict=True
            ):
                if not forecast.usable:
                    scores.append(
                        OriginScore(
                            symbol=member.symbol,
                            issuer_id=member.issuer_id,
                            sector=member.sector,
                            eligibility=result,
                            daily=daily_features,
                            features=None,
                            sigma_2d=daily_features.sigma_2d,
                            forecast=forecast,
                            skip_reason=f"forecast blocked: {forecast.status.value}",
                        )
                    )
                    continue
                derived = forecast_features(
                    quantiles=forecast.as_mapping(),
                    terminal_index_d1=forecast.terminal_indices[0],
                    terminal_index_d2=forecast.terminal_indices[-1],
                    scale=daily_features.sigma_2d,
                    task=task,
                )
                scores.append(
                    OriginScore(
                        symbol=member.symbol,
                        issuer_id=member.issuer_id,
                        sector=member.sector,
                        eligibility=result,
                        daily=daily_features,
                        features=assemble_feature_row(
                            daily=daily_features, forecast=derived
                        ),
                        sigma_2d=daily_features.sigma_2d,
                        forecast=forecast,
                    )
                )

        scores.sort(key=lambda score: score.symbol)
        return OriginBatch(
            origin_session=origin_session,
            variant=variant,
            scores=tuple(scores),
            eligible_count=len(eligible_symbols),
            notes=tuple(notes),
        )

    def _panel(
        self, symbol: str, origin_session: dt.date, count: int
    ) -> BarPanel | None:
        try:
            return self.source.hourly_panel(symbol, origin_session, count)
        except Exception:
            # A missing panel is a suppression input, never a filled-in series.
            return None

    # -- labels ------------------------------------------------------------ #

    def label(
        self,
        origin_session: dt.date,
        symbol: str,
        *,
        cost_scenario: str = "base",
    ) -> HoldingAccount | None:
        """Realised net return of the reference policy for one origin.

        Returns ``None`` when the entry or exit reference price is unavailable.
        The caller keeps such a row out of fitting, and keeps the failed trade
        visible in the ledger rather than dropping it.
        """
        timing = label_available_at(
            self.calendar,
            origin_session,
            horizon_sessions=self.config.model.horizon_sessions,
        )
        daily = self.source.daily_panel(symbol)
        try:
            entry_row = daily.row(timing.entry_session)
            exit_row = daily.row(timing.exit_session)
        except Exception:
            return None
        entry_open = float(entry_row["open"])
        exit_close = float(exit_row["close"])
        if not (math.isfinite(entry_open) and math.isfinite(exit_close)):
            return None
        slippage = self.config.execution.slippage_fraction(cost_scenario)
        costs = HoldingCosts(
            buy_slippage=slippage,
            sell_slippage=slippage,
            explicit_fee_fraction=self.config.execution.explicit_fee_fraction,
        )
        return label_net_return(
            entry_session=timing.entry_session,
            exit_session=timing.exit_session,
            entry_open_price=entry_open,
            exit_close_price=exit_close,
            costs=costs,
            ledger=self.source.action_ledger(symbol),
        )

    def labelled_rows(
        self, batch: OriginBatch, *, cost_scenario: str = "base"
    ) -> list[OriginRow]:
        """Rows of ``batch`` with their realised labels attached."""
        timing = label_available_at(
            self.calendar,
            batch.origin_session,
            horizon_sessions=self.config.model.horizon_sessions,
        )
        rows: list[OriginRow] = []
        for score in batch.scorable():
            account = self.label(
                batch.origin_session, score.symbol, cost_scenario=cost_scenario
            )
            if account is None:
                continue
            rows.append(
                OriginRow(
                    origin_session=batch.origin_session,
                    symbol=score.symbol,
                    features=dict(score.features or {}),
                    sigma_2d=score.sigma_2d,
                    r_net=account.net_return,
                    label_available_at=timing.available_at.to_pydatetime(),
                    eligible_count=batch.eligible_count,
                )
            )
        return rows

    # -- correlation ------------------------------------------------------- #

    def correlation_lookup(self, origin_session: dt.date) -> Callable[[str, str], float | None]:
        """Trailing daily-return correlation, bounded at ``origin_session``.

        Returns ``None`` when the window cannot be filled, which blocks the
        allocation rather than assuming independence.
        """
        window = self.config.portfolio.correlation_window_sessions
        cache: dict[str, np.ndarray | None] = {}

        def returns_for(symbol: str) -> np.ndarray | None:
            if symbol not in cache:
                panel = self.source.daily_panel(symbol).up_to(origin_session)
                series = panel.log_returns()
                tail = series[-window:]
                cache[symbol] = tail if tail.size >= window else None
            return cache[symbol]

        def lookup(left: str, right: str) -> float | None:
            if left == right:
                return 1.0
            a, b = returns_for(left), returns_for(right)
            if a is None or b is None:
                return None
            if np.std(a) == 0.0 or np.std(b) == 0.0:
                return None
            value = float(np.corrcoef(a, b)[0, 1])
            return value if math.isfinite(value) else None

        return lookup


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of one walk-forward run for one system."""

    variant: str
    trades: tuple[TradeRecord, ...]
    daily: DailySeries
    portfolio: ReferencePortfolio
    origins_scanned: int
    alerts: int
    no_alert_origins: tuple[dt.date, ...]
    policy_outcomes: tuple[PolicyOutcome, ...]
    forecaster_provenance: Mapping[str, object] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "origins_scanned": self.origins_scanned,
            "alerts": self.alerts,
            "trades": len(self.trades),
            "no_alert_origins": len(self.no_alert_origins),
            "portfolio": self.portfolio.summary(),
            "forecaster": dict(self.forecaster_provenance),
            "notes": list(self.notes),
        }


@dataclass
class WalkForwardRunner:
    """Scores a block of origins and walks the reference portfolio through it."""

    pipeline: ResearchPipeline
    cost_scenario: str = "base"

    @property
    def config(self) -> DesignConfig:
        return self.pipeline.config

    def fit(
        self,
        variant: str,
        *,
        fit_sessions: Sequence[dt.date],
        calibration_sessions: Sequence[dt.date],
        fit_deadline: dt.datetime | None = None,
        calibration_deadline: dt.datetime | None = None,
        batch_cache: dict[tuple[str, dt.date], OriginBatch] | None = None,
    ) -> DecisionModel:
        """Fit one model from labelled rows of the two blocks."""
        fit_rows: list[OriginRow] = []
        calibration_rows: list[OriginRow] = []
        for sessions, sink in (
            (fit_sessions, fit_rows),
            (calibration_sessions, calibration_rows),
        ):
            for session in sessions:
                batch = self._batch(variant, session, batch_cache)
                sink.extend(
                    self.pipeline.labelled_rows(batch, cost_scenario=self.cost_scenario)
                )
        return fit_decision_model(
            variant=variant,
            fit_rows=fit_rows,
            calibration_rows=calibration_rows,
            config=self.config,
            fit_deadline=fit_deadline,
            calibration_deadline=calibration_deadline,
        )

    def run(
        self,
        variant: str,
        *,
        model: DecisionModel,
        score_sessions: Sequence[dt.date],
        portfolio: ReferencePortfolio | None = None,
        batch_cache: dict[tuple[str, dt.date], OriginBatch] | None = None,
        model_for: Callable[[dt.date], DecisionModel] | None = None,
    ) -> BacktestResult:
        """Score ``score_sessions`` in order and run the reference portfolio.

        Args:
            model_for: Optional prequential model selector. Supplying it is how
                the registered 20-session refit schedule is replayed: the model
                in force at an origin is the one fitted from matured history
                before it.
        """
        if not score_sessions:
            raise PipelineError("no origins to score")
        calendar = self.pipeline.calendar
        book = portfolio or ReferencePortfolio(
            config=self.config, variant=variant, cost_scenario=self.cost_scenario
        )
        engine = PolicyEngine(self.config)

        pending: dict[dt.date, list[EntryRequest]] = {}
        outcomes: list[PolicyOutcome] = []
        no_alert: list[dt.date] = []
        alerts = 0
        trades: list[TradeRecord] = []
        entry_meta: dict[str, tuple[str, str]] = {}

        ordered = sorted(set(score_sessions))
        horizon_sessions = self.config.model.horizon_sessions
        last_origin = ordered[-1]
        final_exit = label_available_at(
            calendar, last_origin, horizon_sessions=horizon_sessions
        ).exit_session
        walk_sessions = calendar.sessions(ordered[0], final_exit)

        origin_set = set(ordered)
        for session in walk_sessions:
            # 1. Entries at the official open, sized from then-observable state.
            due = pending.pop(session, [])
            if due:
                equity_at_open = (
                    book.days[-1].equity
                    if book.days
                    else self.config.portfolio.initial_equity
                )
                gross_at_open = (
                    book.days[-1].gross_exposure * book.days[-1].equity
                    if book.days
                    else 0.0
                )
                book.enter_all(
                    due,
                    equity_at_open=equity_at_open,
                    gross_value_at_open=gross_at_open,
                )

            ledgers = {
                position.symbol: self.pipeline.source.action_ledger(position.symbol)
                for position in book.positions
            }
            book.apply_splits(session, ledgers)
            book.credit_dividends(session, ledgers)
            book.record_excursion(self._session_lows(book, session))

            # 2. Scoring happens after the close of a signal session.
            if session in origin_set:
                active = model_for(session) if model_for is not None else model
                outcome = self._score_origin(
                    variant, session, active, book, engine, batch_cache
                )
                outcomes.append(outcome)
                if outcome.alert_count == 0:
                    no_alert.append(session)
                alerts += outcome.alert_count
                timing = label_available_at(
                    calendar, session, horizon_sessions=horizon_sessions
                )
                for decision in outcome.alerts:
                    candidate = decision.candidate
                    entry_open = self._reference_price(
                        candidate.symbol, timing.entry_session, "open"
                    )
                    if entry_open is None:
                        continue
                    entry_meta[candidate.symbol] = (
                        candidate.issuer_id,
                        candidate.sector,
                    )
                    pending.setdefault(timing.entry_session, []).append(
                        EntryRequest(
                            signal_id=f"{variant}-{session.isoformat()}-{candidate.symbol}",
                            symbol=candidate.symbol,
                            issuer_id=candidate.issuer_id,
                            sector=candidate.sector,
                            signal_session=session,
                            entry_session=timing.entry_session,
                            planned_exit_session=timing.exit_session,
                            reference_open_price=entry_open,
                            action_ledger=self.pipeline.source.action_ledger(
                                candidate.symbol
                            ),
                        )
                    )

            # 3. Exits at the official close, then the daily mark.
            closes = self._session_closes(book, session)
            closed = book.close_due(session, closes, ledgers)
            for position in closed:
                trades.append(self._trade_record(position))
            book.mark(session, self._session_closes(book, session))

        return BacktestResult(
            variant=variant,
            trades=tuple(trades),
            daily=DailySeries(
                name=variant,
                sessions=tuple(book.sessions()),
                returns=np.asarray(book.daily_returns(), dtype=float),
            ),
            portfolio=book,
            origins_scanned=len(ordered),
            alerts=alerts,
            no_alert_origins=tuple(no_alert),
            policy_outcomes=tuple(outcomes),
            forecaster_provenance=(
                self.pipeline.forecaster.provenance()
                if self.pipeline.forecaster is not None
                else {"forecaster": "none"}
            ),
        )

    # -- helpers ----------------------------------------------------------- #

    def _batch(
        self,
        variant: str,
        session: dt.date,
        cache: dict[tuple[str, dt.date], OriginBatch] | None,
    ) -> OriginBatch:
        key = (variant, session)
        if cache is not None and key in cache:
            return cache[key]
        batch = self.pipeline.build_batch(session, variant)
        if cache is not None:
            cache[key] = batch
        return batch

    def _score_origin(
        self,
        variant: str,
        session: dt.date,
        model: DecisionModel,
        book: ReferencePortfolio,
        engine: PolicyEngine,
        cache: dict[tuple[str, dt.date], OriginBatch] | None,
    ) -> PolicyOutcome:
        batch = self._batch(variant, session, cache)
        candidates: list[AlertCandidate] = []
        for score in batch.scorable():
            row = OriginRow(
                origin_session=session,
                symbol=score.symbol,
                features=dict(score.features or {}),
                sigma_2d=score.sigma_2d,
                eligible_count=batch.eligible_count,
            )
            try:
                estimate = model.estimate(row)
            except Exception:
                continue
            candidates.append(
                AlertCandidate(
                    symbol=score.symbol,
                    issuer_id=score.issuer_id,
                    sector=score.sector,
                    estimate=estimate,
                    eligibility=score.eligibility,
                    out_of_sample_observations=model.report.calibration_rows,
                    out_of_sample_dates=model.report.calibration_dates,
                )
            )
        open_views = [
            OpenPositionView(
                symbol=position.symbol,
                issuer_id=position.issuer_id,
                sector=position.sector,
                planned_exit_session=position.planned_exit_session,
            )
            for position in book.positions
        ]
        return engine.evaluate(
            origin_session=session,
            candidates=candidates,
            open_positions=open_views,
            correlation=self.pipeline.correlation_lookup(session),
            paused=book.paused,
        )

    def _reference_price(
        self, symbol: str, session: dt.date, column: str
    ) -> float | None:
        try:
            row = self.pipeline.source.daily_panel(symbol).row(session)
        except Exception:
            return None
        value = float(row[column])
        return value if math.isfinite(value) and value > 0.0 else None

    def _session_closes(
        self, book: ReferencePortfolio, session: dt.date
    ) -> dict[str, float]:
        prices: dict[str, float] = {}
        for position in book.positions:
            price = self._reference_price(position.symbol, session, "close")
            if price is not None:
                prices[position.symbol] = price
            else:
                # Mark at the entry fill rather than inventing a price, and let
                # the exit path record the unresolved exit.
                prices[position.symbol] = position.entry_fill_price
        return prices

    def _session_lows(
        self, book: ReferencePortfolio, session: dt.date
    ) -> dict[str, float]:
        lows: dict[str, float] = {}
        for position in book.positions:
            price = self._reference_price(position.symbol, session, "low")
            if price is not None:
                lows[position.symbol] = price
        return lows

    def _trade_record(self, position: object) -> TradeRecord:
        source = self.pipeline.source
        ledger = source.action_ledger(position.symbol)
        resolved = position.status == "closed"
        base = position.realised_net_return() if resolved else float("nan")
        scenarios: dict[str, float] = {"base": base}
        for scenario in ("stress", "severe"):
            if not resolved or position.exit_reference_price is None:
                scenarios[scenario] = float("nan")
                continue
            slippage = self.config.execution.slippage_fraction(scenario)
            account = label_net_return(
                entry_session=position.entry_session,
                exit_session=position.exit_session,  # type: ignore[arg-type]
                entry_open_price=position.entry_reference_price,
                exit_close_price=position.exit_reference_price,
                costs=HoldingCosts(
                    buy_slippage=slippage,
                    sell_slippage=slippage,
                    explicit_fee_fraction=self.config.execution.explicit_fee_fraction,
                ),
                ledger=ledger,
            )
            scenarios[scenario] = account.net_return
        return TradeRecord(
            origin_session=position.signal_session,
            symbol=position.symbol,
            issuer_id=position.issuer_id,
            sector=position.sector,
            r_net_base=scenarios["base"],
            r_net_stress=scenarios["stress"],
            r_net_severe=scenarios["severe"],
            max_adverse_excursion=position.max_adverse_excursion,
            resolved=resolved,
        )

    # -- momentum control -------------------------------------------------- #

    def run_momentum_control(
        self,
        *,
        score_sessions: Sequence[dt.date],
        reference_variant: str = "B0",
        batch_cache: dict[tuple[str, dt.date], OriginBatch] | None = None,
        name: str = "momentum",
    ) -> BacktestResult:
        """The fixed relative-momentum control, on identical dates and costs.

        It reuses the same eligibility mask and the same portfolio machinery, so
        the only difference from a candidate is the selection rule.
        """
        calendar = self.pipeline.calendar
        book = ReferencePortfolio(
            config=self.config, variant=name, cost_scenario=self.cost_scenario
        )
        pending: dict[dt.date, list[EntryRequest]] = {}
        trades: list[TradeRecord] = []
        alerts = 0
        no_alert: list[dt.date] = []

        ordered = sorted(set(score_sessions))
        horizon_sessions = self.config.model.horizon_sessions
        final_exit = label_available_at(
            calendar, ordered[-1], horizon_sessions=horizon_sessions
        ).exit_session
        origin_set = set(ordered)

        for session in calendar.sessions(ordered[0], final_exit):
            due = pending.pop(session, [])
            if due:
                equity_at_open = (
                    book.days[-1].equity if book.days else self.config.portfolio.initial_equity
                )
                gross_at_open = (
                    book.days[-1].gross_exposure * book.days[-1].equity if book.days else 0.0
                )
                book.enter_all(
                    due, equity_at_open=equity_at_open, gross_value_at_open=gross_at_open
                )
            ledgers = {
                position.symbol: self.pipeline.source.action_ledger(position.symbol)
                for position in book.positions
            }
            book.apply_splits(session, ledgers)
            book.credit_dividends(session, ledgers)
            book.record_excursion(self._session_lows(book, session))

            if session in origin_set:
                batch = self._batch(reference_variant, session, batch_cache)
                candidates = [
                    MomentumCandidate(
                        symbol=score.symbol,
                        issuer_id=score.issuer_id,
                        sector=score.sector,
                        stock_minus_sector_5s=float(
                            (score.daily.values if score.daily else {}).get(
                                "f10_stock_minus_sector_return_5s", float("nan")
                            )
                        ),
                    )
                    for score in batch.scorable()
                ]
                picks = (
                    []
                    if book.paused
                    else rank_momentum_candidates(
                        candidates,
                        config=self.config,
                        open_sectors=book.sector_counts(),
                        open_issuers=sorted(book.open_issuers()),
                        open_position_count=len(book.positions),
                    )
                )
                if not picks:
                    no_alert.append(session)
                alerts += len(picks)
                timing = label_available_at(
                    calendar, session, horizon_sessions=horizon_sessions
                )
                for pick in picks:
                    entry_open = self._reference_price(
                        pick.symbol, timing.entry_session, "open"
                    )
                    if entry_open is None:
                        continue
                    pending.setdefault(timing.entry_session, []).append(
                        EntryRequest(
                            signal_id=f"{name}-{session.isoformat()}-{pick.symbol}",
                            symbol=pick.symbol,
                            issuer_id=pick.issuer_id,
                            sector=pick.sector,
                            signal_session=session,
                            entry_session=timing.entry_session,
                            planned_exit_session=timing.exit_session,
                            reference_open_price=entry_open,
                            action_ledger=self.pipeline.source.action_ledger(pick.symbol),
                        )
                    )

            closes = self._session_closes(book, session)
            for position in book.close_due(session, closes, ledgers):
                trades.append(self._trade_record(position))
            book.mark(session, self._session_closes(book, session))

        return BacktestResult(
            variant=name,
            trades=tuple(trades),
            daily=DailySeries(
                name=name,
                sessions=tuple(book.sessions()),
                returns=np.asarray(book.daily_returns(), dtype=float),
            ),
            portfolio=book,
            origins_scanned=len(ordered),
            alerts=alerts,
            no_alert_origins=tuple(no_alert),
            policy_outcomes=(),
            forecaster_provenance={"forecaster": "none (fixed momentum control)"},
            notes=("fixed relative-momentum control: identical dates, fills and costs",),
        )
