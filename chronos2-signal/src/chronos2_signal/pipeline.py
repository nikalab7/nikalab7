"""Origin batches and labels.

Everything an origin sees comes from one :class:`~chronos2_signal.sources.MarketView`
bounded at that origin, so no feature can read a later bar. Labels come from a
separate view bounded at the exit session, and are only asked for once that
session has been observed.

Three guarantees are enforced here rather than assumed:

* **The holdout.** Every batch and every label is checked against the
  pipeline's :class:`~chronos2_signal.holdout.HoldoutGuard` before any data is
  read. A development run that reaches a reserved origin raises.
* **The split contract.** Unusable split audits inside an origin's feature
  window make that symbol ineligible; inside a holding window they make the
  label unavailable. Neither is ever reconstructed from a guess.
* **Forecasts before outcomes.** With a forecast cache attached, every forecast
  is written to the append-only ledger when the batch is built -- before any
  label for that origin is computed.

The walk-forward simulation that consumes these batches lives in
:mod:`chronos2_signal.simulation`.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .actions import HoldingAccount, HoldingCosts, SplitAudit, label_net_return
from .calendar_spec import CalendarError, ExchangeCalendar
from .config import DesignConfig
from .decision import OriginRow
from .features import (
    DailyFeatures,
    FeatureError,
    assemble_feature_row,
    build_chronos_task,
    build_daily_features,
    forecast_features,
    recommended_panel_bars,
)
from .forecast_store import ForecastCache
from .forecaster import ForecastResult, Forecaster
from .holdout import HoldoutGuard
from .market import BarPanel, DailyPanel, MarketDataError, aligned_log_returns
from .protocol import label_available_at
from .provenance import stable_hash
from .quality import EligibilityResult, evaluate_eligibility
from .sources import MarketSource, MarketView, SourceError
from .universe import WatchlistManifest
from .variants import REGISTERED_VARIANTS, variant_spec

__all__ = [
    "PipelineError",
    "OriginScore",
    "OriginBatch",
    "ResearchPipeline",
]

#: Failures that mean "this data is not available at this origin". Anything else
#: is a defect and is allowed to propagate rather than being mistaken for a gap.
_DATA_UNAVAILABLE = (CalendarError, MarketDataError, SourceError, ValueError, KeyError)


class PipelineError(RuntimeError):
    """Raised when an origin cannot be processed."""


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
    #: Set when the market proxy itself was unusable at this origin.
    scan_suppressed: bool = False
    #: Sectors whose ETF input was unusable at this origin.
    suppressed_sectors: frozenset[str] = frozenset()

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
    """Builds origin batches and labels under the registered configuration.

    Attributes:
        guard: Holdout access control. The default reserves nothing, which is
            right for integrity fixtures; every study replaces it with its
            schedule's guard.
        forecast_cache: When set, forecasts are recorded to and served from the
            append-only ledger. The operational after-close run always sets it.
    """

    config: DesignConfig
    calendar: ExchangeCalendar
    source: MarketSource
    watchlist: WatchlistManifest
    forecaster: Forecaster | None = None
    guard: HoldoutGuard = field(default_factory=HoldoutGuard.open)
    market_proxy: str | None = None
    forecast_cache: ForecastCache | None = None

    def __post_init__(self) -> None:
        if self.market_proxy is None:
            self.market_proxy = self.config.runtime.market_proxy

    # -- batches ----------------------------------------------------------- #

    def panel_bars(self) -> int:
        """Hourly bars every variant loads, whatever its own context length.

        Two requirements meet here: the common eligibility mask is defined over
        a fixed number of scheduled bars, and the matching-slot volume channel
        needs twenty further sessions before a context's left edge. Sizing the
        panel by the *longest registered* context, rather than by each
        variant's own, is what makes the split audit -- and so the mask -- the
        same for every variant: a data problem in the stretch only C512 reads
        excludes the stock for all of them, never for C512 alone.
        """
        longest = max(
            spec.context_length or self.config.model.context_length
            for spec in REGISTERED_VARIANTS.values()
        )
        return max(
            recommended_panel_bars(longest),
            self.config.universe.common_hourly_history_bars,
        )

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

        Raises:
            HoldoutViolation: If the origin is reserved and the guard is locked.
        """
        self.guard.check(origin_session, purpose=f"{variant} batch construction")
        spec = variant_spec(variant)
        view = self.source.view(origin_session)
        notes: list[str] = []
        context = spec.context_length or self.config.model.context_length
        panel_bars = self.panel_bars()
        proxy = self.market_proxy or "SPY"

        market_hourly, reason = self._usable_context_panel(view, proxy, panel_bars)
        if market_hourly is None:
            return OriginBatch(
                origin_session=origin_session,
                variant=variant,
                scores=(),
                eligible_count=0,
                notes=(f"market proxy {proxy} unusable ({reason}): whole scan suppressed",),
                scan_suppressed=True,
            )
        market_daily = view.daily_panel(proxy)

        sector_daily: dict[str, DailyPanel] = {}
        sector_hourly: dict[str, BarPanel | None] = {}
        suppressed = set(suppressed_sectors)
        for sector, etf in self.watchlist.sector_etfs().items():
            sector_hourly[sector], reason = self._usable_context_panel(view, etf, panel_bars)
            if sector_hourly[sector] is None:
                suppressed.add(sector)
                notes.append(f"sector {sector}: ETF {etf} unusable ({reason}), sector suppressed")
            else:
                sector_daily[sector] = view.daily_panel(etf)

        # Eligibility first, so that the peer set used for breadth contains only
        # eligible members -- breadth is watchlist breadth, not market breadth.
        eligibility: dict[str, EligibilityResult] = {}
        panels: dict[str, BarPanel] = {}
        for member in self.watchlist.members:
            panel, reason = self._panel(view, member.symbol, panel_bars)
            if panel is None or sector_hourly.get(member.sector) is None:
                eligibility[member.symbol] = EligibilityResult(
                    symbol=member.symbol,
                    origin_session=origin_session,
                    eligible=False,
                    failures=(
                        f"panel: required hourly history unavailable at this origin ({reason})"
                        if panel is None
                        else "sector_suppressed: required sector input unavailable at this origin",
                    ),
                )
                continue
            panels[member.symbol] = panel
            eligibility[member.symbol] = evaluate_eligibility(
                symbol=member.symbol,
                origin_session=origin_session,
                config=self.config,
                hourly=panel,
                daily=view.daily_panel(member.symbol),
                market_hourly=market_hourly,
                sector_hourly=sector_hourly[member.sector],  # type: ignore[arg-type]
                action_ledger=view.action_ledger(member.symbol),
                split_audits=_audits_in_window(view, member.symbol, panel),
                issuer_id=member.issuer_id,
                sector_suppressed=member.sector in suppressed,
            )

        eligible_symbols = [
            symbol for symbol, result in eligibility.items() if result.eligible
        ]
        peers_by_sector: dict[str, dict[str, DailyPanel]] = {}
        for member in self.watchlist.members:
            if member.symbol in eligible_symbols:
                peers_by_sector.setdefault(member.sector, {})[member.symbol] = (
                    view.daily_panel(member.symbol)
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
                stock_daily=view.daily_panel(member.symbol),
                market_daily=market_daily,
                sector_daily=sector_daily[member.sector],
                peer_daily=peers_by_sector.get(member.sector, {}),
                config_sigma_window=self.config.decision.sigma_daily_window_sessions,
                config_sigma_floor=self.config.decision.sigma_2d_floor,
                calendar=self.calendar,
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
            except (FeatureError, CalendarError) as exc:
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
            forecasts, keys = self._forecast(
                tasks,
                variant=variant,
                snapshot_hashes={
                    task.symbol: stable_hash(
                        [
                            panels[task.symbol].snapshot_hash,
                            market_hourly.snapshot_hash,
                            sector_hourly[owner.sector].snapshot_hash,  # type: ignore[union-attr]
                        ]
                    )
                    for task, (owner, _result, _daily) in zip(tasks, task_owners, strict=True)
                },
            )
            for (member, result, daily_features), task, forecast, key in zip(
                task_owners, tasks, forecasts, keys, strict=True
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
                            forecast_cache_key=key,
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
                        forecast_cache_key=key,
                    )
                )

        scores.sort(key=lambda score: score.symbol)
        return OriginBatch(
            origin_session=origin_session,
            variant=variant,
            scores=tuple(scores),
            eligible_count=len(eligible_symbols),
            notes=tuple(notes),
            suppressed_sectors=frozenset(suppressed),
        )

    def _forecast(
        self,
        tasks: list,
        *,
        variant: str,
        snapshot_hashes: Mapping[str, str],
    ) -> tuple[list[ForecastResult], list[str | None]]:
        if self.forecaster is None:
            raise PipelineError(
                f"variant {variant} needs forecasts but no forecaster is configured; "
                "the pipeline does not fabricate them"
            )
        if self.forecast_cache is None:
            return self.forecaster.predict(tasks), [None] * len(tasks)
        pairs = self.forecast_cache.predict(
            self.forecaster, tasks, variant=variant, snapshot_hashes=snapshot_hashes
        )
        return [result for result, _key in pairs], [key for _result, key in pairs]

    @staticmethod
    def _panel(
        view: MarketView, symbol: str, count: int
    ) -> tuple[BarPanel | None, str | None]:
        """An hourly panel, or ``None`` with the reason it is unavailable.

        A missing panel is a suppression input, never a filled-in series. Only
        data-availability errors are converted; a defect propagates.
        """
        try:
            return view.hourly_panel(symbol, count), None
        except _DATA_UNAVAILABLE as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _usable_context_panel(
        self, view: MarketView, symbol: str, count: int
    ) -> tuple[BarPanel | None, str | None]:
        """A market or sector input panel, unless its data is unusable.

        A context series with an unresolved split in its window would feed a
        manufactured move into every stock that uses it, so it is treated the
        same as a missing one.
        """
        panel, reason = self._panel(view, symbol, count)
        if panel is None:
            return None, reason
        unusable = [audit for audit in _audits_in_window(view, symbol, panel) if not audit.usable]
        if unusable:
            first = unusable[0]
            return None, (
                f"unresolved split on {first.action.ex_date.isoformat()} "
                f"({first.verdict.value})"
            )
        return panel, None

    # -- labels ------------------------------------------------------------ #

    def label(
        self,
        origin_session: dt.date,
        symbol: str,
        *,
        cost_scenario: str = "base",
    ) -> HoldingAccount | None:
        """Realised net return of the reference policy for one origin.

        Read from a view bounded at the exit session, so both reference prices
        come from one consistently restated series. Returns ``None`` when the
        entry or exit price is unavailable, or when a split inside the holding
        window could not be resolved: that episode is quarantined rather than
        turned into a manufactured return.

        Raises:
            HoldoutViolation: If the origin is reserved and the guard is locked.
        """
        self.guard.check(origin_session, purpose="label read")
        timing = label_available_at(
            self.calendar,
            origin_session,
            horizon_sessions=self.config.model.horizon_sessions,
        )
        view = self.source.view(timing.exit_session)
        for audit in view.split_audits(symbol):
            if timing.entry_session < audit.action.ex_date <= timing.exit_session and not audit.usable:
                return None
        daily = view.daily_panel(symbol)
        try:
            entry_row = daily.row(timing.entry_session)
            exit_row = daily.row(timing.exit_session)
        except MarketDataError:
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
            ledger=view.action_ledger(symbol),
        )

    def realised_log_return(self, origin_session: dt.date, symbol: str) -> float | None:
        """``100 * ln(C_D2 / C_D0)``: the quantity the forecast quantiles describe.

        Both closes come from the exit-session view. Mixing a close from the
        origin view with one from the exit view would compare two different
        unit bases whenever a split fell between them.
        """
        self.guard.check(origin_session, purpose="realised outcome read")
        timing = label_available_at(
            self.calendar,
            origin_session,
            horizon_sessions=self.config.model.horizon_sessions,
        )
        view = self.source.view(timing.exit_session)
        for audit in view.split_audits(symbol):
            if origin_session < audit.action.ex_date <= timing.exit_session and not audit.usable:
                return None
        daily = view.daily_panel(symbol)
        try:
            start = float(daily.row(origin_session)["close"])
            end = float(daily.row(timing.exit_session)["close"])
        except MarketDataError:
            return None
        if not (math.isfinite(start) and math.isfinite(end)) or start <= 0.0 or end <= 0.0:
            return None
        return 100.0 * math.log(end / start)

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

        Computed on the sessions both symbols actually have. Returns ``None``
        when the window cannot be filled, which blocks the allocation rather
        than assuming independence.
        """
        window = self.config.portfolio.correlation_window_sessions
        view = self.source.view(origin_session)
        cache: dict[tuple[str, str], float | None] = {}

        def lookup(left: str, right: str) -> float | None:
            if left == right:
                return 1.0
            key = (left, right) if left <= right else (right, left)
            if key in cache:
                return cache[key]
            value: float | None = None
            joined = aligned_log_returns(view.daily_panel(key[0]), view.daily_panel(key[1]))
            if joined is not None:
                a, b = joined
                if a.size >= window:
                    a, b = a[-window:], b[-window:]
                    finite = np.isfinite(a) & np.isfinite(b)
                    if finite.all() and np.std(a) > 0.0 and np.std(b) > 0.0:
                        candidate = float(np.corrcoef(a, b)[0, 1])
                        value = candidate if math.isfinite(candidate) else None
            cache[key] = value
            return value

        return lookup


def _audits_in_window(
    view: MarketView, symbol: str, panel: BarPanel
) -> list[SplitAudit]:
    """Split audits whose ex-date falls inside the data an origin reads.

    The hourly panel reaches furthest back of anything a feature uses -- the
    daily features need at most sixty sessions -- so its first session bounds
    the window. A split before it leaves no step in any feature input.
    """
    window_start = panel.bars[0].session
    return [
        audit for audit in view.split_audits(symbol) if window_start < audit.action.ex_date
    ]
