"""The walk-forward simulation: one session engine for every system.

Candidates, B0 and the fixed momentum control are all simulated by the same
:func:`simulate` loop. The only thing that differs between them is the
:class:`Selector` that decides what to buy after each signal session. That is
what makes invariant 12 -- identical dates, fills, costs and capital accounting
-- structural rather than a property two parallel copies of a loop happen to
share.

Scoring is likewise one function. :func:`score_origin` is called by the
backtest selector and by the operational after-close run alike, so a backtest
exercises exactly the code path the live system would, and the two cannot drift
apart.

The order of events inside a session follows the reference execution policy:

1. entries fill at the official open, sized from the equity observed at the
   previous close;
2. every open position is brought into this session's price units -- a view
   after an ex-date can quote post-split prices where the entry was filled
   from pre-split ones;
3. corporate actions with that ex-date take effect;
4. positions due to exit leave at the official close;
5. the portfolio is marked at the close, which also updates the drawdown pause;
6. only then is the session scored as a signal origin -- after the close, against
   the state that close produced. Positions that exited today no longer occupy
   capacity, and a pause triggered today already applies.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .actions import SPLIT_TOLERANCE, ActionLedger, HoldingCosts, label_net_return
from .config import DesignConfig
from .decision import DecisionError, DecisionModel, OriginRow, fit_decision_model
from .evaluation import DailySeries, TradeRecord
from .market import MarketDataError
from .pipeline import OriginBatch, ResearchPipeline
from .policy import (
    AlertCandidate,
    OpenPositionView,
    PolicyEngine,
    PolicyOutcome,
)
from .portfolio import EntryRequest, PaperPosition, ReferencePortfolio
from .protocol import label_available_at
from .sources import MarketView
from .variants import MomentumCandidate, rank_momentum_candidates, variant_spec

__all__ = [
    "SimulationError",
    "PredictionRecord",
    "ScoredOrigin",
    "score_origin",
    "ModelSchedule",
    "Selection",
    "SelectionOutcome",
    "Selector",
    "PolicySelector",
    "MomentumSelector",
    "BacktestResult",
    "simulate",
    "cached_batch",
    "WalkForwardRunner",
]

#: Batches keyed by ``(variant, session)``. A plain dict so that callers can
#: share one cache across fits, runs and controls.
BatchCache = dict

#: Relative band within which a re-read entry quote counts as unchanged, or as
#: restated by exactly the recorded splits. It is the split audit's own band:
#: wide enough that an ordinary vendor revision is not read as a change of
#: units, and a split too close to one to separate from it is already
#: ambiguous to the audit.
_UNIT_TOLERANCE = SPLIT_TOLERANCE


class SimulationError(RuntimeError):
    """Raised when a simulation cannot be run as registered."""


# --------------------------------------------------------------------------- #
# Scoring: one function for the backtest and the live run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PredictionRecord:
    """One scored row, kept for out-of-sample predictive diagnostics.

    Every eligible row is recorded -- not only alerts -- because calibration
    measured on the alerted subset alone would be measured on a selected
    sample.

    Attributes:
        model_label: Which fitted model scored the row, so that a refit
            boundary is visible in the diagnostics.
        base_rate: The training-derived positive rate of that model: the
            constant a calibrated probability has to beat on the Brier score.
        terminal_quantiles: Forecast quantiles at the D2 terminal bar, in the
            rebased log target units; ``None`` for B0, which has no forecast.
    """

    variant: str
    origin_session: dt.date
    symbol: str
    model_label: str
    calibrated_probability: float
    estimated_net_return: float
    sigma_2d: float
    base_rate: float
    terminal_quantiles: Mapping[float, float] | None = None


@dataclass(frozen=True)
class ScoredOrigin:
    """The policy outcome for one origin, plus every row that was scored."""

    outcome: PolicyOutcome
    predictions: tuple[PredictionRecord, ...]
    unscorable: int = 0


def score_origin(
    *,
    batch: OriginBatch,
    model: DecisionModel,
    engine: PolicyEngine,
    open_positions: Sequence[OpenPositionView],
    correlation: Callable[[str, str], float | None] | None,
    paused: bool = False,
    suppressed_sectors: frozenset[str] = frozenset(),
    scan_suppressed: bool = False,
    model_label: str = "",
) -> ScoredOrigin:
    """Estimate every scorable row of a batch and apply the alert policy.

    A row the decision model cannot score -- an invalid volatility scale, a
    missing required feature -- is counted and skipped. Only that documented
    failure is caught; anything else is a defect and propagates.
    """
    candidates: list[AlertCandidate] = []
    predictions: list[PredictionRecord] = []
    unscorable = 0
    for score in batch.scorable():
        row = OriginRow(
            origin_session=batch.origin_session,
            symbol=score.symbol,
            features=dict(score.features or {}),
            sigma_2d=score.sigma_2d,
            eligible_count=batch.eligible_count,
        )
        try:
            estimate = model.estimate(row)
        except DecisionError:
            unscorable += 1
            continue
        candidates.append(
            AlertCandidate(
                symbol=score.symbol,
                issuer_id=score.issuer_id,
                sector=score.sector,
                estimate=estimate,
                eligibility=score.eligibility,
                forecast_cache_key=score.forecast_cache_key,
                out_of_sample_observations=model.report.calibration_rows,
                out_of_sample_dates=model.report.calibration_dates,
            )
        )
        terminal: dict[float, float] | None = None
        if score.forecast is not None and score.forecast.usable:
            index = score.forecast.terminal_indices[-1]
            terminal = {
                level: float(path[index]) for level, path in score.forecast.as_mapping().items()
            }
        predictions.append(
            PredictionRecord(
                variant=batch.variant,
                origin_session=batch.origin_session,
                symbol=score.symbol,
                model_label=model_label,
                calibrated_probability=estimate.calibrated_probability,
                estimated_net_return=estimate.estimated_net_return,
                sigma_2d=estimate.sigma_2d,
                base_rate=model.training_base_rate,
                terminal_quantiles=terminal,
            )
        )
    outcome = engine.evaluate(
        origin_session=batch.origin_session,
        candidates=candidates,
        open_positions=open_positions,
        correlation=correlation,
        paused=paused,
        suppressed_sectors=suppressed_sectors | batch.suppressed_sectors,
        scan_suppressed=scan_suppressed or batch.scan_suppressed,
    )
    return ScoredOrigin(outcome=outcome, predictions=tuple(predictions), unscorable=unscorable)


def open_position_views(book: ReferencePortfolio) -> list[OpenPositionView]:
    """The portfolio's open positions as the policy engine sees them."""
    return [
        OpenPositionView(
            symbol=position.symbol,
            issuer_id=position.issuer_id,
            sector=position.sector,
            planned_exit_session=position.planned_exit_session,
        )
        for position in book.positions
    ]


# --------------------------------------------------------------------------- #
# Which model is in force when
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSchedule:
    """Fitted models and the first origin each one scores.

    This is how both registered chronologies are replayed through one loop: the
    development folds (one model per fold, effective from its first validation
    origin) and the prequential final test (one model per refit point).
    """

    entries: tuple[tuple[dt.date, DecisionModel], ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise SimulationError("a model schedule needs at least one fitted model")
        dates = [effective for effective, _model in self.entries]
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            raise SimulationError("model schedule dates must be strictly increasing")

    @classmethod
    def single(cls, model: DecisionModel, effective_from: dt.date = dt.date.min) -> "ModelSchedule":
        return cls(entries=((effective_from, model),))

    def model_for(self, session: dt.date) -> tuple[DecisionModel, str]:
        """The model in force at ``session``, and a label naming it."""
        chosen: tuple[dt.date, DecisionModel] | None = None
        for effective, model in self.entries:
            if effective <= session:
                chosen = (effective, model)
        if chosen is None:
            raise SimulationError(
                f"no fitted model is in force at {session.isoformat()}; the first takes "
                f"effect on {self.entries[0][0].isoformat()}"
            )
        effective, model = chosen
        label = "single" if effective == dt.date.min else f"from {effective.isoformat()}"
        return model, label

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "effective_from": None if effective == dt.date.min else effective.isoformat(),
                "fit_last_session": model.report.fit_last_session.isoformat(),
                "calibration_last_session": model.report.calibration_last_session.isoformat(),
                "fit_dates": model.report.fit_dates,
            }
            for effective, model in self.entries
        ]


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Selection:
    """A symbol chosen after a signal session, to be entered at the next open."""

    symbol: str
    issuer_id: str
    sector: str


@dataclass(frozen=True)
class SelectionOutcome:
    selections: tuple[Selection, ...]
    policy_outcome: PolicyOutcome | None = None
    predictions: tuple[PredictionRecord, ...] = ()


class Selector(Protocol):
    """Decides what to buy after a signal session's close."""

    name: str

    def select(self, session: dt.date, book: ReferencePortfolio) -> SelectionOutcome:  # pragma: no cover
        ...

    def provenance(self) -> dict[str, Any]:  # pragma: no cover
        ...


def cached_batch(
    pipeline: ResearchPipeline,
    variant: str,
    session: dt.date,
    cache: BatchCache | None,
) -> OriginBatch:
    """A batch from ``cache``, building and storing it on a miss."""
    key = (variant, session)
    if cache is not None and key in cache:
        return cache[key]
    batch = pipeline.build_batch(session, variant)
    if cache is not None:
        cache[key] = batch
    return batch


@dataclass
class PolicySelector:
    """A learned variant: its fitted model scores the batch, the policy decides."""

    pipeline: ResearchPipeline
    variant: str
    schedule: ModelSchedule
    batches: BatchCache | None = None
    engine: PolicyEngine | None = None

    def __post_init__(self) -> None:
        variant_spec(self.variant)  # raises for an unregistered variant
        if self.engine is None:
            self.engine = PolicyEngine(self.pipeline.config)

    @property
    def name(self) -> str:
        return self.variant

    def select(self, session: dt.date, book: ReferencePortfolio) -> SelectionOutcome:
        batch = cached_batch(self.pipeline, self.variant, session, self.batches)
        model, label = self.schedule.model_for(session)
        scored = score_origin(
            batch=batch,
            model=model,
            engine=self.engine,  # type: ignore[arg-type]
            open_positions=open_position_views(book),
            correlation=self.pipeline.correlation_lookup(session),
            paused=book.paused,
            model_label=label,
        )
        return SelectionOutcome(
            selections=tuple(
                Selection(
                    symbol=decision.candidate.symbol,
                    issuer_id=decision.candidate.issuer_id,
                    sector=decision.candidate.sector,
                )
                for decision in scored.outcome.alerts
            ),
            policy_outcome=scored.outcome,
            predictions=scored.predictions,
        )

    def provenance(self) -> dict[str, Any]:
        if not variant_spec(self.variant).uses_forecast:
            return {"forecaster": f"none ({self.variant} uses no forecasts)"}
        if self.pipeline.forecaster is None:
            return {"forecaster": "none"}
        return self.pipeline.forecaster.provenance()


@dataclass
class MomentumSelector:
    """The fixed relative-momentum control.

    Reads the same batches as B0 -- the same eligibility mask and the same
    feature ten -- so that the only difference from a candidate is the rule.
    """

    pipeline: ResearchPipeline
    batches: BatchCache | None = None
    reference_variant: str = "B0"
    name: str = "momentum"

    def select(self, session: dt.date, book: ReferencePortfolio) -> SelectionOutcome:
        if book.paused:
            return SelectionOutcome(selections=())
        batch = cached_batch(self.pipeline, self.reference_variant, session, self.batches)
        if batch.scan_suppressed:
            return SelectionOutcome(selections=())
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
            if score.sector not in batch.suppressed_sectors
        ]
        picks = rank_momentum_candidates(
            candidates,
            config=self.pipeline.config,
            open_sectors=book.sector_counts(),
            open_issuers=sorted(book.open_issuers()),
            open_position_count=len(book.positions),
            open_symbols=[position.symbol for position in book.positions],
            correlation=self.pipeline.correlation_lookup(session),
        )
        return SelectionOutcome(
            selections=tuple(
                Selection(symbol=pick.symbol, issuer_id=pick.issuer_id, sector=pick.sector)
                for pick in picks
            )
        )

    def provenance(self) -> dict[str, Any]:
        return {"forecaster": "none (fixed momentum control)"}


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of one simulation for one system."""

    variant: str
    trades: tuple[TradeRecord, ...]
    daily: DailySeries
    portfolio: ReferencePortfolio
    origins_scanned: int
    alerts: int
    no_alert_origins: tuple[dt.date, ...]
    policy_outcomes: tuple[PolicyOutcome, ...]
    predictions: tuple[PredictionRecord, ...] = ()
    forecaster_provenance: Mapping[str, object] = field(default_factory=dict)
    source_provenance: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, int] = field(default_factory=dict)
    model_schedule: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "origins_scanned": self.origins_scanned,
            "alerts": self.alerts,
            "trades": len(self.trades),
            "no_alert_origins": len(self.no_alert_origins),
            "predictions": len(self.predictions),
            "portfolio": self.portfolio.summary(),
            "forecaster": dict(self.forecaster_provenance),
            "source": dict(self.source_provenance),
            "diagnostics": dict(self.diagnostics),
            "model_schedule": [dict(entry) for entry in self.model_schedule],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class _PendingEntry:
    selection: Selection
    signal_session: dt.date
    entry_session: dt.date
    planned_exit_session: dt.date


def simulate(
    *,
    pipeline: ResearchPipeline,
    selector: Selector,
    score_sessions: Sequence[dt.date],
    cost_scenario: str = "base",
    portfolio: ReferencePortfolio | None = None,
    model_schedule: ModelSchedule | None = None,
) -> BacktestResult:
    """Walk the reference portfolio through ``score_sessions``, session by session.

    Raises:
        HoldoutViolation: Before any work, if a score session is reserved and the
            pipeline's guard is locked. Failing fast matters: a partial run that
            touched a reserved origin cannot be un-seen.
        SimulationError: If there is nothing to score.
    """
    if not score_sessions:
        raise SimulationError("no origins to score")
    ordered = sorted(set(score_sessions))
    pipeline.guard.check_all(ordered, purpose=f"{selector.name} scoring")

    config = pipeline.config
    calendar = pipeline.calendar
    horizon_sessions = config.model.horizon_sessions
    final_exit = label_available_at(
        calendar, ordered[-1], horizon_sessions=horizon_sessions
    ).exit_session
    origin_set = set(ordered)

    book = portfolio or ReferencePortfolio(
        config=config, variant=selector.name, cost_scenario=cost_scenario
    )
    pending: dict[dt.date, list[_PendingEntry]] = {}
    outcomes: list[PolicyOutcome] = []
    predictions: list[PredictionRecord] = []
    no_alert: list[dt.date] = []
    trades: list[TradeRecord] = []
    selected = 0
    diagnostics = {
        "entry_price_unavailable": 0,
        "stale_marks": 0,
        "exit_price_unavailable": 0,
        "unit_restatements": 0,
        "unit_change_unresolved": 0,
    }
    #: Each open position's last mark, in its own units, by position id.
    last_marks: dict[str, float] = {}

    for session in calendar.sessions(ordered[0], final_exit):
        view = pipeline.source.view(session)

        # 1. Entries at the official open, sized from then-observable equity.
        due = pending.pop(session, [])
        if due:
            requests: list[EntryRequest] = []
            for entry in due:
                price = _session_price(view, entry.selection.symbol, session, "open")
                if price is None:
                    diagnostics["entry_price_unavailable"] += 1
                    continue
                requests.append(
                    EntryRequest(
                        signal_id=(
                            f"{selector.name}-{entry.signal_session.isoformat()}-"
                            f"{entry.selection.symbol}"
                        ),
                        symbol=entry.selection.symbol,
                        issuer_id=entry.selection.issuer_id,
                        sector=entry.selection.sector,
                        signal_session=entry.signal_session,
                        entry_session=entry.entry_session,
                        planned_exit_session=entry.planned_exit_session,
                        reference_open_price=price,
                        action_ledger=view.action_ledger(entry.selection.symbol),
                    )
                )
            last = book.days[-1] if book.days else None
            book.enter_all(
                requests,
                equity_at_open=last.equity if last else config.portfolio.initial_equity,
                gross_value_at_open=last.gross_exposure * last.equity if last else 0.0,
            )

        # 2. One set of units per position. Each view is split-consistent within
        # itself, but a view after an ex-date can quote post-split prices where
        # the entry was filled from pre-split ones. A position whose units this
        # view cannot reconcile is left unpriced by it.
        unpriced: set[str] = set()
        for position in book.positions:
            factor = _unit_change(view, position)
            if factor is None:
                unpriced.add(position.position_id)
            elif factor != 1.0:
                book.restate_units(position, factor)
                diagnostics["unit_restatements"] += 1
        diagnostics["unit_change_unresolved"] += len(unpriced)

        # 3. Corporate actions with this ex-date, and the session's lows.
        ledgers = {
            position.symbol: view.action_ledger(position.symbol) for position in book.positions
        }
        book.apply_splits(session, ledgers)
        book.credit_dividends(session, ledgers)
        book.record_excursion(
            {
                position.symbol: low
                for position in book.positions
                if position.position_id not in unpriced
                and (low := _session_price(view, position.symbol, session, "low")) is not None
            }
        )

        # 4. Exits at the official close. Only the exit session's own close
        # counts; a missing one leaves the exit unresolved rather than filled
        # at a stale price.
        exit_prices = {
            position.symbol: price
            for position in book.positions
            if position.planned_exit_session == session
            and position.position_id not in unpriced
            and (price := _session_price(view, position.symbol, session, "close")) is not None
        }
        for position in book.close_due(session, exit_prices, ledgers):
            if position.status != "closed":
                diagnostics["exit_price_unavailable"] += 1
            trades.append(_trade_record(config, position, ledgers.get(position.symbol)))

        # 5. The daily mark, which also moves the drawdown pause.
        marks, stale = _mark_prices(view, book, session, unpriced=unpriced, previous=last_marks)
        diagnostics["stale_marks"] += stale
        book.mark(session, marks)
        last_marks = {position.position_id: marks[position.symbol] for position in book.positions}

        # 6. After the close: score this session as a signal origin.
        if session in origin_set:
            outcome = selector.select(session, book)
            if outcome.policy_outcome is not None:
                outcomes.append(outcome.policy_outcome)
            predictions.extend(outcome.predictions)
            if not outcome.selections:
                no_alert.append(session)
            selected += len(outcome.selections)
            timing = label_available_at(calendar, session, horizon_sessions=horizon_sessions)
            for selection in outcome.selections:
                pending.setdefault(timing.entry_session, []).append(
                    _PendingEntry(
                        selection=selection,
                        signal_session=session,
                        entry_session=timing.entry_session,
                        planned_exit_session=timing.exit_session,
                    )
                )

    notes: list[str] = []
    if diagnostics["stale_marks"]:
        notes.append(
            f"{diagnostics['stale_marks']} daily mark(s) used the last known close because "
            "the session's own close was missing"
        )
    if diagnostics["entry_price_unavailable"]:
        notes.append(
            f"{diagnostics['entry_price_unavailable']} selection(s) could not enter: no "
            "official open on the entry session"
        )
    if diagnostics["unit_change_unresolved"]:
        notes.append(
            f"{diagnostics['unit_change_unresolved']} position-session(s) went unpriced: the "
            "view's units differed from the entry's by more than the recorded splits explain"
        )
    return BacktestResult(
        variant=selector.name,
        trades=tuple(trades),
        daily=DailySeries(
            name=selector.name,
            sessions=tuple(book.sessions()),
            returns=np.asarray(book.daily_returns(), dtype=float),
        ),
        portfolio=book,
        origins_scanned=len(ordered),
        alerts=selected,
        no_alert_origins=tuple(no_alert),
        policy_outcomes=tuple(outcomes),
        predictions=tuple(predictions),
        forecaster_provenance=selector.provenance(),
        source_provenance=pipeline.source.describe(),
        diagnostics=dict(diagnostics),
        model_schedule=tuple(model_schedule.describe()) if model_schedule else (),
        notes=tuple(notes),
    )


def _session_price(
    view: MarketView, symbol: str, session: dt.date, column: str
) -> float | None:
    """A reference price for exactly ``session``, or ``None``."""
    try:
        row = view.daily_panel(symbol).row(session)
    except MarketDataError:
        return None
    value = float(row[column])
    return value if math.isfinite(value) and value > 0.0 else None


def _unit_change(view: MarketView, position: PaperPosition) -> float | None:
    """The factor that brings ``position`` into this view's price units.

    Two views need not share units. A point-in-time view before an ex-date
    quotes pre-split prices and one after it post-split prices; so does a view
    that restated a split the provider left raw, against one from before the
    split was in the data. The position's own entry quote, read again from this
    view, measures the change. It is accepted only as *no change* or as
    *exactly the splits the action ledger records inside the hold*, so a vendor
    revision is never taken for a split, nor a split for a return.

    Returns:
        1.0 when the units are unchanged; the factor still to be applied when
        they changed by recorded splits; ``None`` when the change cannot be
        reconciled, or the entry quote is missing from this view.
    """
    quoted = _session_price(view, position.symbol, position.entry_session, "open")
    if quoted is None:
        return None
    measured = position.entry_reference_price / quoted
    if abs(measured - 1.0) <= _UNIT_TOLERANCE:
        return 1.0
    recorded = view.action_ledger(position.symbol).split_factor_in(
        position.entry_session, view.session
    )
    pending = recorded / position.unit_restatement
    if pending != 1.0 and abs(measured / pending - 1.0) <= _UNIT_TOLERANCE:
        return pending
    return None


def _mark_prices(
    view: MarketView,
    book: ReferencePortfolio,
    session: dt.date,
    *,
    unpriced: set[str] | frozenset[str] = frozenset(),
    previous: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    """Closing marks for every open position, and how many were stale.

    A missing close is marked at the last close that *was* observed. Marking at
    the entry fill instead -- the earlier behaviour -- invents a price and hides
    any drawdown since entry, which is exactly when a halted name matters most.
    A position in ``unpriced`` keeps its previous mark: this view's prices are
    not known to be in its units.
    """
    marks: dict[str, float] = {}
    stale = 0
    previous = previous or {}
    for position in book.positions:
        if position.position_id not in unpriced:
            price = _session_price(view, position.symbol, session, "close")
            if price is not None:
                marks[position.symbol] = price
                continue
            closes = view.daily_panel(position.symbol).column("close")
            finite = closes[np.isfinite(closes) & (closes > 0.0)]
            if finite.size:
                stale += 1
                marks[position.symbol] = float(finite[-1])
                continue
        stale += 1
        marks[position.symbol] = previous.get(position.position_id, position.entry_fill_price)
    return marks, stale


def _trade_record(
    config: DesignConfig, position: PaperPosition, ledger: ActionLedger | None
) -> TradeRecord:
    resolved = position.status == "closed"
    scenarios: dict[str, float] = {}
    for scenario in ("base", "stress", "severe"):
        if not resolved or position.exit_reference_price is None or position.exit_session is None:
            scenarios[scenario] = float("nan")
            continue
        if scenario == position.cost_scenario:
            # The portfolio's own accounting, at the scenario it ran under.
            scenarios[scenario] = position.realised_net_return()
            continue
        slippage = config.execution.slippage_fraction(scenario)
        account = label_net_return(
            entry_session=position.entry_session,
            exit_session=position.exit_session,
            entry_open_price=position.entry_reference_price,
            exit_close_price=position.exit_reference_price,
            costs=HoldingCosts(
                buy_slippage=slippage,
                sell_slippage=slippage,
                explicit_fee_fraction=config.execution.explicit_fee_fraction,
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


# --------------------------------------------------------------------------- #
# Convenience façade
# --------------------------------------------------------------------------- #


@dataclass
class WalkForwardRunner:
    """Fits decision models and simulates systems through the one engine."""

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
        batch_cache: BatchCache | None = None,
    ) -> DecisionModel:
        """Fit one model from labelled rows of the two blocks."""
        return fit_decision_model(
            variant=variant,
            fit_rows=self.training_rows(variant, sessions=fit_sessions, batch_cache=batch_cache),
            calibration_rows=self.training_rows(
                variant, sessions=calibration_sessions, batch_cache=batch_cache
            ),
            config=self.config,
            fit_deadline=fit_deadline,
            calibration_deadline=calibration_deadline,
        )

    def training_rows(
        self,
        variant: str,
        *,
        sessions: Sequence[dt.date],
        batch_cache: BatchCache | None = None,
    ) -> list[OriginRow]:
        """Labelled rows for fitting, always at base costs.

        The decision heads are registered on base-cost labels: the minimum
        estimate is a base-cost figure and the stress requirement reprices it.
        Training on stress-cost labels would count the stress step twice, so the
        runner's own cost scenario -- which governs the simulated fills -- does
        not reach the labels.
        """
        rows: list[OriginRow] = []
        for session in sessions:
            batch = cached_batch(self.pipeline, variant, session, batch_cache)
            rows.extend(self.pipeline.labelled_rows(batch, cost_scenario="base"))
        return rows

    def run(
        self,
        variant: str,
        *,
        model: DecisionModel | ModelSchedule,
        score_sessions: Sequence[dt.date],
        portfolio: ReferencePortfolio | None = None,
        batch_cache: BatchCache | None = None,
    ) -> BacktestResult:
        """Simulate a learned variant.

        ``model`` is either one fitted model or a :class:`ModelSchedule`; the
        schedule is how fold boundaries and prequential refits are replayed.
        """
        schedule = model if isinstance(model, ModelSchedule) else ModelSchedule.single(model)
        return simulate(
            pipeline=self.pipeline,
            selector=PolicySelector(
                pipeline=self.pipeline,
                variant=variant,
                schedule=schedule,
                batches=batch_cache,
            ),
            score_sessions=score_sessions,
            cost_scenario=self.cost_scenario,
            portfolio=portfolio,
            model_schedule=schedule,
        )

    def run_momentum_control(
        self,
        *,
        score_sessions: Sequence[dt.date],
        reference_variant: str = "B0",
        batch_cache: BatchCache | None = None,
        name: str = "momentum",
    ) -> BacktestResult:
        """The fixed relative-momentum control, on identical dates and costs."""
        return simulate(
            pipeline=self.pipeline,
            selector=MomentumSelector(
                pipeline=self.pipeline,
                batches=batch_cache,
                reference_variant=reference_variant,
                name=name,
            ),
            score_sessions=score_sessions,
            cost_scenario=self.cost_scenario,
        )
