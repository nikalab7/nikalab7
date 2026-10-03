"""Operational flows: the after-close run, the development comparison, the study
and the single final test.

* :func:`run_after_close` -- one origin, end to end: validate, forecast, score,
  apply the policy, **commit**, then notify. It scores through the same
  :func:`~chronos2_signal.simulation.score_origin` the backtest uses, so a
  backtest describes the code that would run live.
* :func:`run_development_comparison` -- build-order step 5. Every registered
  variant, B0 included, across the development folds, with one portfolio per
  system carried across fold boundaries, the fixed controls on the same dates,
  a paired date-block bootstrap, and the section-12 selection rule -- which may
  legitimately conclude that there is no winner.
* :func:`run_study` -- one candidate against B0 and the fixed controls, with the
  promotion gates applied at a named checkpoint.
* :func:`run_final_test` -- build-order step 6. The reserved origins, replayed
  once through the registered prequential refit schedule.

Every flow that reads outcomes runs under the schedule's holdout guard and
verifies afterwards that no reserved origin was touched; the report states the
result of that check rather than assuming it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calendar_spec import ExchangeCalendar
from .config import DesignConfig
from .decision import DecisionModel
from .evaluation import (
    BootstrapEstimate,
    DailySeries,
    GateReport,
    InformationTest,
    LabelledPrediction,
    PredictiveReport,
    block_profitability,
    contributor_concentration,
    evaluate_promotion_gates,
    information_test,
    paired_block_bootstrap,
    predictive_report,
    trade_metrics,
)
from .forecast_store import ForecastCache
from .holdout import AccessMode
from .market import MarketDataError
from .notifier import Notifier, render_no_alert_notice
from .pipeline import OriginBatch, ResearchPipeline
from .policy import PolicyEngine, PolicyOutcome, dedup_key
from .portfolio import ReferencePortfolio
from .protocol import Fold, ProtocolSchedule, label_available_at
from .provenance import ReleaseManifest, capture_environment
from .simulation import (
    BacktestResult,
    ModelSchedule,
    PredictionRecord,
    WalkForwardRunner,
    open_position_views,
    score_origin,
)
from .sources import SourceError
from .storage import Ledger
from .variants import (
    REGISTERED_VARIANTS,
    MarketLeg,
    cash_series,
    exposure_matched_series,
    variant_spec,
)

__all__ = [
    "OperationsError",
    "AfterCloseResult",
    "run_after_close",
    "SelectionDecision",
    "ComparisonResult",
    "run_development_comparison",
    "StudyResult",
    "run_study",
    "run_final_test",
    "build_release_manifest",
    "render_markdown",
    "render_comparison_markdown",
]


class OperationsError(RuntimeError):
    """Raised when an operational flow cannot proceed."""


def build_release_manifest(
    *,
    config: DesignConfig,
    calendar: ExchangeCalendar,
    universe_manifest_hash: str,
    code_revision: str = "unknown",
    notes: Sequence[str] = (),
) -> ReleaseManifest:
    """Bind data, code, weights, policy and costs into one identity."""
    return ReleaseManifest(
        design_version=config.design_version,
        design_fingerprint=config.fingerprint(),
        code_revision=code_revision,
        checkpoint_id=config.model.id,
        checkpoint_revision=config.model.revision,
        checkpoint_sha256=config.model.safetensors_sha256,
        calendar=calendar.provenance(),
        environment=capture_environment(),
        cost_scenarios={
            "base": config.execution.slippage_fraction("base"),
            "stress": config.execution.slippage_fraction("stress"),
            "severe": config.execution.slippage_fraction("severe"),
        },
        universe_manifest_hash=universe_manifest_hash,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# After-close run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AfterCloseResult:
    """What one after-close run produced."""

    origin_session: dt.date
    variant: str
    batch: OriginBatch
    outcome: PolicyOutcome
    signal_ids: tuple[str, ...]
    notified: int
    run_id: str
    label: str

    def summary(self) -> dict[str, object]:
        return {
            "origin_session": self.origin_session.isoformat(),
            "variant": self.variant,
            "eligible": self.batch.eligible_count,
            "scorable": len(self.batch.scorable()),
            "alerts": self.outcome.alert_count,
            "signals_persisted": len(self.signal_ids),
            "notified": self.notified,
            "output_label": self.label,
            "suppressions": self.outcome.suppression_counts(),
            "origin_notes": list(self.outcome.origin_notes),
        }


def run_after_close(
    *,
    pipeline: ResearchPipeline,
    model: DecisionModel,
    ledger: Ledger,
    origin_session: dt.date,
    variant: str,
    notifier: Notifier | None = None,
    portfolio: ReferencePortfolio | None = None,
    now: dt.datetime | None = None,
    suppressed_sectors: frozenset[str] = frozenset(),
    scan_suppressed: bool = False,
) -> AfterCloseResult:
    """Run one origin: validate, forecast, score, commit, then notify.

    The ordering is load-bearing. Forecasts are written to the ledger when the
    batch is built; every decision for the origin is then committed in one
    transaction; only after that commit is any channel contacted. A retry after
    a delivery failure therefore finds the rows already present and sends
    nothing twice -- including the zero-alert notice.
    """
    config = pipeline.config
    now = now or dt.datetime.now(dt.timezone.utc)
    engine = PolicyEngine(config)
    environment = capture_environment()
    run_id = f"after-close-{variant}-{origin_session.isoformat()}-{uuid.uuid4().hex[:8]}"
    if pipeline.forecast_cache is None:
        # The operational path always records forecasts before outcomes.
        pipeline = dataclasses.replace(pipeline, forecast_cache=ForecastCache(ledger))

    ledger.start_run(
        run_id=run_id,
        run_kind="after_close",
        origin_session=origin_session,
        started_at=now,
        status="running",
        message=None,
        design_fingerprint=config.fingerprint(),
        environment_digest=environment.digest(),
        counts={"variant": variant},
    )

    try:
        batch = pipeline.build_batch(
            origin_session, variant, suppressed_sectors=suppressed_sectors
        )
        scored = score_origin(
            batch=batch,
            model=model,
            engine=engine,
            open_positions=open_position_views(portfolio) if portfolio is not None else [],
            correlation=pipeline.correlation_lookup(origin_session),
            paused=bool(portfolio.paused) if portfolio is not None else False,
            suppressed_sectors=suppressed_sectors,
            scan_suppressed=scan_suppressed,
        )
        outcome = scored.outcome

        timing = label_available_at(
            pipeline.calendar,
            origin_session,
            horizon_sessions=config.model.horizon_sessions,
        )
        data_timestamp = pipeline.calendar.session_close(origin_session).to_pydatetime()
        expires_at = pipeline.calendar.session_open(timing.entry_session).to_pydatetime()
        forecaster_provenance = (
            pipeline.forecaster.provenance()
            if pipeline.forecaster is not None and variant_spec(variant).uses_forecast
            else {"forecaster": "none"}
        )

        # Every decision, alerted or suppressed, lands in one transaction, so a
        # quiet origin is explainable afterwards and a crash cannot leave half
        # of an origin behind.
        signal_ids: list[str] = []
        with ledger.transaction():
            for decision in outcome.decisions:
                candidate = decision.candidate
                key = dedup_key(
                    config.design_version, candidate.issuer_id, origin_session, "2_sessions"
                )
                signal_id = ledger.record_signal(
                    signal_id=f"{config.design_version}-{variant}-{origin_session.isoformat()}"
                    f"-{candidate.symbol}",
                    dedup_key=key if decision.alerted else f"{key}|suppressed|{candidate.symbol}",
                    strategy_version=config.design_version,
                    issuer_id=candidate.issuer_id,
                    symbol=candidate.symbol,
                    sector=candidate.sector,
                    signal_session=origin_session,
                    holding_period="2_sessions",
                    variant=variant,
                    generated_at=now,
                    data_timestamp=data_timestamp,
                    estimated_net_return=candidate.estimate.estimated_net_return,
                    estimated_net_return_stress=candidate.estimate.estimated_net_return_stress,
                    calibrated_probability=candidate.estimate.calibrated_probability,
                    sigma_2d=candidate.estimate.sigma_2d,
                    rank_score=decision.rank_score,
                    rank_position=decision.rank_position,
                    forecast_cache_key=candidate.forecast_cache_key,
                    fit_id=None,
                    decision="alert" if decision.alerted else "suppressed",
                    suppression_reason="; ".join(decision.reasons) or None,
                    output_label=engine.output_label.value,
                    payload=engine.signal_payload(
                        decision,
                        data_timestamp=data_timestamp,
                        forecaster_provenance=forecaster_provenance,
                    ),
                    expires_at=expires_at,
                )
                if decision.alerted:
                    signal_ids.append(signal_id)

        notified = 0
        no_alert_notified = False
        if notifier is not None:
            if outcome.alert_count == 0:
                if not _no_alert_notice_already_sent(ledger, origin_session, variant):
                    subject, body = render_no_alert_notice(
                        origin_session, outcome.origin_notes, engine.output_label.value
                    )
                    notifier.channel.send(subject, body)
                    no_alert_notified = True
            else:
                notified = notifier.notify_session(origin_session, variant=variant, now=now)

        ledger.finish_run(
            run_id,
            status="ok",
            finished_at=dt.datetime.now(dt.timezone.utc),
            counts={
                "variant": variant,
                "eligible": batch.eligible_count,
                "scorable": len(batch.scorable()),
                "alerts": outcome.alert_count,
                "notified": notified,
                "no_alert_notified": no_alert_notified,
            },
        )
        return AfterCloseResult(
            origin_session=origin_session,
            variant=variant,
            batch=batch,
            outcome=outcome,
            signal_ids=tuple(signal_ids),
            notified=notified,
            run_id=run_id,
            label=engine.output_label.value,
        )
    except Exception as exc:
        ledger.finish_run(
            run_id,
            status="failed",
            finished_at=dt.datetime.now(dt.timezone.utc),
            message=f"{type(exc).__name__}: {exc}",
        )
        raise


def _no_alert_notice_already_sent(
    ledger: Ledger, origin_session: dt.date, variant: str
) -> bool:
    """Whether an earlier successful run already delivered the zero-alert notice.

    An origin with no candidates persists no signal rows, so the signals table
    cannot carry this; the run log does.
    """
    rows = ledger.query(
        "SELECT counts FROM run_log WHERE run_kind = 'after_close' "
        "AND origin_session = ? AND status = 'ok'",
        [origin_session.isoformat()],
    )
    for row in rows:
        counts = json.loads(row["counts"]) if row["counts"] else {}
        if counts.get("variant") == variant and counts.get("no_alert_notified"):
            return True
    return False


# --------------------------------------------------------------------------- #
# Shared machinery for the registered comparisons
# --------------------------------------------------------------------------- #


def _guarded(pipeline: ResearchPipeline, schedule: ProtocolSchedule, mode: AccessMode) -> ResearchPipeline:
    """The pipeline under this schedule's holdout guard, in ``mode``.

    Whatever guard the caller configured is replaced: a study must not depend
    on having been handed the right one.
    """
    guard = schedule.guard(AccessMode.DEVELOPMENT)
    if mode is AccessMode.FINAL_TEST:
        guard.unlock_final_test()
    return dataclasses.replace(pipeline, guard=guard)


def _fold_schedule(
    runner: WalkForwardRunner,
    variant: str,
    folds: Sequence[Fold],
    cache: dict,
) -> ModelSchedule:
    """One model per development fold, in force from its first validation origin."""
    entries = []
    for fold in folds:
        model = runner.fit(
            variant,
            fit_sessions=fold.fit_sessions,
            calibration_sessions=fold.calibration_sessions,
            fit_deadline=fold.fit_deadline.to_pydatetime(),
            calibration_deadline=fold.calibration_deadline.to_pydatetime(),
            batch_cache=cache,
        )
        entries.append((fold.validation_sessions[0], model))
    return ModelSchedule(entries=tuple(entries))


def _refit_schedule(
    runner: WalkForwardRunner,
    variant: str,
    schedule: ProtocolSchedule,
    cache: dict,
) -> ModelSchedule:
    """One model per registered refit point, fitted from matured history only."""
    entries = []
    for point in schedule.refit_points:
        deadline = point.deadline.to_pydatetime()
        model = runner.fit(
            variant,
            fit_sessions=point.fit_sessions,
            calibration_sessions=point.calibration_sessions,
            # The fold recipe: fitting runs when calibration begins, so a fit
            # label not yet observable then is dropped rather than shared.
            fit_deadline=(
                point.fit_deadline.to_pydatetime() if point.fit_deadline is not None else deadline
            ),
            calibration_deadline=deadline,
            batch_cache=cache,
        )
        entries.append((point.effective_from, model))
    return ModelSchedule(entries=tuple(entries))


def _label_predictions(
    pipeline: ResearchPipeline, predictions: Sequence[PredictionRecord]
) -> list[LabelledPrediction]:
    """Attach realised outcomes to scored rows, skipping quarantined ones."""
    labelled: list[LabelledPrediction] = []
    for record in predictions:
        account = pipeline.label(record.origin_session, record.symbol)
        if account is None:
            continue
        labelled.append(
            LabelledPrediction(
                origin_session=record.origin_session,
                symbol=record.symbol,
                calibrated_probability=record.calibrated_probability,
                estimated_net_return=record.estimated_net_return,
                sigma_2d=record.sigma_2d,
                base_rate=record.base_rate,
                r_net=account.net_return,
                realised_log_return=pipeline.realised_log_return(
                    record.origin_session, record.symbol
                ),
                terminal_quantiles=record.terminal_quantiles,
            )
        )
    return labelled


def _open_and_close(view: Any, symbol: str, sessions: Sequence[dt.date]) -> tuple[dict, dict]:
    """Official opens and closes of ``symbol`` on ``sessions``, where observed."""
    opens: dict[dt.date, float] = {}
    closes: dict[dt.date, float] = {}
    try:
        panel = view.daily_panel(symbol)
    except (MarketDataError, SourceError):
        return opens, closes
    for session in sessions:
        try:
            row = panel.row(session)
        except MarketDataError:
            continue
        for column, target in (("open", opens), ("close", closes)):
            value = float(row[column])
            if math.isfinite(value) and value > 0.0:
                target[session] = value
    return opens, closes


def _matched_series(
    pipeline: ResearchPipeline, result: BacktestResult, name: str
) -> DailySeries:
    """The exposure-matched market benchmark for ``result``.

    Every price comes from one view at the last session, so the benchmark and
    the position values share one consistently restated series; a return
    stitched from several views could straddle a restatement.
    """
    sessions = list(result.daily.sessions)
    view = pipeline.source.view(sessions[-1])
    market_open, market_close = _open_and_close(view, pipeline.market_proxy or "SPY", sessions)
    days = result.portfolio.days
    initial = pipeline.config.portfolio.initial_equity
    equity_before = {
        session: (days[position - 1].equity if position else initial)
        for position, session in enumerate(sessions)
    }
    legs: list[MarketLeg] = []
    book = result.portfolio
    for held_position in (*book.closed, *book.positions):
        last = held_position.exit_session or held_position.planned_exit_session
        held = [s for s in sessions if held_position.entry_session <= s <= last]
        if not held:
            continue
        stock_open, stock_close = _open_and_close(view, held_position.symbol, held)
        legs.append(MarketLeg(session=held[0], notional=held_position.cost_basis, from_open=True))
        # The market value at each previous close, from price ratios inside this
        # one view -- unit-free, so a split during or after the hold cannot
        # distort it. Without the prices, the committed capital stands in.
        entry_open = stock_open.get(held_position.entry_session)
        fill_premium = held_position.entry_fill_price / held_position.entry_reference_price
        for previous, session in zip(held[:-1], held[1:], strict=True):
            close = stock_close.get(previous)
            notional = (
                held_position.cost_basis / fill_premium * close / entry_open
                if entry_open and close
                else held_position.cost_basis
            )
            legs.append(MarketLeg(session=session, notional=notional, from_open=False))
    return exposure_matched_series(
        name=name,
        sessions=sessions,
        legs=legs,
        equity_before=equity_before,
        market_open=market_open,
        market_close=market_close,
    )


def _block_sensitivity(
    series: Sequence[DailySeries],
    *,
    config: DesignConfig,
    samples: int,
    confidence: float,
    differences: Sequence[tuple[str, str]],
) -> dict[int, dict[str, BootstrapEstimate]]:
    """The same paired bootstrap at each registered sensitivity block length.

    Section 14 registers block length 10 "with 5/20-session sensitivity". These
    intervals are reported beside the primary ones and never replace them: a
    gate reads only the registered block length.
    """
    return {
        block: paired_block_bootstrap(
            series,
            block_sessions=block,
            samples=samples,
            confidence=confidence,
            differences=differences,
        )
        for block in config.validation.bootstrap_block_sensitivity
    }


def _sensitivity_dict(sensitivity: Mapping[int, Mapping[str, BootstrapEstimate]]) -> dict[str, object]:
    return {
        "note": (
            "block-length sensitivity (section 14); reported beside the registered "
            "block length, never used by a gate"
        ),
        "blocks": {
            str(block): {name: estimate.to_dict() for name, estimate in estimates.items()}
            for block, estimates in sorted(sensitivity.items())
        },
    }


def _information_test(
    config: DesignConfig,
    labelled: Mapping[str, Sequence[LabelledPrediction]],
    *,
    samples: int,
    confidence: float,
) -> InformationTest:
    """The registered primary information test: every candidate against B0."""
    return information_test(
        labelled,
        baseline="B0",
        min_rows=config.validation.information_min_common_rows_per_date,
        block_sessions=config.validation.bootstrap_block_sessions,
        samples=samples,
        confidence=confidence,
    )


def _verify_holdout(
    schedule: ProtocolSchedule,
    cache: Mapping[tuple[str, dt.date], OriginBatch],
    results: Sequence[BacktestResult],
    *,
    mode: AccessMode,
) -> str:
    """Check, rather than assume, what a run read from the reserved block."""
    touched = {session for _variant, session in cache}
    for result in results:
        touched |= {record.origin_session for record in result.predictions}
    reserved = set(schedule.test_origins)
    read = touched & reserved
    if mode is AccessMode.DEVELOPMENT:
        if read:
            raise OperationsError(
                f"{len(read)} reserved origin(s) were read in development mode; the "
                "guard should have prevented this and the result is invalid"
            )
        return (
            f"holdout verified: 0 of the {len(reserved)} reserved final-test origins "
            f"were read ({len(touched)} origins touched in total)"
        )
    return (
        f"final-test pass: {len(read)} of the {len(reserved)} reserved origins read, "
        "as registered"
    )


# --------------------------------------------------------------------------- #
# Development comparison (build-order step 5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SelectionDecision:
    """The outcome of the section-12 development selection rule (amended in v2).

    ``selected`` is ``None`` whenever the evidence does not single out one
    candidate. That is a permitted, recorded result -- and in that case C256 is
    retained for continued research *without* being declared superior.

    Attributes:
        eligibility: Per candidate, why it may not be selected under the
            information criterion; empty when it is eligible.
        trading_floors: Per candidate, which trading floors it misses. Reported
            beside the selection; since ``chronos2_hourly_v2`` they do not select.
        transformer_edge_claimable: Whether development evidence shows the
            selected candidate adding information beyond B0, from forecasts the
            frozen checkpoint produced.
    """

    selected: str | None
    retained_for_research: str
    transformer_edge_claimable: bool
    eligibility: Mapping[str, tuple[str, ...]]
    ranking: tuple[tuple[str, float], ...]
    statement: str
    trading_floors: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected,
            "retained_for_research": self.retained_for_research,
            "transformer_edge_claimable": self.transformer_edge_claimable,
            "eligibility": {name: list(reasons) for name, reasons in self.eligibility.items()},
            "ranking": [list(item) for item in self.ranking],
            "statement": self.statement,
            "trading_floors": {
                name: list(reasons) for name, reasons in self.trading_floors.items()
            },
        }


@dataclass(frozen=True)
class ComparisonResult:
    """Every registered system on one shared set of development dates."""

    variants: tuple[str, ...]
    folds: tuple[int, ...]
    systems: Mapping[str, BacktestResult]
    controls: Mapping[str, DailySeries]
    bootstrap: Mapping[str, BootstrapEstimate]
    predictive: Mapping[str, PredictiveReport]
    selection: SelectionDecision
    manifest: ReleaseManifest
    notes: tuple[str, ...] = ()
    bootstrap_sensitivity: Mapping[int, Mapping[str, BootstrapEstimate]] = field(
        default_factory=dict
    )
    #: The primary information test (amended section 12) that made the selection.
    information: InformationTest | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "variants": list(self.variants),
            "information_test": self.information.to_dict() if self.information else None,
            "folds": list(self.folds),
            "systems": {name: result.summary() for name, result in self.systems.items()},
            "trade_metrics": {
                name: trade_metrics(result.trades, origins_scanned=result.origins_scanned).to_dict()
                for name, result in self.systems.items()
            },
            "controls": {
                name: {"mean_daily_return": series.mean, "sessions": len(series)}
                for name, series in self.controls.items()
            },
            "bootstrap": {name: estimate.to_dict() for name, estimate in self.bootstrap.items()},
            "bootstrap_block_sensitivity": _sensitivity_dict(self.bootstrap_sensitivity),
            "predictive": {name: report.to_dict() for name, report in self.predictive.items()},
            "selection": self.selection.to_dict(),
            "release_manifest": self.manifest.to_dict(),
            "notes": list(self.notes),
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return target


def select_candidate(
    *,
    config: DesignConfig,
    information: InformationTest,
    systems: Mapping[str, BacktestResult],
) -> SelectionDecision:
    """Apply the section-12 development selection rule, as amended in v2.

    The question is whether a Chronos candidate ranks stocks better than B0, so
    it is answered with the paired information test rather than with the
    three-position portfolio, which cannot answer it at these sample sizes.

    1. A Chronos candidate is eligible with at least the registered number of
       development dates carrying common rows, a positive mean daily rank-IC
       gain over B0, and a gain that survives removing its single best date.
    2. Eligible candidates are ranked by the 90% date-block-bootstrap lower bound
       of that gain.
    3. If the leader's IC advantage over the runner-up is inconclusive, there is
       no winner and C256 is retained for continued research without being
       declared superior.
    4. Development evidence that Chronos-2 adds information requires the
       selected candidate's lower bound against B0 to be above zero -- and the
       frozen checkpoint to have produced its forecasts. A fixture's numbers
       are plumbing, whatever they show.

    The trading floors are still computed for every candidate and reported;
    they no longer select.
    """
    validation = config.validation
    candidates = [
        name
        for name in information.candidates
        if name in REGISTERED_VARIANTS and REGISTERED_VARIANTS[name].uses_forecast
    ]
    eligibility: dict[str, tuple[str, ...]] = {}
    for name in candidates:
        reasons: list[str] = []
        if information.dates < validation.development_min_information_dates:
            reasons.append(
                f"{information.dates} development dates with "
                f"{information.min_rows}+ common rows, fewer than "
                f"{validation.development_min_information_dates}"
            )
        gain = information.gain(name)
        if gain is None or not gain.point > 0.0:
            reasons.append("mean daily rank-IC gain over B0 is not positive")
        if not information.gain_excluding_best_date(name) > 0.0:
            reasons.append("the gain does not survive removing the single best date")
        if gain is not None and not math.isfinite(gain.lower):
            reasons.append(
                f"{gain.n_sessions} dates are too few to resample in "
                f"{gain.block_sessions}-session blocks"
            )
        eligibility[name] = tuple(reasons)

    trading_floors = {
        name: _trading_floor_failures(config, systems[name]) for name in candidates if name in systems
    }

    eligible = [name for name in candidates if not eligibility[name]]
    ranking = tuple(
        sorted(
            ((name, information.gain(name).lower) for name in eligible),  # type: ignore[union-attr]
            key=lambda item: (-item[1], item[0]),
        )
    )
    confidence = f"{information.confidence:.0%}"

    def no_winner(statement: str) -> SelectionDecision:
        return SelectionDecision(
            selected=None,
            retained_for_research="C256",
            transformer_edge_claimable=False,
            eligibility=eligibility,
            ranking=ranking,
            statement=statement
            + " C256 is retained for continued research without being declared superior.",
            trading_floors=trading_floors,
        )

    if not eligible:
        return no_winner("No Chronos candidate met the information criterion's floors.")
    leader = ranking[0][0]
    if len(ranking) > 1:
        runner_up = ranking[1][0]
        gap = information.bootstrap.get(f"{leader}_minus_{runner_up}")
        if gap is None or not gap.lower > 0.0:
            return no_winner(
                f"{leader} and {runner_up} rank stocks indistinguishably well at "
                f"{confidence}."
            )
    separated = information.demonstrated(leader)
    from_checkpoint = leader in systems and _from_checkpoint(systems[leader])
    if separated and from_checkpoint:
        conclusion = (
            "That lower bound is above zero: development evidence that Chronos-2 adds "
            "information, for the final test to confirm."
        )
    elif separated:
        conclusion = (
            "That lower bound is above zero, but a test fixture rather than the frozen "
            "checkpoint produced the forecasts, so nothing is claimed for Chronos-2."
        )
    else:
        conclusion = (
            "That lower bound is not above zero, so no Chronos-specific information is "
            "claimed."
        )
    return SelectionDecision(
        selected=leader,
        retained_for_research=leader,
        transformer_edge_claimable=separated and from_checkpoint,
        eligibility=eligibility,
        ranking=ranking,
        statement=(
            f"{leader} selected on the {confidence} lower bound of its rank-IC gain over "
            f"B0. {conclusion}"
        ),
        trading_floors=trading_floors,
    )


def _from_checkpoint(result: BacktestResult) -> bool:
    """Whether the frozen checkpoint, not a fixture, produced a run's forecasts."""
    return result.forecaster_provenance.get("is_frozen_checkpoint") is True


def _trading_floor_failures(config: DesignConfig, result: BacktestResult) -> tuple[str, ...]:
    """Which section-12 trading floors a development system misses. Reported only."""
    validation = config.validation
    metrics = trade_metrics(result.trades, origins_scanned=result.origins_scanned)
    reasons: list[str] = []
    if metrics.trades < validation.development_min_executed_trades:
        reasons.append(
            f"{metrics.trades} executed trades, fewer than "
            f"{validation.development_min_executed_trades}"
        )
    if metrics.distinct_sessions < validation.development_min_signal_sessions:
        reasons.append(
            f"{metrics.distinct_sessions} distinct signal sessions, fewer than "
            f"{validation.development_min_signal_sessions}"
        )
    if not (metrics.mean_net_return > 0.0):
        reasons.append("mean net return at base costs is not positive")
    if not (metrics.stress_mean_net_return > 0.0):
        reasons.append("mean net return at stress costs is not positive")
    if not contributor_concentration(list(result.trades)).get("profitable_excluding_best_trade"):
        reasons.append("profit does not survive removing the single best trade")
    return tuple(reasons)


def run_development_comparison(
    *,
    pipeline: ResearchPipeline,
    schedule: ProtocolSchedule,
    variants: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
    cost_scenario: str = "base",
    bootstrap_samples: int | None = None,
    code_revision: str = "unknown",
    notes: Sequence[str] = (),
) -> ComparisonResult:
    """Build-order step 5: the registered development comparison.

    Args:
        variants: Learned variants to compare. Defaults to all five registered
            ones; B0 is always included because it is the baseline the central
            question is asked against.
        folds: Development folds to use. Defaults to all of them.

    Raises:
        OperationsError: If there are no folds, or a reserved origin was read.
    """
    config = pipeline.config
    if not schedule.folds:
        raise OperationsError(
            "the schedule contains no development folds; collect more dates rather "
            "than shortening a block"
        )
    chosen_folds = tuple(range(len(schedule.folds))) if folds is None else tuple(folds)
    for index in chosen_folds:
        if not 0 <= index < len(schedule.folds):
            raise OperationsError(f"fold {index} outside the {len(schedule.folds)} available")
    fold_objects = [schedule.folds[index] for index in chosen_folds]
    names = tuple(dict.fromkeys(["B0", *(variants or config.validation.initial_learned_variants)]))
    for name in names:
        variant_spec(name)

    guarded = _guarded(pipeline, schedule, AccessMode.DEVELOPMENT)
    runner = WalkForwardRunner(pipeline=guarded, cost_scenario=cost_scenario)
    cache: dict = {}
    score_sessions = sorted({s for fold in fold_objects for s in fold.validation_sessions})

    systems: dict[str, BacktestResult] = {}
    for name in names:
        systems[name] = runner.run(
            name,
            model=_fold_schedule(runner, name, fold_objects, cache),
            score_sessions=score_sessions,
            batch_cache=cache,
        )
    systems["momentum"] = runner.run_momentum_control(
        score_sessions=score_sessions, batch_cache=cache
    )

    controls: dict[str, DailySeries] = {"cash": cash_series("cash", systems["B0"].daily.sessions)}
    for name in names:
        controls[f"{name}_spy_matched"] = _matched_series(guarded, systems[name], f"{name}_spy_matched")

    candidates = [name for name in names if REGISTERED_VARIANTS[name].uses_forecast]
    differences: list[tuple[str, str]] = []
    for name in names:
        differences += [(name, "momentum"), (name, f"{name}_spy_matched"), (name, "cash")]
        if name != "B0":
            differences.append((name, "B0"))
    differences += [(a, b) for a in candidates for b in candidates if a != b]

    compared = [*(result.daily for result in systems.values()), *controls.values()]
    bootstrap = paired_block_bootstrap(
        compared,
        block_sessions=config.validation.bootstrap_block_sessions,
        samples=bootstrap_samples or config.validation.bootstrap_samples,
        confidence=config.validation.development_selection_bootstrap_confidence,
        differences=differences,
    )
    sensitivity = _block_sensitivity(
        compared,
        config=config,
        samples=bootstrap_samples or config.validation.bootstrap_samples,
        confidence=config.validation.development_selection_bootstrap_confidence,
        differences=differences,
    )
    labelled = {name: _label_predictions(guarded, systems[name].predictions) for name in names}
    predictive = {
        name: predictive_report(labelled[name], quantile_levels=config.model.quantile_levels)
        for name in names
    }
    information = _information_test(
        config,
        labelled,
        samples=bootstrap_samples or config.validation.bootstrap_samples,
        confidence=config.validation.development_selection_bootstrap_confidence,
    )
    selection = select_candidate(config=config, information=information, systems=systems)
    holdout_note = _verify_holdout(
        schedule, cache, list(systems.values()), mode=AccessMode.DEVELOPMENT
    )
    manifest = build_release_manifest(
        config=config,
        calendar=pipeline.calendar,
        universe_manifest_hash=pipeline.watchlist.hash(),
        code_revision=code_revision,
        notes=(*notes, *_forecaster_notes(systems.values())),
    )
    return ComparisonResult(
        variants=names,
        folds=chosen_folds,
        systems=systems,
        controls=controls,
        bootstrap=bootstrap,
        predictive=predictive,
        selection=selection,
        manifest=manifest,
        bootstrap_sensitivity=sensitivity,
        information=information,
        notes=(
            *notes,
            *_forecaster_notes(systems.values()),
            *schedule.notes,
            holdout_note,
            f"{len(chosen_folds)} development fold(s); one portfolio per system carried "
            "across fold boundaries",
            *_source_notes(guarded),
        ),
    )


# --------------------------------------------------------------------------- #
# Single-candidate study and the final test
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StudyResult:
    """One candidate against B0 and the fixed controls, on one date index."""

    variant: str
    checkpoint: str
    candidate: BacktestResult
    baseline: BacktestResult | None
    controls: Mapping[str, DailySeries]
    bootstrap: Mapping[str, BootstrapEstimate]
    predictive: PredictiveReport
    gates: GateReport
    manifest: ReleaseManifest
    folds: tuple[Fold, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
    bootstrap_sensitivity: Mapping[int, Mapping[str, BootstrapEstimate]] = field(
        default_factory=dict
    )
    #: The primary information test against B0 (amended section 14); ``None``
    #: when the candidate is B0 itself.
    information: InformationTest | None = None

    @property
    def information_demonstrated(self) -> bool:
        """Whether this study shows the candidate adding information beyond B0.

        Requires the frozen checkpoint to have produced the forecasts: with a
        fixture the statistic is still reported, but nothing is claimed.
        """
        return bool(
            self.information is not None
            and self.information.demonstrated(self.variant)
            and _from_checkpoint(self.candidate)
        )

    def to_dict(self) -> dict[str, object]:
        metrics = trade_metrics(
            self.candidate.trades, origins_scanned=self.candidate.origins_scanned
        )
        return {
            "variant": self.variant,
            "checkpoint": self.checkpoint,
            "output_label": self.gates.status_label,
            "information_test": self.information.to_dict() if self.information else None,
            "information_demonstrated": self.information_demonstrated,
            "candidate": self.candidate.summary(),
            "baseline": self.baseline.summary() if self.baseline else None,
            "trade_metrics": metrics.to_dict(),
            "alert_coverage": _alert_coverage(self.candidate),
            "controls": {
                name: {"mean_daily_return": series.mean, "sessions": len(series)}
                for name, series in self.controls.items()
            },
            "bootstrap": {
                name: estimate.to_dict() for name, estimate in self.bootstrap.items()
            },
            "bootstrap_block_sensitivity": _sensitivity_dict(self.bootstrap_sensitivity),
            "predictive": self.predictive.to_dict(),
            "block_profitability": block_profitability(
                self.candidate.daily, block_sessions=20
            ),
            "concentration": contributor_concentration(list(self.candidate.trades)),
            "gates": self.gates.to_dict(),
            "release_manifest": self.manifest.to_dict(),
            "folds": [fold.describe() for fold in self.folds],
            "notes": list(self.notes),
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return target


def _assemble_study(
    *,
    pipeline: ResearchPipeline,
    schedule: ProtocolSchedule,
    variant: str,
    checkpoint: str,
    candidate: BacktestResult,
    baseline: BacktestResult | None,
    momentum: BacktestResult,
    cache: Mapping,
    mode: AccessMode,
    folds: Sequence[Fold],
    bootstrap_samples: int | None,
    code_revision: str,
    notes: Sequence[str],
) -> StudyResult:
    config = pipeline.config
    matched = _matched_series(pipeline, candidate, "spy_exposure_matched")
    cash = cash_series("cash", candidate.daily.sessions)
    for other in (momentum, baseline):
        if other is not None and other.daily.sessions != candidate.daily.sessions:
            raise OperationsError(
                f"{other.variant} and {candidate.variant} date indices diverged; the "
                "comparison would be between samples rather than between systems"
            )

    systems = [candidate.daily, momentum.daily, matched, cash]
    differences = [
        (candidate.variant, "momentum"),
        (candidate.variant, "spy_exposure_matched"),
        (candidate.variant, "cash"),
    ]
    if baseline is not None:
        systems.append(baseline.daily)
        differences.append((candidate.variant, baseline.variant))
    bootstrap = paired_block_bootstrap(
        systems,
        block_sessions=config.validation.bootstrap_block_sessions,
        samples=bootstrap_samples or config.validation.bootstrap_samples,
        confidence=config.validation.promotion_bootstrap_confidence,
        differences=differences,
    )
    sensitivity = _block_sensitivity(
        systems,
        config=config,
        samples=bootstrap_samples or config.validation.bootstrap_samples,
        confidence=config.validation.promotion_bootstrap_confidence,
        differences=differences,
    )
    candidate_rows = _label_predictions(pipeline, candidate.predictions)
    predictive = predictive_report(candidate_rows, quantile_levels=config.model.quantile_levels)
    # The primary research question, separate from the trading gates: does the
    # candidate rank stocks better than B0 on the same rows and dates?
    information = (
        _information_test(
            config,
            {
                candidate.variant: candidate_rows,
                "B0": _label_predictions(pipeline, baseline.predictions),
            },
            samples=bootstrap_samples or config.validation.bootstrap_samples,
            confidence=config.validation.promotion_bootstrap_confidence,
        )
        if baseline is not None
        else None
    )
    controls = {momentum.daily.name: momentum.daily, matched.name: matched, cash.name: cash}
    baselines: dict[str, DailySeries] = dict(controls)
    if baseline is not None:
        baselines[baseline.variant] = baseline.daily

    gates = evaluate_promotion_gates(
        checkpoint=checkpoint,
        historical_trades=list(candidate.trades),
        forward_trades=[],
        forward_sessions=0,
        portfolio_series=candidate.daily,
        baseline_series=baselines,
        bootstrap=bootstrap,
        calibration_brier=predictive.brier,
        base_rate_brier=predictive.base_rate_brier,
        coverage_p10_p90=predictive.coverage_p10_p90,
        reliability=list(predictive.reliability),
        config_min_forward_sessions=config.validation.forward_min_sessions,
        config_min_forward_signals=config.validation.forward_min_executed_signals,
        config_min_combined_signals=config.validation.combined_evaluation_min_executed_signals,
        config_min_combined_sessions=config.validation.combined_evaluation_min_signal_sessions,
        config_min_profitable_blocks=config.validation.promotion_min_profitable_blocks,
        config_block_sessions=config.validation.promotion_block_sessions,
        historical_origins_scanned=candidate.origins_scanned,
    )
    results = [candidate, momentum] + ([baseline] if baseline is not None else [])
    holdout_note = _verify_holdout(schedule, cache, results, mode=mode)
    forecaster_notes = _forecaster_notes([candidate])
    manifest = build_release_manifest(
        config=config,
        calendar=pipeline.calendar,
        universe_manifest_hash=pipeline.watchlist.hash(),
        code_revision=code_revision,
        notes=(*notes, *forecaster_notes),
    )
    return StudyResult(
        variant=variant,
        checkpoint=checkpoint,
        candidate=candidate,
        baseline=baseline,
        controls=controls,
        bootstrap=bootstrap,
        predictive=predictive,
        gates=gates,
        manifest=manifest,
        folds=tuple(folds),
        bootstrap_sensitivity=sensitivity,
        information=information,
        notes=(
            # The caller's notes lead: a warning about what the data is belongs
            # at the top of what a reader looks at, not only in the manifest.
            *forecaster_notes,
            *notes,
            *schedule.notes,
            holdout_note,
            *_source_notes(pipeline),
        ),
    )


def run_study(
    *,
    pipeline: ResearchPipeline,
    schedule: ProtocolSchedule,
    variant: str,
    checkpoint: str,
    folds: Sequence[int] | None = None,
    cost_scenario: str = "base",
    bootstrap_samples: int | None = None,
    code_revision: str = "unknown",
    notes: Sequence[str] = (),
) -> StudyResult:
    """One candidate against B0 and the fixed controls, on the development folds.

    Raises:
        OperationsError: If the schedule has no folds, or a reserved origin was
            read.
    """
    if not schedule.folds:
        raise OperationsError(
            "the schedule contains no development folds; collect more dates rather "
            "than shortening a block"
        )
    chosen = tuple(range(len(schedule.folds))) if folds is None else tuple(folds)
    for index in chosen:
        if not 0 <= index < len(schedule.folds):
            raise OperationsError(f"fold {index} outside the {len(schedule.folds)} available")
    fold_objects = [schedule.folds[index] for index in chosen]
    variant_spec(variant)

    guarded = _guarded(pipeline, schedule, AccessMode.DEVELOPMENT)
    runner = WalkForwardRunner(pipeline=guarded, cost_scenario=cost_scenario)
    cache: dict = {}
    score_sessions = sorted({s for fold in fold_objects for s in fold.validation_sessions})

    candidate = runner.run(
        variant,
        model=_fold_schedule(runner, variant, fold_objects, cache),
        score_sessions=score_sessions,
        batch_cache=cache,
    )
    baseline = None
    if variant != "B0":
        baseline = runner.run(
            "B0",
            model=_fold_schedule(runner, "B0", fold_objects, cache),
            score_sessions=score_sessions,
            batch_cache=cache,
        )
    momentum = runner.run_momentum_control(score_sessions=score_sessions, batch_cache=cache)
    return _assemble_study(
        pipeline=guarded,
        schedule=schedule,
        variant=variant,
        checkpoint=checkpoint,
        candidate=candidate,
        baseline=baseline,
        momentum=momentum,
        cache=cache,
        mode=AccessMode.DEVELOPMENT,
        folds=fold_objects,
        bootstrap_samples=bootstrap_samples,
        code_revision=code_revision,
        notes=(*notes, f"{len(fold_objects)} development fold(s); this is not a multi-year validation"),
    )


def run_final_test(
    *,
    pipeline: ResearchPipeline,
    schedule: ProtocolSchedule,
    variant: str,
    checkpoint: str = "final_historical_test",
    cost_scenario: str = "base",
    bootstrap_samples: int | None = None,
    code_revision: str = "unknown",
    notes: Sequence[str] = (),
) -> StudyResult:
    """Build-order step 6: the reserved origins, evaluated through the refit schedule.

    Each registered refit point fits both heads from history that had matured by
    its deadline -- including earlier test outcomes once they were observable --
    and scores the origins until the next refit. The schedule is fixed by the
    protocol before this runs; nothing here depends on results.

    Raises:
        OperationsError: If the refit schedule does not start at the first test
            origin, so part of the test block would have no model in force.
    """
    if not schedule.test_origins:
        raise OperationsError("the schedule reserves no final-test origins")
    if not schedule.refit_points or schedule.refit_points[0].effective_from != schedule.test_origins[0]:
        raise OperationsError(
            "the registered refit schedule does not cover the first test origin; the "
            "final test cannot be run without a model in force from its start"
        )
    variant_spec(variant)
    guarded = _guarded(pipeline, schedule, AccessMode.FINAL_TEST)
    runner = WalkForwardRunner(pipeline=guarded, cost_scenario=cost_scenario)
    cache: dict = {}
    candidate = runner.run(
        variant,
        model=_refit_schedule(runner, variant, schedule, cache),
        score_sessions=schedule.test_origins,
        batch_cache=cache,
    )
    baseline = None
    if variant != "B0":
        baseline = runner.run(
            "B0",
            model=_refit_schedule(runner, "B0", schedule, cache),
            score_sessions=schedule.test_origins,
            batch_cache=cache,
        )
    momentum = runner.run_momentum_control(
        score_sessions=schedule.test_origins, batch_cache=cache
    )
    return _assemble_study(
        pipeline=guarded,
        schedule=schedule,
        variant=variant,
        checkpoint=checkpoint,
        candidate=candidate,
        baseline=baseline,
        momentum=momentum,
        cache=cache,
        mode=AccessMode.FINAL_TEST,
        folds=(),
        bootstrap_samples=bootstrap_samples,
        code_revision=code_revision,
        notes=(
            *notes,
            f"final historical test over {len(schedule.test_origins)} reserved origins, "
            f"{len(schedule.refit_points)} registered refit point(s)",
            "a failed final test cannot be relabelled development and reused",
        ),
    )


def _forecaster_notes(results: Any) -> tuple[str, ...]:
    for result in results:
        if result.forecaster_provenance.get("is_frozen_checkpoint") is False:
            return (
                "produced with a non-model forecaster fixture: no predictive content, "
                "plumbing only",
            )
    return ()


def _source_notes(pipeline: ResearchPipeline) -> tuple[str, ...]:
    limitation = pipeline.source.describe().get("limitation")
    return (str(limitation),) if limitation else ()


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def render_markdown(result: StudyResult) -> str:
    """A compact, honest report.

    Coverage, trade counts and no-alert dates are reported alongside returns,
    and the label is whatever the gates actually support.
    """
    metrics = trade_metrics(
        result.candidate.trades, origins_scanned=result.candidate.origins_scanned
    )
    predictive = result.predictive
    lines = [
        f"# {result.variant} - {result.checkpoint}",
        "",
        f"**Adds information beyond B0:** {_information_verdict(result)}",
        f"**Label:** {result.gates.status_label}",
        f"**Design:** {result.manifest.design_version} "
        f"(fingerprint `{result.manifest.design_fingerprint[:12]}`)",
        f"**Forecaster:** {result.candidate.forecaster_provenance.get('forecaster')}",
        "",
        *_information_lines(result.information, result.variant),
        "## Coverage",
        "",
        f"- Origins scanned: {metrics.origins_scanned}",
        f"- Origins with alerts: "
        f"{result.candidate.origins_scanned - len(result.candidate.no_alert_origins)}",
        f"- Origins with no alert: {len(result.candidate.no_alert_origins)}",
        f"- Origins with an executed trade: {metrics.origins_with_trades}",
        f"- Executed trades: {metrics.trades} "
        f"(unresolved exits: {metrics.unresolved_exits})",
        f"- Distinct signal sessions: {metrics.distinct_sessions}",
        "",
        "## Returns after costs",
        "",
        f"- Mean net trade return (base): {_fmt(metrics.mean_net_return)}",
        f"- Mean net trade return (stress): {_fmt(metrics.stress_mean_net_return)}",
        f"- Mean net trade return (severe): {_fmt(metrics.severe_mean_net_return)}",
        f"- Win rate: {_fmt(metrics.win_rate)}",
        f"- Portfolio max drawdown: "
        f"{_fmt(result.candidate.portfolio.summary()['max_drawdown'])}",
        f"- Mean gross exposure: "
        f"{_fmt(result.candidate.portfolio.summary()['mean_gross_exposure'])}",
        "",
        "## Baseline and controls on the same dates, fills and costs",
        "",
    ]
    if result.baseline is not None:
        lines.append(
            f"- {result.baseline.variant}: mean daily net return {_fmt(result.baseline.daily.mean)}"
        )
    for name, series in sorted(result.controls.items()):
        lines.append(f"- {name}: mean daily net return {_fmt(series.mean)}")
    lines.extend(["", "## Predictive diagnostics (out of sample)", ""])
    lines.extend(
        [
            f"- Scored rows: {predictive.rows} across {predictive.dates} dates",
            f"- Brier: {_fmt(predictive.brier)} against the training base rate's "
            f"{_fmt(predictive.base_rate_brier)}",
            f"- p10-p90 coverage: {_fmt(predictive.coverage_p10_p90)} (nominal 0.80)",
            f"- Terminal pinball: {_fmt(predictive.pinball_terminal)} against a "
            f"trailing-volatility reference of {_fmt(predictive.pinball_trailing_vol_reference)}",
            f"- Median absolute error: {_fmt(predictive.median_abs_error)} against "
            f"persistence's {_fmt(predictive.persistence_abs_error)}",
            f"- Mean cross-sectional rank IC: {_fmt(predictive.rank_ic_mean)} over "
            f"{predictive.rank_ic_dates} dates",
        ]
    )
    lines.extend(["", "## Date-block bootstrap", ""])
    for name, estimate in sorted(result.bootstrap.items()):
        lines.append(
            f"- {name}: point {_fmt(estimate.point)}, "
            f"{estimate.confidence:.0%} interval "
            f"[{_fmt(estimate.lower)}, {_fmt(estimate.upper)}] "
            f"over {estimate.n_sessions} sessions, "
            f"block {estimate.block_sessions}"
        )
    if result.bootstrap_sensitivity:
        lines.extend(["", "## Block-length sensitivity (reported, not gated)", ""])
        for block, estimates in sorted(result.bootstrap_sensitivity.items()):
            own = estimates.get(result.variant)
            if own is not None:
                lines.append(
                    f"- block {block}: {result.variant} "
                    f"[{_fmt(own.lower)}, {_fmt(own.upper)}] at {own.confidence:.0%}"
                )
    lines.extend(["", "## Promotion gates", ""])
    for gate in result.gates.results:
        mark = "PASS" if gate.passed else "FAIL"
        lines.append(f"- [{mark}] {gate.number}. {gate.name} - {gate.detail}")
    lines.extend(["", "## Notes", ""])
    for note in (*result.notes, *result.gates.notes):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def render_comparison_markdown(result: ComparisonResult) -> str:
    """The development comparison, led by what the selection rule concluded."""
    selection = result.selection
    lines = [
        f"# Development comparison - folds {', '.join(map(str, result.folds))}",
        "",
        f"**Selection:** {selection.statement}",
        f"**Transformer edge claimable:** {selection.transformer_edge_claimable}",
        "",
        *(
            _information_lines(result.information, None)
            if result.information is not None
            else []
        ),
        "## Systems on identical dates, fills and costs",
        "",
        "| System | Trades | Sessions | Mean net (base) | Mean net (stress) | Mean daily |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, backtest in result.systems.items():
        metrics = trade_metrics(backtest.trades, origins_scanned=backtest.origins_scanned)
        lines.append(
            f"| {name} | {metrics.trades} | {metrics.distinct_sessions} | "
            f"{_fmt(metrics.mean_net_return)} | {_fmt(metrics.stress_mean_net_return)} | "
            f"{_fmt(backtest.daily.mean)} |"
        )
    lines.extend(["", "## Eligibility for selection", ""])
    for name, reasons in selection.eligibility.items():
        lines.append(f"- {name}: {'eligible' if not reasons else '; '.join(reasons)}")
    lines.extend(["", "## Trading floors (reported; since v2 they do not select)", ""])
    for name, reasons in selection.trading_floors.items():
        lines.append(f"- {name}: {'all met' if not reasons else '; '.join(reasons)}")
    lines.extend(["", "## Predictive diagnostics", ""])
    for name, report in result.predictive.items():
        lines.append(
            f"- {name}: Brier {_fmt(report.brier)} vs base rate {_fmt(report.base_rate_brier)}; "
            f"coverage {_fmt(report.coverage_p10_p90)}; rank IC {_fmt(report.rank_ic_mean)}"
        )
    lines.extend(["", "## Date-block bootstrap", ""])
    for name, estimate in sorted(result.bootstrap.items()):
        lines.append(
            f"- {name}: [{_fmt(estimate.lower)}, {_fmt(estimate.upper)}] at "
            f"{estimate.confidence:.0%}"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def _alert_coverage(result: BacktestResult) -> dict[str, float | int]:
    """Section 14's alert coverage, from the selector's own record of alerts."""
    with_alerts = result.origins_scanned - len(result.no_alert_origins)
    return {
        "origins_scanned": result.origins_scanned,
        "origins_with_alerts": with_alerts,
        "no_alert_origins": len(result.no_alert_origins),
        "alert_coverage": (with_alerts / result.origins_scanned) if result.origins_scanned else float("nan"),
    }


def _information_verdict(result: StudyResult) -> str:
    if result.information is None:
        return "not tested (the candidate is B0)"
    estimate = result.information.gain(result.variant)
    if estimate is None:
        return "not measurable (no dates with enough common rows)"
    if not math.isfinite(estimate.lower):
        return (
            f"not measurable ({estimate.n_sessions} dates are too few to resample in "
            f"{estimate.block_sessions}-session blocks)"
        )
    bound = (
        f"{estimate.confidence:.0%} lower bound of the mean daily rank-IC gain: "
        f"{_fmt(estimate.lower)}"
    )
    if result.information_demonstrated:
        return f"demonstrated ({bound})"
    if result.information.demonstrated(result.variant):
        return f"not claimed: a test fixture produced the forecasts ({bound})"
    return f"not demonstrated ({bound})"


def _information_lines(information: InformationTest | None, only: str | None) -> list[str]:
    """The primary information test, as report lines."""
    if information is None:
        return []
    lines = [
        "## Primary information test: rank IC against B0",
        "",
        f"- {information.dates} dates with at least {information.min_rows} rows every "
        "compared system scored",
        f"- B0 mean daily rank IC: {_fmt(information.mean_ic(information.baseline))}",
    ]
    for name in information.candidates:
        if only is not None and name != only:
            continue
        estimate = information.gain(name)
        if estimate is None:
            lines.append(f"- {name}: not measurable")
            continue
        lines.append(
            f"- {name}: mean daily rank IC {_fmt(information.mean_ic(name))}; gain over B0 "
            f"{_fmt(estimate.point)}, {estimate.confidence:.0%} interval "
            f"[{_fmt(estimate.lower)}, {_fmt(estimate.upper)}]; without the best date "
            f"{_fmt(information.gain_excluding_best_date(name))}"
        )
    lines.extend(
        [
            "- A research claim about ranking skill, not a trading claim: trading still "
            "needs every promotion gate.",
            "",
        ]
    )
    return lines


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(number):
        return "unavailable"
    return f"{number:.6f}"
