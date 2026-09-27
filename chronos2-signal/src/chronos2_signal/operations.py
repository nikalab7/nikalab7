"""Operational flows: the after-close run and the registered study.

Two entry points matter:

* :func:`run_after_close` -- one origin, end to end: validate, forecast, score,
  apply the policy, **commit**, then notify. The commit precedes the
  notification so that a retry after a delivery failure is idempotent rather
  than duplicative.
* :func:`run_study` -- a registered walk-forward comparison of one candidate
  against the fixed controls, on one shared date index, with the date-block
  bootstrap and the promotion gates applied at a named checkpoint.

Both write their provenance before their results. A number without a manifest
is not reportable.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .calendar_spec import ExchangeCalendar
from .config import DesignConfig
from .decision import DecisionModel
from .evaluation import (
    BootstrapEstimate,
    DailySeries,
    GateReport,
    block_profitability,
    contributor_concentration,
    evaluate_promotion_gates,
    paired_block_bootstrap,
    trade_metrics,
)
from .holdout import AccessMode
from .notifier import Notifier, render_no_alert_notice
from .pipeline import BacktestResult, OriginBatch, ResearchPipeline, WalkForwardRunner
from .policy import PolicyEngine, PolicyOutcome, dedup_key
from .portfolio import ReferencePortfolio
from .protocol import Fold, ProtocolSchedule, label_available_at
from .provenance import ReleaseManifest, capture_environment
from .storage import Ledger
from .variants import cash_series, exposure_matched_series

__all__ = [
    "OperationsError",
    "AfterCloseResult",
    "run_after_close",
    "StudyResult",
    "run_study",
    "build_release_manifest",
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

    The ordering is load-bearing. Signals are written to the ledger before any
    channel is contacted, so a failed delivery leaves a committed row that the
    next attempt finds already present.
    """
    config = pipeline.config
    now = now or dt.datetime.now(dt.timezone.utc)
    engine = PolicyEngine(config)
    environment = capture_environment()
    run_id = f"after-close-{variant}-{origin_session.isoformat()}-{uuid.uuid4().hex[:8]}"

    ledger.start_run(
        run_id=run_id,
        run_kind="after_close",
        origin_session=origin_session,
        started_at=now,
        status="running",
        message=None,
        design_fingerprint=config.fingerprint(),
        environment_digest=environment.digest(),
        counts=None,
    )

    try:
        batch = pipeline.build_batch(
            origin_session, variant, suppressed_sectors=suppressed_sectors
        )
        candidates = []
        for score in batch.scorable():
            from .decision import OriginRow
            from .policy import AlertCandidate

            row = OriginRow(
                origin_session=origin_session,
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

        open_positions = []
        if portfolio is not None:
            from .policy import OpenPositionView

            open_positions = [
                OpenPositionView(
                    symbol=position.symbol,
                    issuer_id=position.issuer_id,
                    sector=position.sector,
                    planned_exit_session=position.planned_exit_session,
                )
                for position in portfolio.positions
            ]

        outcome = engine.evaluate(
            origin_session=origin_session,
            candidates=candidates,
            open_positions=open_positions,
            correlation=pipeline.correlation_lookup(origin_session),
            paused=bool(portfolio.paused) if portfolio is not None else False,
            suppressed_sectors=suppressed_sectors,
            scan_suppressed=scan_suppressed,
        )

        timing = label_available_at(
            pipeline.calendar,
            origin_session,
            horizon_sessions=config.model.horizon_sessions,
        )
        data_timestamp = pipeline.calendar.session_close(origin_session).to_pydatetime()
        expires_at = pipeline.calendar.session_open(timing.entry_session).to_pydatetime()

        # Persist every decision, alerted or suppressed, so a quiet origin is
        # explainable afterwards without re-running anything.
        signal_ids: list[str] = []
        for decision in outcome.decisions:
            candidate = decision.candidate
            key = dedup_key(
                config.design_version, candidate.issuer_id, origin_session, "2_sessions"
            )
            payload = engine.signal_payload(
                decision,
                data_timestamp=data_timestamp,
                forecaster_provenance=(
                    pipeline.forecaster.provenance()
                    if pipeline.forecaster is not None
                    else {"forecaster": "none"}
                ),
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
                payload=payload,
                expires_at=expires_at,
            )
            if decision.alerted:
                signal_ids.append(signal_id)

        notified = 0
        if notifier is not None:
            if outcome.alert_count == 0:
                subject, body = render_no_alert_notice(
                    origin_session, outcome.origin_notes, engine.output_label.value
                )
                notifier.channel.send(subject, body)
            else:
                notified = notifier.notify_session(
                    origin_session, variant=variant, now=now
                )

        ledger.finish_run(
            run_id,
            status="ok",
            finished_at=dt.datetime.now(dt.timezone.utc),
            counts={
                "eligible": batch.eligible_count,
                "scorable": len(batch.scorable()),
                "alerts": outcome.alert_count,
                "notified": notified,
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


# --------------------------------------------------------------------------- #
# Registered study
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StudyResult:
    """A candidate and its controls on one shared date index."""

    variant: str
    checkpoint: str
    candidate: BacktestResult
    controls: Mapping[str, DailySeries]
    bootstrap: Mapping[str, BootstrapEstimate]
    gates: GateReport
    manifest: ReleaseManifest
    fold: Fold | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        metrics = trade_metrics(
            self.candidate.trades, origins_scanned=self.candidate.origins_scanned
        )
        return {
            "variant": self.variant,
            "checkpoint": self.checkpoint,
            "output_label": self.gates.status_label,
            "candidate": self.candidate.summary(),
            "trade_metrics": metrics.to_dict(),
            "controls": {
                name: {"mean_daily_return": series.mean, "sessions": len(series)}
                for name, series in self.controls.items()
            },
            "bootstrap": {
                name: estimate.to_dict() for name, estimate in self.bootstrap.items()
            },
            "block_profitability": block_profitability(
                self.candidate.daily, block_sessions=20
            ),
            "concentration": contributor_concentration(list(self.candidate.trades)),
            "gates": self.gates.to_dict(),
            "release_manifest": self.manifest.to_dict(),
            "fold": self.fold.describe() if self.fold else None,
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


def run_study(
    *,
    pipeline: ResearchPipeline,
    schedule: ProtocolSchedule,
    variant: str,
    checkpoint: str,
    fold_index: int = 0,
    cost_scenario: str = "base",
    bootstrap_samples: int | None = None,
    code_revision: str = "unknown",
    notes: Sequence[str] = (),
) -> StudyResult:
    """Run one registered fold for ``variant`` against the fixed controls.

    The candidate, the momentum control, the exposure-matched market benchmark
    and cash all share one date index, one set of fills and one cost scenario,
    and the paired bootstrap resamples the same date blocks for all of them.

    Raises:
        OperationsError: If the schedule has no folds, which means the clean
            history cannot support the registered blocks.
    """
    config = pipeline.config
    if not schedule.folds:
        raise OperationsError(
            "the schedule contains no development folds; collect more dates rather "
            "than shortening a block"
        )
    if not 0 <= fold_index < len(schedule.folds):
        raise OperationsError(
            f"fold {fold_index} outside the {len(schedule.folds)} available folds"
        )
    fold = schedule.folds[fold_index]

    runner = WalkForwardRunner(pipeline=pipeline, cost_scenario=cost_scenario)
    cache: dict[tuple[str, dt.date], OriginBatch] = {}
    model = runner.fit(
        variant,
        fit_sessions=fold.fit_sessions,
        calibration_sessions=fold.calibration_sessions,
        fit_deadline=fold.fit_deadline.to_pydatetime(),
        calibration_deadline=fold.calibration_deadline.to_pydatetime(),
        batch_cache=cache,
    )
    candidate = runner.run(
        variant,
        model=model,
        score_sessions=fold.validation_sessions,
        batch_cache=cache,
    )
    momentum = runner.run_momentum_control(
        score_sessions=fold.validation_sessions, batch_cache=cache
    )

    sessions = candidate.daily.sessions
    market_series = _market_returns(pipeline, sessions)
    exposure = [day.gross_exposure for day in candidate.portfolio.days]
    matched = exposure_matched_series(
        name="spy_exposure_matched",
        sessions=sessions,
        market_returns=market_series,
        candidate_gross_exposure=exposure,
    )
    cash = cash_series("cash", sessions)

    systems = [candidate.daily, momentum.daily, matched, cash]
    if momentum.daily.sessions != sessions:
        raise OperationsError(
            "control and candidate date indices diverged; the comparison would be "
            "between samples rather than between systems"
        )
    estimates = paired_block_bootstrap(
        systems,
        block_sessions=config.validation.bootstrap_block_sessions,
        samples=bootstrap_samples or config.validation.bootstrap_samples,
        confidence=config.validation.promotion_bootstrap_confidence,
        reference=momentum.daily.name,
    )

    gates = evaluate_promotion_gates(
        checkpoint=checkpoint,
        historical_trades=list(candidate.trades),
        forward_trades=[],
        forward_sessions=0,
        portfolio_series=candidate.daily,
        baseline_series={
            momentum.daily.name: momentum.daily,
            matched.name: matched,
            cash.name: cash,
        },
        bootstrap=estimates,
        calibration_brier=float("nan"),
        base_rate_brier=float("nan"),
        coverage_p10_p90=float("nan"),
        reliability=[],
        config_min_forward_sessions=config.validation.forward_min_sessions,
        config_min_forward_signals=config.validation.forward_min_executed_signals,
        config_min_combined_signals=config.validation.combined_evaluation_min_executed_signals,
        config_min_combined_sessions=config.validation.combined_evaluation_min_signal_sessions,
        config_min_profitable_blocks=config.validation.promotion_min_profitable_blocks,
        config_block_sessions=config.validation.promotion_block_sessions,
        historical_origins_scanned=candidate.origins_scanned,
    )

    manifest = build_release_manifest(
        config=config,
        calendar=pipeline.calendar,
        universe_manifest_hash=pipeline.watchlist.hash(),
        code_revision=code_revision,
        notes=(
            *notes,
            f"forecaster: {candidate.forecaster_provenance.get('forecaster', 'unknown')}",
            *(
                (
                    "produced with a non-model forecaster fixture: no predictive "
                    "content, plumbing only",
                )
                if candidate.forecaster_provenance.get("is_frozen_checkpoint") is False
                else ()
            ),
        ),
    )

    # The caller's notes lead the report. A warning about what the data actually
    # is belongs at the top of the thing a reader looks at, not only inside the
    # manifest.
    study_notes = [
        *notes,
        *schedule.notes,
        "one development fold; this is not a multi-year validation",
        f"holdout mode: {pipeline.guard.mode.value}",
    ]
    if candidate.forecaster_provenance.get("is_frozen_checkpoint") is False:
        study_notes.insert(
            0,
            "produced with a non-model forecaster fixture: no predictive content, "
            "plumbing only",
        )
    if pipeline.guard.mode is AccessMode.DEVELOPMENT:
        study_notes.append("reserved final-test origins were not read")

    return StudyResult(
        variant=variant,
        checkpoint=checkpoint,
        candidate=candidate,
        controls={
            momentum.daily.name: momentum.daily,
            matched.name: matched,
            cash.name: cash,
        },
        bootstrap=estimates,
        gates=gates,
        manifest=manifest,
        fold=fold,
        notes=tuple(study_notes),
    )


def _market_returns(
    pipeline: ResearchPipeline, sessions: Sequence[dt.date]
) -> list[float]:
    """Daily market-proxy returns on exactly ``sessions``."""
    panel = pipeline.source.daily_panel(pipeline.market_proxy or "SPY")
    returns: list[float] = []
    previous: float | None = None
    for session in sessions:
        try:
            close = float(panel.row(session)["close"])
        except Exception:
            returns.append(0.0)
            continue
        returns.append(0.0 if previous is None else close / previous - 1.0)
        previous = close
    return returns


def render_markdown(result: StudyResult) -> str:
    """A compact, honest report.

    Coverage, trade counts and no-alert dates are reported alongside returns,
    and the label is whatever the gates actually support.
    """
    metrics = trade_metrics(
        result.candidate.trades, origins_scanned=result.candidate.origins_scanned
    )
    lines = [
        f"# {result.variant} - {result.checkpoint}",
        "",
        f"**Label:** {result.gates.status_label}",
        f"**Design:** {result.manifest.design_version} "
        f"(fingerprint `{result.manifest.design_fingerprint[:12]}`)",
        f"**Forecaster:** {result.candidate.forecaster_provenance.get('forecaster')}",
        "",
        "## Coverage",
        "",
        f"- Origins scanned: {metrics.origins_scanned}",
        f"- Origins with alerts: {metrics.origins_with_alerts}",
        f"- Origins with no alert: {metrics.no_alert_origins}",
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
        "## Controls on the same dates, fills and costs",
        "",
    ]
    for name, series in sorted(result.controls.items()):
        lines.append(f"- {name}: mean daily net return {_fmt(series.mean)}")
    lines.extend(["", "## Date-block bootstrap", ""])
    for name, estimate in sorted(result.bootstrap.items()):
        lines.append(
            f"- {name}: point {_fmt(estimate.point)}, "
            f"{estimate.confidence:.0%} interval "
            f"[{_fmt(estimate.lower)}, {_fmt(estimate.upper)}] "
            f"over {estimate.n_sessions} sessions, "
            f"block {estimate.block_sessions}"
        )
    lines.extend(["", "## Promotion gates", ""])
    for gate in result.gates.results:
        mark = "PASS" if gate.passed else "FAIL"
        lines.append(f"- [{mark}] {gate.number}. {gate.name} - {gate.detail}")
    lines.extend(["", "## Notes", ""])
    for note in (*result.notes, *result.gates.notes):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(number):
        return "unavailable"
    return f"{number:.6f}"
