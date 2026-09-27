"""Alert policy: thresholds, ranking, capacity and deduplication.

The thresholds are fixed before results. They do not assert that a 60%
calibrated probability or a 30 basis point net estimate is achievable -- if the
estimator rarely or never qualifies, that is a reportable outcome. Zero alerts
is a valid result, there is no forced daily top pick, and no threshold moves to
make a dashboard look active.

Every suppression carries its reason, so a quiet stretch can be explained
afterwards without re-running anything.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from .config import DesignConfig
from .decision import DecisionEstimate
from .quality import EligibilityResult

__all__ = [
    "PolicyError",
    "OutputLabel",
    "AlertCandidate",
    "OpenPositionView",
    "AlertDecision",
    "PolicyOutcome",
    "PolicyEngine",
    "dedup_key",
]


class PolicyError(RuntimeError):
    """Raised on an inconsistent policy input."""


class OutputLabel(str, Enum):
    """The label every rendered output must carry.

    ``QUALIFIED_SIGNAL`` becomes available only after the registered evaluation
    gates pass. There is no label that promises a rise.
    """

    RESEARCH_UNVALIDATED = "RESEARCH / UNVALIDATED"
    QUALIFIED_SIGNAL = "QUALIFIED SIGNAL"


def dedup_key(
    strategy_version: str,
    issuer_id: str,
    signal_session: dt.date,
    holding_period: str,
) -> str:
    """The registered deduplication key.

    Keyed on the issuer rather than the symbol: a rename or a second share
    class must not produce two alerts for the same company.
    """
    return "|".join([strategy_version, issuer_id, signal_session.isoformat(), holding_period])


@dataclass(frozen=True)
class AlertCandidate:
    """One scored, eligible symbol at one origin."""

    symbol: str
    issuer_id: str
    sector: str
    estimate: DecisionEstimate
    eligibility: EligibilityResult
    forecast_cache_key: str | None = None
    out_of_sample_observations: int = 0
    out_of_sample_dates: int = 0
    event_flags: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()

    @property
    def rank_score(self) -> float:
        """Estimated net return per unit of two-session volatility."""
        sigma = self.estimate.sigma_2d
        if not math.isfinite(sigma) or sigma <= 0.0:
            return float("-inf")
        return self.estimate.estimated_net_return / sigma


@dataclass(frozen=True)
class OpenPositionView:
    """An already-open position, as the policy engine needs to see it."""

    symbol: str
    issuer_id: str
    sector: str
    planned_exit_session: dt.date


@dataclass(frozen=True)
class AlertDecision:
    """What the policy decided about one candidate."""

    candidate: AlertCandidate
    alerted: bool
    reasons: tuple[str, ...]
    rank_position: int
    rank_score: float

    @property
    def symbol(self) -> str:
        return self.candidate.symbol


@dataclass(frozen=True)
class PolicyOutcome:
    """All decisions for one origin, plus why the origin as a whole behaved so."""

    origin_session: dt.date
    decisions: tuple[AlertDecision, ...]
    origin_notes: tuple[str, ...] = ()
    paused: bool = False

    @property
    def alerts(self) -> tuple[AlertDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.alerted)

    @property
    def alert_count(self) -> int:
        return len(self.alerts)

    def suppression_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            if decision.alerted:
                continue
            for reason in decision.reasons:
                key = reason.split(":", 1)[0]
                counts[key] = counts.get(key, 0) + 1
        return counts


#: Signature of a pairwise correlation lookup. Returning ``None`` means the
#: required history is missing, which blocks the allocation rather than
#: defaulting to "uncorrelated".
CorrelationLookup = Callable[[str, str], float | None]


@dataclass
class PolicyEngine:
    """Applies the registered alert policy at one origin."""

    config: DesignConfig
    strategy_version: str | None = None
    holding_period: str = "2_sessions"

    def __post_init__(self) -> None:
        if self.strategy_version is None:
            self.strategy_version = self.config.design_version

    @property
    def output_label(self) -> OutputLabel:
        return (
            OutputLabel.QUALIFIED_SIGNAL
            if self.config.is_validated
            else OutputLabel.RESEARCH_UNVALIDATED
        )

    def evaluate(
        self,
        *,
        origin_session: dt.date,
        candidates: Sequence[AlertCandidate],
        open_positions: Sequence[OpenPositionView] = (),
        correlation: CorrelationLookup | None = None,
        paused: bool = False,
        suppressed_sectors: frozenset[str] = frozenset(),
        scan_suppressed: bool = False,
    ) -> PolicyOutcome:
        """Decide which candidates become alerts.

        Args:
            paused: Set when the reference portfolio is inside a drawdown
                pause. New qualified alerts are suppressed; existing positions
                keep their registered time exits.
            scan_suppressed: Set when a required market-wide input failed.

        The order is deliberate: thresholds first, then ranking, then capacity.
        Capacity is applied to an already-ranked list so that the *reason* a
        candidate was dropped is unambiguous.
        """
        decision = self.config.decision
        portfolio = self.config.portfolio
        notes: list[str] = []

        if scan_suppressed:
            notes.append("scan suppressed: a required market-wide input was unavailable")
        if paused:
            notes.append(
                "new alerts paused: reference portfolio drawdown reached the registered "
                "limit; existing positions keep their time exits"
            )

        qualified: list[AlertCandidate] = []
        rejected: list[tuple[AlertCandidate, list[str]]] = []

        for candidate in candidates:
            reasons: list[str] = []
            if not candidate.eligibility.eligible:
                reasons.append(
                    "eligibility: " + "; ".join(candidate.eligibility.failures[:3])
                )
            if candidate.sector in suppressed_sectors:
                reasons.append("sector_suppressed: sector input unavailable at this origin")

            estimate = candidate.estimate
            if not math.isfinite(estimate.calibrated_probability):
                reasons.append("probability: calibrated probability is not finite")
            elif estimate.calibrated_probability < decision.min_probability:
                reasons.append(
                    f"probability: {estimate.calibrated_probability:.4f} below "
                    f"{decision.min_probability:.2f}"
                )
            if not math.isfinite(estimate.estimated_net_return):
                reasons.append("estimate: estimated net return is not finite")
            elif estimate.estimated_net_return < decision.min_estimated_net_return:
                reasons.append(
                    f"estimate: {estimate.estimated_net_return:.5f} below "
                    f"{decision.min_estimated_net_return:.5f}"
                )
            if decision.require_positive_stress_estimate:
                if not math.isfinite(estimate.estimated_net_return_stress):
                    reasons.append("stress: repriced estimate is not finite")
                elif estimate.estimated_net_return_stress <= 0.0:
                    reasons.append(
                        f"stress: repriced estimate {estimate.estimated_net_return_stress:.5f} "
                        "is not positive"
                    )
            if any(position.issuer_id == candidate.issuer_id for position in open_positions):
                reasons.append("open_position: this issuer already has an open position")
            if paused or scan_suppressed:
                reasons.append(
                    "paused: no new allocation while the portfolio pause or scan "
                    "suppression is in force"
                )

            if reasons:
                rejected.append((candidate, reasons))
            else:
                qualified.append(candidate)

        ranked = sorted(
            qualified,
            key=lambda item: (-item.rank_score, -item.estimate.estimated_net_return, item.symbol),
        )

        decisions: list[AlertDecision] = []
        accepted: list[AlertCandidate] = []
        sector_taken: dict[str, int] = {}
        for position in open_positions:
            sector_taken[position.sector] = sector_taken.get(position.sector, 0) + 1
        open_count = len(open_positions)

        for position_index, candidate in enumerate(ranked, start=1):
            reasons: list[str] = []
            if len(accepted) >= portfolio.max_new_alerts_per_origin:
                reasons.append(
                    f"capacity: already at {portfolio.max_new_alerts_per_origin} new "
                    "alerts for this origin"
                )
            if open_count + len(accepted) >= portfolio.max_open_positions:
                reasons.append(
                    f"capacity: already at {portfolio.max_open_positions} simultaneous "
                    "open positions"
                )
            if (
                sector_taken.get(candidate.sector, 0)
                >= portfolio.max_open_per_sector
            ):
                reasons.append(
                    f"capacity: sector {candidate.sector} already at "
                    f"{portfolio.max_open_per_sector} open position(s)"
                )
            if not reasons:
                correlation_reason = self._correlation_block(
                    candidate, accepted, open_positions, correlation
                )
                if correlation_reason:
                    reasons.append(correlation_reason)

            if reasons:
                decisions.append(
                    AlertDecision(
                        candidate=candidate,
                        alerted=False,
                        reasons=tuple(reasons),
                        rank_position=position_index,
                        rank_score=candidate.rank_score,
                    )
                )
                continue

            accepted.append(candidate)
            sector_taken[candidate.sector] = sector_taken.get(candidate.sector, 0) + 1
            decisions.append(
                AlertDecision(
                    candidate=candidate,
                    alerted=True,
                    reasons=(),
                    rank_position=position_index,
                    rank_score=candidate.rank_score,
                )
            )

        for candidate, reasons in rejected:
            decisions.append(
                AlertDecision(
                    candidate=candidate,
                    alerted=False,
                    reasons=tuple(reasons),
                    rank_position=0,
                    rank_score=candidate.rank_score,
                )
            )

        if not any(item.alerted for item in decisions):
            notes.append(
                "no alerts at this origin: a valid outcome, not a failure of the run"
            )
        return PolicyOutcome(
            origin_session=origin_session,
            decisions=tuple(decisions),
            origin_notes=tuple(notes),
            paused=paused,
        )

    def _correlation_block(
        self,
        candidate: AlertCandidate,
        accepted: Sequence[AlertCandidate],
        open_positions: Sequence[OpenPositionView],
        correlation: CorrelationLookup | None,
    ) -> str | None:
        """Whether the correlation cap blocks this candidate.

        Missing history blocks the allocation. Treating an unmeasurable pair as
        uncorrelated would quietly defeat the cap.
        """
        cap = self.config.portfolio.pairwise_correlation_cap
        others = [item.symbol for item in accepted] + [
            item.symbol for item in open_positions
        ]
        if not others:
            return None
        if correlation is None:
            return (
                "correlation: no correlation source supplied; the allocation is blocked "
                "rather than assumed uncorrelated"
            )
        for other in others:
            value = correlation(candidate.symbol, other)
            if value is None or not math.isfinite(value):
                return (
                    f"correlation: missing trailing-"
                    f"{self.config.portfolio.correlation_window_sessions}-session history "
                    f"against {other}; allocation blocked"
                )
            if value > cap:
                return (
                    f"correlation: {value:.3f} against {other} exceeds the {cap:.2f} cap"
                )
        return None

    def signal_payload(
        self,
        decision: AlertDecision,
        *,
        data_timestamp: dt.datetime,
        empirical_coverage: Mapping[str, float] | None = None,
        forecaster_provenance: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """The fields a useful alert must carry.

        Notably: the probability is labelled a model estimate, the forecast
        range is reported with observed coverage context, and the number of
        out-of-sample comparable observations and dates is stated. Feature
        summaries are descriptive, not causal explanations.
        """
        candidate = decision.candidate
        estimate = candidate.estimate
        return {
            "output_label": self.output_label.value,
            "symbol": candidate.symbol,
            "issuer_id": candidate.issuer_id,
            "sector": candidate.sector,
            "signal_session": decision.candidate.estimate.origin_session.isoformat(),
            "data_timestamp": data_timestamp.isoformat(),
            "entry_convention": self.config.execution.entry,
            "exit_convention": self.config.execution.exit,
            "estimated_net_return_model_estimate": estimate.estimated_net_return,
            "estimated_net_return_stress_cost": estimate.estimated_net_return_stress,
            "calibrated_probability_model_estimate": estimate.calibrated_probability,
            "probability_caveat": (
                "a model estimate, not a measured frequency; see reliability and "
                "coverage diagnostics"
            ),
            "sigma_2d": estimate.sigma_2d,
            "rank_position": decision.rank_position,
            "rank_score": decision.rank_score,
            "forecast_cache_key": candidate.forecast_cache_key,
            "empirical_coverage": dict(empirical_coverage or {}),
            "out_of_sample_observations": candidate.out_of_sample_observations,
            "out_of_sample_dates": candidate.out_of_sample_dates,
            "event_flags": list(candidate.event_flags),
            "risk_flags": list(candidate.risk_flags),
            "eligibility_metrics": dict(candidate.eligibility.metrics),
            "forecaster": dict(forecaster_provenance or {}),
            "caveats": [
                "Version 1 has a fixed time exit, no stop and no take-profit.",
                "The p10 quantile is not a guaranteed loss bound.",
                "Unscheduled news and overnight gaps are material unmodelled risks.",
            ],
        }
