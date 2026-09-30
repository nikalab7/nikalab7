"""Metrics, the date-block bootstrap and the promotion gates.

Three habits are built into this module rather than left to discipline:

* **Dates, not rows, are the unit of uncertainty.** Fifty stocks on one day are
  not fifty independent observations. Every interval here resamples *date
  blocks*, and the same blocks are applied to every competing system so the
  comparison is paired.
* **A bootstrap does not create missing regimes.** The interval describes
  sampling variability within the observed history. Two years containing one
  market regime stays two years containing one market regime, and the reports
  say so.
* **A failed gate means the research stays unvalidated.** The gates are
  conservative acceptance rules, not a theorem that future returns are
  positive. A 55% win rate can still lose money.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "EvaluationError",
    "TradeRecord",
    "TradeMetrics",
    "trade_metrics",
    "DailySeries",
    "BootstrapEstimate",
    "moving_block_bootstrap",
    "paired_block_bootstrap",
    "pinball_loss",
    "interval_coverage",
    "brier_score",
    "reliability_table",
    "spearman_correlation",
    "LabelledPrediction",
    "PredictiveReport",
    "predictive_report",
    "block_profitability",
    "contributor_concentration",
    "GateResult",
    "GateReport",
    "evaluate_promotion_gates",
]


class EvaluationError(RuntimeError):
    """Raised on an inconsistent evaluation input."""


# --------------------------------------------------------------------------- #
# Trade-level metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TradeRecord:
    """One executed reference trade."""

    origin_session: dt.date
    symbol: str
    issuer_id: str
    sector: str
    r_net_base: float
    r_net_stress: float
    r_net_severe: float
    max_adverse_excursion: float | None = None
    resolved: bool = True


@dataclass(frozen=True)
class TradeMetrics:
    """Trade-level summary. Distinct from portfolio-level results."""

    trades: int
    distinct_sessions: int
    origins_scanned: int
    origins_with_alerts: int
    no_alert_origins: int
    alert_coverage: float
    win_rate: float
    mean_net_return: float
    median_net_return: float
    profit_factor: float
    payoff_ratio: float
    worst_trade: float
    mean_adverse_excursion: float
    unresolved_exits: int
    stress_mean_net_return: float
    severe_mean_net_return: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "trades": self.trades,
            "distinct_sessions": self.distinct_sessions,
            "origins_scanned": self.origins_scanned,
            "origins_with_alerts": self.origins_with_alerts,
            "no_alert_origins": self.no_alert_origins,
            "alert_coverage": self.alert_coverage,
            "win_rate": self.win_rate,
            "mean_net_return": self.mean_net_return,
            "median_net_return": self.median_net_return,
            "profit_factor": self.profit_factor,
            "payoff_ratio": self.payoff_ratio,
            "worst_trade": self.worst_trade,
            "mean_adverse_excursion": self.mean_adverse_excursion,
            "unresolved_exits": self.unresolved_exits,
            "stress_mean_net_return": self.stress_mean_net_return,
            "severe_mean_net_return": self.severe_mean_net_return,
        }


def trade_metrics(
    trades: Sequence[TradeRecord],
    *,
    origins_scanned: int,
) -> TradeMetrics:
    """Summarise executed trades.

    ``origins_scanned`` counts every origin the scanner evaluated, including
    those that produced nothing, so alert coverage is reported rather than
    implied by the number of trades.
    """
    resolved = [trade for trade in trades if trade.resolved]
    unresolved = len(trades) - len(resolved)
    returns = np.asarray([trade.r_net_base for trade in resolved], dtype=float)
    sessions = {trade.origin_session for trade in trades}

    if returns.size:
        wins = returns[returns > 0.0]
        losses = returns[returns <= 0.0]
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        if gross_loss > 0.0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0.0:
            # No losing trade in the sample. Reported as infinite rather than
            # as a large finite number that would look like a measurement.
            profit_factor = float("inf")
        else:
            profit_factor = float("nan")
        payoff = (
            float(wins.mean() / abs(losses.mean()))
            if wins.size and losses.size and losses.mean() != 0.0
            else float("nan")
        )
        win_rate = float((returns > 0.0).mean())
        excursions = [
            trade.max_adverse_excursion
            for trade in resolved
            if trade.max_adverse_excursion is not None
            and math.isfinite(trade.max_adverse_excursion)
        ]
        mean_excursion = float(np.mean(excursions)) if excursions else float("nan")
    else:
        profit_factor = float("nan")
        payoff = float("nan")
        win_rate = float("nan")
        mean_excursion = float("nan")

    return TradeMetrics(
        trades=len(trades),
        distinct_sessions=len(sessions),
        origins_scanned=origins_scanned,
        origins_with_alerts=len(sessions),
        no_alert_origins=max(0, origins_scanned - len(sessions)),
        alert_coverage=(len(sessions) / origins_scanned) if origins_scanned else float("nan"),
        win_rate=win_rate,
        mean_net_return=float(returns.mean()) if returns.size else float("nan"),
        median_net_return=float(np.median(returns)) if returns.size else float("nan"),
        profit_factor=profit_factor,
        payoff_ratio=payoff,
        worst_trade=float(returns.min()) if returns.size else float("nan"),
        mean_adverse_excursion=mean_excursion,
        unresolved_exits=unresolved,
        stress_mean_net_return=(
            float(np.mean([t.r_net_stress for t in resolved])) if resolved else float("nan")
        ),
        severe_mean_net_return=(
            float(np.mean([t.r_net_severe for t in resolved])) if resolved else float("nan")
        ),
    )


# --------------------------------------------------------------------------- #
# Date-block bootstrap
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DailySeries:
    """A complete daily record of one system's net portfolio return."""

    name: str
    sessions: tuple[dt.date, ...]
    returns: np.ndarray

    def __post_init__(self) -> None:
        if len(self.sessions) != len(self.returns):
            raise EvaluationError(
                f"{self.name}: {len(self.sessions)} sessions and {len(self.returns)} returns"
            )

    def __len__(self) -> int:
        return len(self.sessions)

    @property
    def mean(self) -> float:
        return float(np.mean(self.returns)) if len(self.returns) else float("nan")


@dataclass(frozen=True)
class BootstrapEstimate:
    """A block-bootstrap interval for one statistic."""

    name: str
    point: float
    lower: float
    upper: float
    confidence: float
    block_sessions: int
    samples: int
    n_sessions: int

    @property
    def excludes_zero_above(self) -> bool:
        """Whether the lower bound is strictly above zero."""
        return math.isfinite(self.lower) and self.lower > 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "block_sessions": self.block_sessions,
            "samples": self.samples,
            "n_sessions": self.n_sessions,
        }


def _block_indices(
    n: int, block: int, samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Moving-block index matrix of shape ``(samples, n)``.

    Blocks start anywhere in ``0..n-block`` and are concatenated until the
    replicate is as long as the original series, preserving the within-block
    dependence that makes overlapping holds and contemporaneous correlation
    legible.
    """
    if n <= 0:
        raise EvaluationError("cannot bootstrap an empty series")
    block = max(1, min(block, n))
    starts_available = n - block + 1
    blocks_needed = math.ceil(n / block)
    starts = rng.integers(0, starts_available, size=(samples, blocks_needed))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(samples, -1)
    return indices[:, :n]


def moving_block_bootstrap(
    series: DailySeries,
    *,
    block_sessions: int,
    samples: int,
    confidence: float,
    seed: int = 20260927,
) -> BootstrapEstimate:
    """Interval for the mean daily net return of one system."""
    rng = np.random.default_rng(seed)
    indices = _block_indices(len(series), block_sessions, samples, rng)
    means = series.returns[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return BootstrapEstimate(
        name=series.name,
        point=series.mean,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        block_sessions=block_sessions,
        samples=samples,
        n_sessions=len(series),
    )


def paired_block_bootstrap(
    systems: Sequence[DailySeries],
    *,
    block_sessions: int,
    samples: int,
    confidence: float,
    differences: Sequence[tuple[str, str]] = (),
    seed: int = 20260927,
) -> dict[str, BootstrapEstimate]:
    """Intervals for every system, and for each requested ``(a, b)`` difference.

    All systems are resampled on the **same** date blocks. That is what makes
    an incremental claim meaningful: the candidate and its baseline see the same
    days, so contemporaneous dependence is preserved rather than averaged away.

    Differences are named explicitly -- ``C256`` against ``B0``, against the
    momentum control, against its own exposure-matched benchmark -- because
    each answers a different question, and a single reference system cannot
    ask all of them.
    """
    if not systems:
        return {}
    lengths = {len(system) for system in systems}
    if len(lengths) != 1:
        raise EvaluationError(
            f"paired bootstrap needs one common date index; got lengths {sorted(lengths)}"
        )
    session_sets = {system.sessions for system in systems}
    if len(session_sets) != 1:
        raise EvaluationError(
            "paired bootstrap needs identical session indices across systems; "
            "a candidate and a baseline must be evaluated on the same dates"
        )

    n = len(systems[0])
    rng = np.random.default_rng(seed)
    indices = _block_indices(n, block_sessions, samples, rng)
    alpha = (1.0 - confidence) / 2.0

    estimates: dict[str, BootstrapEstimate] = {}
    replicate_means: dict[str, np.ndarray] = {}
    for system in systems:
        means = system.returns[indices].mean(axis=1)
        replicate_means[system.name] = means
        lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
        estimates[system.name] = BootstrapEstimate(
            name=system.name,
            point=system.mean,
            lower=float(lower),
            upper=float(upper),
            confidence=confidence,
            block_sessions=block_sessions,
            samples=samples,
            n_sessions=n,
        )

    for left, right in differences:
        for name in (left, right):
            if name not in replicate_means:
                raise EvaluationError(f"system {name!r} was not supplied to the bootstrap")
        if left == right:
            raise EvaluationError(f"cannot difference {left!r} against itself")
        difference = replicate_means[left] - replicate_means[right]
        lower, upper = np.quantile(difference, [alpha, 1.0 - alpha])
        key = f"{left}_minus_{right}"
        estimates[key] = BootstrapEstimate(
            name=key,
            point=estimates[left].point - estimates[right].point,
            lower=float(lower),
            upper=float(upper),
            confidence=confidence,
            block_sessions=block_sessions,
            samples=samples,
            n_sessions=n,
        )
    return estimates


# --------------------------------------------------------------------------- #
# Predictive metrics
# --------------------------------------------------------------------------- #


def pinball_loss(
    quantile_levels: Sequence[float],
    predictions: np.ndarray,
    realised: float | Sequence[float],
) -> float:
    """Mean pinball loss across levels (and horizon steps, if supplied).

    ``predictions`` is ``(n_levels,)`` or ``(n_levels, horizon)``; ``realised``
    matches its trailing shape.
    """
    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(realised, dtype=float)
    if predicted.ndim == 1:
        predicted = predicted[:, None]
        actual = actual.reshape(1)
    if predicted.shape[1] != actual.size:
        raise EvaluationError(
            f"pinball loss: {predicted.shape[1]} horizon steps vs {actual.size} realised"
        )
    levels = np.asarray(quantile_levels, dtype=float)[:, None]
    if levels.shape[0] != predicted.shape[0]:
        raise EvaluationError("pinball loss: level count does not match prediction rows")
    difference = actual[None, :] - predicted
    loss = np.maximum(levels * difference, (levels - 1.0) * difference)
    return float(np.mean(loss))


def interval_coverage(
    lower: Sequence[float], upper: Sequence[float], realised: Sequence[float]
) -> float:
    """Empirical fraction of outcomes inside a predicted interval."""
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    actual = np.asarray(realised, dtype=float)
    if not (low.size == high.size == actual.size):
        raise EvaluationError("coverage inputs must have equal length")
    if actual.size == 0:
        return float("nan")
    mask = np.isfinite(low) & np.isfinite(high) & np.isfinite(actual)
    if not mask.any():
        return float("nan")
    inside = (actual[mask] >= low[mask]) & (actual[mask] <= high[mask])
    return float(inside.mean())


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error of probabilistic forecasts."""
    probability = np.asarray(probabilities, dtype=float)
    label = np.asarray(labels, dtype=float)
    if probability.size != label.size:
        raise EvaluationError("Brier score inputs must have equal length")
    if probability.size == 0:
        return float("nan")
    return float(np.mean((probability - label) ** 2))


def reliability_table(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 5,
) -> list[dict[str, float]]:
    """Binned predicted-versus-observed frequencies.

    Reported alongside the Brier score, because a good aggregate score can hide
    a badly shaped reliability curve.
    """
    probability = np.asarray(probabilities, dtype=float)
    label = np.asarray(labels, dtype=float)
    if probability.size != label.size:
        raise EvaluationError("reliability inputs must have equal length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (probability >= low) & (
            probability < high if index < bins - 1 else probability <= high
        )
        count = int(mask.sum())
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": float(count),
                "mean_predicted": float(probability[mask].mean()) if count else float("nan"),
                "observed_rate": float(label[mask].mean()) if count else float("nan"),
            }
        )
    return rows


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Rank association, used for cross-sectional selection skill.

    Implemented directly on average ranks so that the dependency surface stays
    at numpy; ties receive their mean rank.
    """
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.size != b.size:
        raise EvaluationError("rank correlation inputs must have equal length")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ranked_a = _average_ranks(a[mask])
    ranked_b = _average_ranks(b[mask])
    if np.std(ranked_a) == 0.0 or np.std(ranked_b) == 0.0:
        return float("nan")
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # Average the ranks of tied values.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            tied = inverse == index
            ranks[tied] = ranks[tied].mean()
    return ranks


@dataclass(frozen=True)
class LabelledPrediction:
    """One out-of-sample scored row with its realised outcome attached.

    Attributes:
        base_rate: Training-derived positive rate of the model that scored the
            row -- the constant the calibrated probability has to beat.
        r_net: Realised net return of the reference policy (base costs).
        realised_log_return: ``100 * ln(C_D2 / C_D0)``, the quantity the
            forecast quantiles describe; ``None`` if it could not be observed.
        terminal_quantiles: Forecast quantiles at the D2 terminal bar, in the
            same units; ``None`` for a variant without a forecast.
    """

    origin_session: dt.date
    symbol: str
    calibrated_probability: float
    estimated_net_return: float
    sigma_2d: float
    base_rate: float
    r_net: float
    realised_log_return: float | None = None
    terminal_quantiles: Mapping[float, float] | None = None


@dataclass(frozen=True)
class PredictiveReport:
    """Out-of-sample predictive diagnostics (section 14).

    A model can improve price error without improving selection, and the
    reverse, so the forecast diagnostics and the probability diagnostics are
    reported side by side rather than rolled into one score.
    """

    rows: int
    dates: int
    brier: float
    base_rate_brier: float
    reliability: tuple[Mapping[str, float], ...]
    coverage_p10_p90: float
    pinball_terminal: float
    pinball_trailing_vol_reference: float
    median_abs_error: float
    persistence_abs_error: float
    rank_ic_mean: float
    rank_ic_dates: int

    @property
    def beats_base_rate(self) -> bool:
        return (
            math.isfinite(self.brier)
            and math.isfinite(self.base_rate_brier)
            and self.brier < self.base_rate_brier
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "dates": self.dates,
            "brier": self.brier,
            "base_rate_brier": self.base_rate_brier,
            "beats_base_rate": self.beats_base_rate,
            "reliability": [dict(row) for row in self.reliability],
            "coverage_p10_p90": self.coverage_p10_p90,
            "pinball_terminal": self.pinball_terminal,
            "pinball_trailing_vol_reference": self.pinball_trailing_vol_reference,
            "median_abs_error": self.median_abs_error,
            "persistence_abs_error": self.persistence_abs_error,
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_dates": self.rank_ic_dates,
        }


def predictive_report(
    rows: Sequence[LabelledPrediction],
    *,
    quantile_levels: Sequence[float],
    reliability_bins: int = 5,
) -> PredictiveReport:
    """Probability, forecast and ranking diagnostics over labelled predictions.

    * **Brier** of the calibrated probability against ``1[R_net > 0]``, next to
      the Brier of each row's own training base rate -- the comparison gate 7
      requires.
    * **Coverage** of the D2 p10-p90 band, **pinball loss** at the terminal bar,
      and the same pinball for a trailing-volatility reference band, so a
      forecast is judged against a simple alternative and not in isolation.
    * **Median error** against **persistence**, the zero-change forecast.
    * **Rank IC**: the cross-sectional Spearman association between estimated
      and realised net return, averaged over dates -- dates, not rows, because
      fifty stocks on one day are one observation of cross-sectional skill.
    """
    levels = [float(level) for level in quantile_levels]
    if not rows:
        nan = float("nan")
        return PredictiveReport(
            rows=0, dates=0, brier=nan, base_rate_brier=nan, reliability=(),
            coverage_p10_p90=nan, pinball_terminal=nan,
            pinball_trailing_vol_reference=nan, median_abs_error=nan,
            persistence_abs_error=nan, rank_ic_mean=nan, rank_ic_dates=0,
        )

    probabilities = np.asarray([row.calibrated_probability for row in rows], dtype=float)
    outcomes = np.asarray([1 if row.r_net > 0.0 else 0 for row in rows], dtype=int)
    base_rates = np.asarray([row.base_rate for row in rows], dtype=float)

    forecast_rows = [
        row
        for row in rows
        if row.terminal_quantiles is not None
        and row.realised_log_return is not None
        and math.isfinite(row.realised_log_return)
    ]
    nan = float("nan")
    coverage = pinball = reference_pinball = median_error = persistence_error = nan
    if forecast_rows:
        realised = np.asarray([row.realised_log_return for row in forecast_rows], dtype=float)
        coverage = interval_coverage(
            [row.terminal_quantiles[0.10] for row in forecast_rows],  # type: ignore[index]
            [row.terminal_quantiles[0.90] for row in forecast_rows],  # type: ignore[index]
            realised,
        )
        losses = []
        reference_losses = []
        normal = NormalDist()
        for row, actual in zip(forecast_rows, realised, strict=True):
            predicted = np.asarray([row.terminal_quantiles[level] for level in levels])  # type: ignore[index]
            losses.append(pinball_loss(levels, predicted, [actual]))
            # A trailing-volatility band in the target's units: the two-session
            # volatility scale, times 100 for the rebased log units.
            reference = np.asarray(
                [100.0 * normal.inv_cdf(level) * row.sigma_2d for level in levels]
            )
            reference_losses.append(pinball_loss(levels, reference, [actual]))
        pinball = float(np.mean(losses))
        reference_pinball = float(np.mean(reference_losses))
        medians = np.asarray(
            [row.terminal_quantiles[0.50] for row in forecast_rows], dtype=float  # type: ignore[index]
        )
        median_error = float(np.mean(np.abs(medians - realised)))
        persistence_error = float(np.mean(np.abs(realised)))

    by_date: dict[dt.date, list[LabelledPrediction]] = {}
    for row in rows:
        by_date.setdefault(row.origin_session, []).append(row)
    ics = [
        spearman_correlation(
            [row.estimated_net_return for row in group], [row.r_net for row in group]
        )
        for group in by_date.values()
        if len(group) >= 3
    ]
    finite_ics = [value for value in ics if math.isfinite(value)]

    return PredictiveReport(
        rows=len(rows),
        dates=len(by_date),
        brier=brier_score(probabilities, outcomes),
        base_rate_brier=float(np.mean((base_rates - outcomes) ** 2)),
        reliability=tuple(reliability_table(probabilities, outcomes, bins=reliability_bins)),
        coverage_p10_p90=coverage,
        pinball_terminal=pinball,
        pinball_trailing_vol_reference=reference_pinball,
        median_abs_error=median_error,
        persistence_abs_error=persistence_error,
        rank_ic_mean=float(np.mean(finite_ics)) if finite_ics else nan,
        rank_ic_dates=len(finite_ics),
    )


# --------------------------------------------------------------------------- #
# Robustness diagnostics
# --------------------------------------------------------------------------- #


def block_profitability(
    series: DailySeries, *, block_sessions: int
) -> list[dict[str, float | str]]:
    """Consecutive non-overlapping evaluation blocks and their total return.

    A result carried by one lucky stretch shows up here as a single profitable
    block among several losing ones.
    """
    rows: list[dict[str, float | str]] = []
    for start in range(0, len(series), block_sessions):
        window = series.returns[start : start + block_sessions]
        if window.size == 0:
            continue
        total = float(np.prod(1.0 + window) - 1.0)
        rows.append(
            {
                "first_session": series.sessions[start].isoformat(),
                "last_session": series.sessions[min(start + window.size, len(series)) - 1].isoformat(),
                "sessions": float(window.size),
                "total_return": total,
                "mean_daily_return": float(window.mean()),
            }
        )
    return rows


def contributor_concentration(
    trades: Sequence[TradeRecord], *, key: str = "issuer_id"
) -> dict[str, object]:
    """How much of the total profit one contributor explains.

    If a single issuer or sector explains almost everything, the claim is
    restricted to that contributor rather than generalised to the watchlist.
    """
    if not trades:
        return {"total": 0.0, "top_key": None, "top_share": float("nan"), "by_key": {}}
    totals: dict[str, float] = {}
    for trade in trades:
        label = getattr(trade, key)
        totals[label] = totals.get(label, 0.0) + trade.r_net_base
    total = sum(totals.values())
    top_key = max(totals, key=lambda name: totals[name]) if totals else None
    top_share = (
        totals[top_key] / total
        if top_key is not None and total != 0.0
        else float("nan")
    )
    returns = np.asarray([trade.r_net_base for trade in trades], dtype=float)
    best_index = int(np.argmax(returns)) if returns.size else -1
    without_best = (
        float(np.sum(np.delete(returns, best_index))) if returns.size > 1 else float("nan")
    )
    return {
        "total": total,
        "top_key": top_key,
        "top_share": top_share,
        "by_key": totals,
        "sum_excluding_best_trade": without_best,
        "profitable_excluding_best_trade": (
            bool(without_best > 0.0) if math.isfinite(without_best) else False
        ),
    }


# --------------------------------------------------------------------------- #
# Promotion gates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateResult:
    """One predeclared promotion requirement."""

    number: int
    name: str
    passed: bool
    detail: str
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GateReport:
    """The full gate assessment at one predeclared checkpoint."""

    checkpoint: str
    assessed_at: dt.datetime
    results: tuple[GateResult, ...]
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def status_label(self) -> str:
        return "QUALIFIED SIGNAL" if self.passed else "RESEARCH / UNVALIDATED"

    def failures(self) -> tuple[GateResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "assessed_at": self.assessed_at.isoformat(),
            "passed": self.passed,
            "status_label": self.status_label,
            "results": [
                {
                    "number": result.number,
                    "name": result.name,
                    "passed": result.passed,
                    "detail": result.detail,
                    "evidence": dict(result.evidence),
                }
                for result in self.results
            ],
            "notes": list(self.notes),
        }


def evaluate_promotion_gates(
    *,
    checkpoint: str,
    historical_trades: Sequence[TradeRecord],
    forward_trades: Sequence[TradeRecord],
    forward_sessions: int,
    portfolio_series: DailySeries,
    baseline_series: Mapping[str, DailySeries],
    bootstrap: Mapping[str, BootstrapEstimate],
    calibration_brier: float,
    base_rate_brier: float,
    coverage_p10_p90: float,
    reliability: Sequence[Mapping[str, float]],
    config_min_forward_sessions: int,
    config_min_forward_signals: int,
    config_min_combined_signals: int,
    config_min_combined_sessions: int,
    config_min_profitable_blocks: int,
    config_block_sessions: int,
    historical_origins_scanned: int,
    ablation_survived: bool | None = None,
    concentration_claim_restricted: bool | None = None,
    assessed_at: dt.datetime | None = None,
) -> GateReport:
    """Assess the eight predeclared promotion requirements.

    Two gates depend on judgements a metric cannot make -- whether an ablation
    genuinely survived, and whether a concentrated result has had its claim
    restricted. They are passed in explicitly and default to **not passed**, so
    an unexamined study cannot drift into a qualified label.

    Gates are assessed at predeclared checkpoints. Re-running this function
    until an interval turns positive would itself be a new research decision and
    must be recorded as one.
    """
    assessed_at = assessed_at or dt.datetime.now(dt.timezone.utc)
    combined = [*historical_trades, *forward_trades]
    combined_sessions = {trade.origin_session for trade in combined}
    results: list[GateResult] = []

    historical = trade_metrics(
        historical_trades, origins_scanned=historical_origins_scanned
    )
    base_positive = (
        math.isfinite(historical.mean_net_return) and historical.mean_net_return > 0.0
    )
    stress_positive = (
        math.isfinite(historical.stress_mean_net_return)
        and historical.stress_mean_net_return > 0.0
    )
    results.append(
        GateResult(
            number=1,
            name="final historical results positive at base and stress costs",
            passed=bool(base_positive and stress_positive),
            detail=(
                f"base mean {historical.mean_net_return:.5f}, "
                f"stress mean {historical.stress_mean_net_return:.5f}"
            ),
            evidence=historical.to_dict(),
        )
    )

    forward_ok = (
        forward_sessions >= config_min_forward_sessions
        and len(forward_trades) >= config_min_forward_signals
    )
    results.append(
        GateResult(
            number=2,
            name="forward paper evaluation long enough",
            passed=forward_ok,
            detail=(
                f"{forward_sessions} forward sessions and {len(forward_trades)} executed "
                f"signals against minima {config_min_forward_sessions}/"
                f"{config_min_forward_signals}; otherwise continue collecting"
            ),
            evidence={
                "forward_sessions": forward_sessions,
                "forward_trades": len(forward_trades),
            },
        )
    )

    combined_ok = (
        len(combined) >= config_min_combined_signals
        and len(combined_sessions) >= config_min_combined_sessions
    )
    results.append(
        GateResult(
            number=3,
            name="combined evidence meets the bookkeeping floors",
            passed=combined_ok,
            detail=(
                f"{len(combined)} executed trades across {len(combined_sessions)} distinct "
                f"signal sessions against minima {config_min_combined_signals}/"
                f"{config_min_combined_sessions}. These are bookkeeping floors, not "
                "guaranteed statistical power."
            ),
            evidence={
                "combined_trades": len(combined),
                "combined_sessions": len(combined_sessions),
            },
        )
    )

    own = bootstrap.get(portfolio_series.name)
    incremental = {
        name: estimate.to_dict()
        for name, estimate in bootstrap.items()
        if "_minus_" in name
    }
    lower_bound_ok = bool(own and own.excludes_zero_above)
    inconclusive = [
        name
        for name, estimate in bootstrap.items()
        if "_minus_" in name and not estimate.excludes_zero_above
    ]
    results.append(
        GateResult(
            number=4,
            name="bootstrap lower bound on mean daily net portfolio return above zero",
            passed=lower_bound_ok,
            detail=(
                (
                    f"lower bound {own.lower:.6g} at {own.confidence:.0%} over "
                    f"{own.n_sessions} sessions"
                    if own
                    else "no bootstrap estimate supplied for the candidate portfolio"
                )
                + (
                    "; incremental evidence inconclusive against "
                    + ", ".join(sorted(inconclusive))
                    + " - do not claim Transformer-specific alpha"
                    if inconclusive
                    else ""
                )
            ),
            evidence={"candidate": own.to_dict() if own else {}, "incremental": incremental},
        )
    )

    blocks = block_profitability(portfolio_series, block_sessions=config_block_sessions)
    profitable_blocks = [row for row in blocks if float(row["total_return"]) > 0.0]
    concentration = contributor_concentration(combined, key="issuer_id")
    robust_without_best = bool(concentration.get("profitable_excluding_best_trade"))
    results.append(
        GateResult(
            number=5,
            name="several profitable blocks and no single-trade dependence",
            passed=bool(
                len(profitable_blocks) >= config_min_profitable_blocks and robust_without_best
            ),
            detail=(
                f"{len(profitable_blocks)} of {len(blocks)} {config_block_sessions}-session "
                f"blocks profitable; profitable without the single best trade: "
                f"{robust_without_best}"
            ),
            evidence={"blocks": blocks},
        )
    )

    sector_concentration = contributor_concentration(combined, key="sector")
    top_share = concentration.get("top_share")
    measurable = isinstance(top_share, float) and math.isfinite(top_share)
    concentrated = measurable and top_share > 0.5
    # No trades means nothing to report with the strongest contributor removed,
    # so the gate cannot pass. Gates fail closed on absent evidence.
    gate_six_passed = bool(
        combined
        and measurable
        and ((not concentrated) or bool(concentration_claim_restricted))
    )
    if not combined:
        gate_six_detail = "no executed trades: concentration cannot be assessed"
    elif not measurable:
        gate_six_detail = (
            "summed net return is zero, so no contributor share is measurable"
        )
    else:
        gate_six_detail = (
            f"top issuer {concentration.get('top_key')} explains "
            f"{top_share:.1%} of the summed net return"
            + (
                "; a concentrated result must have its claim restricted rather than "
                "generalised to the whole watchlist"
                if concentrated
                else ""
            )
        )
    results.append(
        GateResult(
            number=6,
            name="result not explained by one issuer or sector",
            passed=gate_six_passed,
            detail=gate_six_detail,
            evidence={"by_issuer": concentration, "by_sector": sector_concentration},
        )
    )

    brier_ok = (
        math.isfinite(calibration_brier)
        and math.isfinite(base_rate_brier)
        and calibration_brier < base_rate_brier
    )
    coverage_ok = math.isfinite(coverage_p10_p90) and 0.60 <= coverage_p10_p90 <= 0.95
    results.append(
        GateResult(
            number=7,
            name="calibrated probabilities beat the constant base rate out of sample",
            passed=bool(brier_ok and coverage_ok),
            detail=(
                f"Brier {calibration_brier:.5f} versus base-rate {base_rate_brier:.5f}; "
                f"observed p10-p90 coverage {coverage_p10_p90:.3f} against the nominal 0.80"
            ),
            evidence={
                "brier": calibration_brier,
                "base_rate_brier": base_rate_brier,
                "coverage_p10_p90": coverage_p10_p90,
                "reliability": [dict(row) for row in reliability],
            },
        )
    )

    results.append(
        GateResult(
            number=8,
            name="improvement survives ablation, costs and capital constraints",
            passed=bool(ablation_survived) if ablation_survived is not None else False,
            detail=(
                "requires an explicit recorded ablation result; absent one this gate "
                "does not pass"
                if ablation_survived is None
                else f"recorded ablation outcome: {ablation_survived}"
            ),
            evidence={"ablation_survived": ablation_survived},
        )
    )

    notes = [
        "These gates are conservative research acceptance rules, not a theorem that "
        "future returns will be positive.",
        "A bootstrap interval describes sampling variability within the observed "
        "history; it does not create regimes the sample never contained.",
    ]
    report = GateReport(
        checkpoint=checkpoint,
        assessed_at=assessed_at,
        results=tuple(results),
        notes=tuple(notes),
    )
    return report
