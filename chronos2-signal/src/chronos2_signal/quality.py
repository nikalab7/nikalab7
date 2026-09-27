"""Per-origin data quality and eligibility.

Eligibility is evaluated at every decision origin, before any forecast is
looked at. Two properties matter more than the individual thresholds:

* The mask is **common to every variant**. B0 and the Chronos variants see the
  same rows on the same dates, so a comparison between them is a comparison of
  models rather than of samples.
* Nothing is repaired. A symbol that fails is skipped and recorded; it is never
  replaced by another symbol after results are seen, and a missing input is
  never imputed into a tradable price.

Screens run in **as-of units**. A price floor applied to restated history would
reject a stock that comfortably passed it at the time.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .actions import ActionLedger, SplitAudit, to_asof_units
from .config import DesignConfig
from .market import BarPanel, DailyPanel

__all__ = [
    "EligibilityResult",
    "OriginQuality",
    "evaluate_eligibility",
    "liquidity_metrics",
]


@dataclass(frozen=True)
class EligibilityResult:
    """Whether one symbol may be considered at one origin, and why."""

    symbol: str
    origin_session: dt.date
    eligible: bool
    failures: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def with_failure(self, reason: str) -> "EligibilityResult":
        return EligibilityResult(
            symbol=self.symbol,
            origin_session=self.origin_session,
            eligible=False,
            failures=(*self.failures, reason),
            metrics=self.metrics,
        )


@dataclass(frozen=True)
class OriginQuality:
    """Aggregate eligibility outcome for one origin."""

    origin_session: dt.date
    results: tuple[EligibilityResult, ...]

    @property
    def eligible_symbols(self) -> tuple[str, ...]:
        return tuple(result.symbol for result in self.results if result.eligible)

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_symbols)

    def failures_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            for reason in result.failures:
                key = reason.split(":", 1)[0]
                counts[key] = counts.get(key, 0) + 1
        return counts


def liquidity_metrics(
    daily: DailyPanel,
    origin_session: dt.date,
    ledger: ActionLedger,
    *,
    window: int,
) -> dict[str, float]:
    """As-of price and median dollar volume over the trailing window.

    Dollar volume is invariant to the split convention -- a two-for-one halves
    the price and doubles the quantity -- but the price level is not, so both
    are converted explicitly rather than assumed comparable.
    """
    history = daily.up_to(origin_session)
    if not len(history):
        return {"asof_price": float("nan"), "median_dollar_volume": float("nan"), "sessions": 0.0}

    closes = to_asof_units(history.column("close"), ledger, origin_session, kind="price")
    volumes = to_asof_units(
        history.column("volume"), ledger, origin_session, kind="quantity"
    )
    tail_closes = closes[-window:]
    tail_volumes = volumes[-window:]
    dollar = tail_closes * tail_volumes
    finite = dollar[np.isfinite(dollar)]
    return {
        "asof_price": float(closes[-1]) if np.isfinite(closes[-1]) else float("nan"),
        "median_dollar_volume": float(np.median(finite)) if finite.size else float("nan"),
        "sessions": float(len(history)),
    }


def evaluate_eligibility(
    *,
    symbol: str,
    origin_session: dt.date,
    config: DesignConfig,
    hourly: BarPanel,
    daily: DailyPanel,
    market_hourly: BarPanel,
    sector_hourly: BarPanel,
    action_ledger: ActionLedger,
    split_audits: Sequence[SplitAudit] = (),
    issuer_id: str | None = None,
    sector_suppressed: bool = False,
) -> EligibilityResult:
    """Apply the registered eligibility gates to one symbol at one origin.

    All checks are evaluated so that the failure list is complete. Reporting
    only the first failure would hide how many symbols fail for several
    independent reasons at once.
    """
    universe = config.universe
    failures: list[str] = []
    metrics: dict[str, float] = {}

    if issuer_id is None or not str(issuer_id).strip():
        failures.append("issuer_mapping: no stable issuer identifier")
    if sector_suppressed:
        failures.append("sector_suppressed: required sector input unavailable at this origin")

    liquidity = liquidity_metrics(
        daily, origin_session, action_ledger, window=universe.liquidity_window_sessions
    )
    metrics.update(liquidity)

    price = liquidity["asof_price"]
    if not math.isfinite(price):
        failures.append("price: no finite as-of close at the origin")
    elif price < universe.min_asof_price_usd:
        failures.append(
            f"price: as-of close {price:.2f} below {universe.min_asof_price_usd:.2f}"
        )

    dollar_volume = liquidity["median_dollar_volume"]
    if not math.isfinite(dollar_volume):
        failures.append("liquidity: no finite dollar volume in the trailing window")
    elif dollar_volume < universe.min_median_daily_dollar_volume_usd:
        failures.append(
            f"liquidity: trailing-{universe.liquidity_window_sessions} median dollar "
            f"volume {dollar_volume:,.0f} below "
            f"{universe.min_median_daily_dollar_volume_usd:,.0f}"
        )

    daily_history = daily.up_to(origin_session)
    valid_daily = daily_history.valid_observations()
    metrics["valid_daily_observations"] = float(valid_daily)
    if valid_daily < universe.min_daily_observations:
        failures.append(
            f"daily_history: {valid_daily} valid daily observations below "
            f"{universe.min_daily_observations}"
        )

    required_bars = universe.common_hourly_history_bars
    metrics["scheduled_hourly_bars"] = float(len(hourly))
    if len(hourly) < required_bars:
        failures.append(
            f"hourly_schedule: {len(hourly)} scheduled bars below the common "
            f"{required_bars}-bar history mask"
        )
    else:
        observed = hourly.observed_fraction(required_bars)
        metrics["observed_fraction"] = observed
        if observed < universe.min_observed_fraction:
            failures.append(
                f"hourly_coverage: {observed:.4f} observed below "
                f"{universe.min_observed_fraction:.4f} over the last {required_bars} bars"
            )

    if universe.latest_session_complete_required:
        for label, panel in (
            ("target", hourly),
            ("market", market_hourly),
            ("sector", sector_hourly),
        ):
            if not panel.session_complete(origin_session):
                failures.append(
                    f"latest_session: {label} series is incomplete on "
                    f"{origin_session.isoformat()}"
                )

    finite_failures = _check_finite_ohlcv(hourly, required_bars)
    failures.extend(finite_failures)

    unusable = [audit for audit in split_audits if not audit.usable]
    if unusable:
        first = unusable[0]
        failures.append(
            f"corporate_action: unresolved split on {first.action.ex_date.isoformat()} "
            f"({first.verdict.value})"
        )

    return EligibilityResult(
        symbol=symbol,
        origin_session=origin_session,
        eligible=not failures,
        failures=tuple(failures),
        metrics=metrics,
    )


def _check_finite_ohlcv(panel: BarPanel, window: int) -> list[str]:
    """Structural checks on the observed rows of a panel's recent window.

    A bar the provider never returned is a mask, which is legitimate. A bar it
    *did* return with impossible values is a data error, which is not.
    """
    failures: list[str] = []
    tail = panel.frame.iloc[-window:] if window < len(panel.frame) else panel.frame
    observed = tail[tail["observed"].astype(bool)]
    if not len(observed):
        return ["ohlcv: no observed bars in the recent window"]

    values = {
        name: observed[name].to_numpy(dtype=float)
        for name in ("open", "high", "low", "close", "volume")
    }
    for name, array in values.items():
        if not np.isfinite(array).all():
            failures.append(f"ohlcv: non-finite {name} on an observed bar")
    if failures:
        return failures

    if (values["high"] < values["low"] - 1e-9).any():
        failures.append("ohlcv: high below low on an observed bar")
    for name in ("open", "close"):
        if (values[name] > values["high"] + 1e-9).any() or (
            values[name] < values["low"] - 1e-9
        ).any():
            failures.append(f"ohlcv: {name} outside the bar range")
    if (values["close"] <= 0.0).any() or (values["open"] <= 0.0).any():
        failures.append("ohlcv: non-positive price on an observed bar")
    if (values["volume"] < 0.0).any():
        failures.append("ohlcv: negative volume on an observed bar")
    return failures
