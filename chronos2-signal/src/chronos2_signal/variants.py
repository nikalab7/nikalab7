"""The five registered learned variants and the fixed controls.

All five variants share dates, the universe mask, features where applicable,
labels, costs, decision-model settings and the alert policy. The only thing
that differs is the question each one answers:

===== ======================================= =========================================
ID    Variant                                 Question
===== ======================================= =========================================
B0    14 non-forecast features + breadth flag Is market and volume information enough?
C128  full channels, 128-bar context          Does short history help?
C256  full channels, 256-bar context          The default hypothesis.
C512  full channels, 512-bar context          Is extra historical context useful?
U256  target and calendar only, 256 bars      Do market/volume covariates add value?
===== ======================================= =========================================

The controls are declared here too, so they cannot be chosen after the fact:
a fixed relative-momentum rule, an exposure-matched market benchmark and
zero-return cash. They are not a menu from which the easiest comparison gets
picked later.

LightGBM and fine-tuning are outside these five. Admitting either requires a
new registered version with additional untouched evaluation data.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .config import DesignConfig
from .evaluation import DailySeries

__all__ = [
    "VariantError",
    "VariantSpec",
    "REGISTERED_VARIANTS",
    "variant_spec",
    "MomentumCandidate",
    "rank_momentum_candidates",
    "exposure_matched_series",
    "cash_series",
]


class VariantError(RuntimeError):
    """Raised when an unregistered variant or control is requested."""


@dataclass(frozen=True)
class VariantSpec:
    """One registered learned variant."""

    name: str
    question: str
    uses_forecast: bool
    context_length: int | None
    include_covariates: bool

    @property
    def requires_chronos(self) -> bool:
        return self.uses_forecast


REGISTERED_VARIANTS: Mapping[str, VariantSpec] = {
    "B0": VariantSpec(
        name="B0",
        question="Is market and volume information enough?",
        uses_forecast=False,
        context_length=None,
        include_covariates=False,
    ),
    "C128": VariantSpec(
        name="C128",
        question="Does short history help?",
        uses_forecast=True,
        context_length=128,
        include_covariates=True,
    ),
    "C256": VariantSpec(
        name="C256",
        question="The default hypothesis.",
        uses_forecast=True,
        context_length=256,
        include_covariates=True,
    ),
    "C512": VariantSpec(
        name="C512",
        question="Is extra historical context useful?",
        uses_forecast=True,
        context_length=512,
        include_covariates=True,
    ),
    "U256": VariantSpec(
        name="U256",
        question="Do the market and volume covariates add value?",
        uses_forecast=True,
        context_length=256,
        include_covariates=False,
    ),
}


def variant_spec(name: str) -> VariantSpec:
    """Look up a registered variant.

    Raises:
        VariantError: For anything outside the five registered variants. A new
            variant needs a new design version, not a new string.
    """
    try:
        return REGISTERED_VARIANTS[name]
    except KeyError as exc:
        raise VariantError(
            f"{name!r} is not a registered variant; the set is "
            f"{sorted(REGISTERED_VARIANTS)}"
        ) from exc


@dataclass(frozen=True)
class MomentumCandidate:
    """One symbol scored by the fixed relative-momentum control."""

    symbol: str
    issuer_id: str
    sector: str
    stock_minus_sector_5s: float


def rank_momentum_candidates(
    candidates: Sequence[MomentumCandidate],
    *,
    config: DesignConfig,
    open_sectors: Mapping[str, int] | None = None,
    open_issuers: Sequence[str] = (),
    open_position_count: int = 0,
) -> list[MomentumCandidate]:
    """The fixed relative-momentum control's picks for one origin.

    Rank positive trailing five-session stock-minus-sector returns and take up
    to the registered maximum, subject to the same eligibility and capacity
    rules as the candidates. The hold and the cost assumptions are identical,
    which is the only way the comparison means anything.

    Eligibility is the caller's job: only already-eligible symbols should be
    passed in, so that every system sees the same mask.
    """
    portfolio = config.portfolio
    taken = dict(open_sectors or {})
    issuers = set(open_issuers)

    positive = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate.stock_minus_sector_5s)
        and candidate.stock_minus_sector_5s > 0.0
        and candidate.issuer_id not in issuers
    ]
    ranked = sorted(
        positive,
        key=lambda item: (-item.stock_minus_sector_5s, item.symbol),
    )

    picks: list[MomentumCandidate] = []
    for candidate in ranked:
        if len(picks) >= portfolio.max_new_alerts_per_origin:
            break
        if open_position_count + len(picks) >= portfolio.max_open_positions:
            break
        if taken.get(candidate.sector, 0) >= portfolio.max_open_per_sector:
            continue
        taken[candidate.sector] = taken.get(candidate.sector, 0) + 1
        picks.append(candidate)
    return picks


def exposure_matched_series(
    *,
    name: str,
    sessions: Sequence[dt.date],
    market_returns: Sequence[float],
    candidate_gross_exposure: Sequence[float],
) -> DailySeries:
    """A market benchmark scaled to the candidate's own gross exposure.

    The exposure applied on session ``t`` is the one observed at the previous
    close, which is what a portfolio could actually have matched. Using the
    same session's exposure would quietly let the benchmark know the day's
    allocation in advance.

    This control exists to answer one question: whether market exposure, rather
    than forecasting, explains the result.
    """
    if not (len(sessions) == len(market_returns) == len(candidate_gross_exposure)):
        raise VariantError("exposure-matched series inputs must have equal length")
    market = np.asarray(market_returns, dtype=float)
    exposure = np.asarray(candidate_gross_exposure, dtype=float)
    lagged = np.empty_like(exposure)
    if exposure.size:
        lagged[0] = 0.0
        lagged[1:] = exposure[:-1]
    returns = lagged * market
    return DailySeries(name=name, sessions=tuple(sessions), returns=returns)


def cash_series(name: str, sessions: Sequence[dt.date]) -> DailySeries:
    """Zero-return cash, treated identically to every other system."""
    return DailySeries(
        name=name, sessions=tuple(sessions), returns=np.zeros(len(sessions), dtype=float)
    )
