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
from .policy import CorrelationLookup, correlation_block

__all__ = [
    "VariantError",
    "VariantSpec",
    "REGISTERED_VARIANTS",
    "variant_spec",
    "MomentumCandidate",
    "rank_momentum_candidates",
    "MarketLeg",
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
    open_symbols: Sequence[str] = (),
    correlation: CorrelationLookup | None = None,
) -> list[MomentumCandidate]:
    """The fixed relative-momentum control's picks for one origin.

    Rank positive trailing five-session stock-minus-sector returns and take up
    to the registered maximum, subject to the same eligibility and capacity
    rules as the candidates. The hold and the cost assumptions are identical,
    which is the only way the comparison means anything.

    Eligibility is the caller's job: only already-eligible symbols should be
    passed in, so that every system sees the same mask. The correlation cap is
    the candidates' own rule (:func:`~chronos2_signal.policy.correlation_block`),
    applied against the open positions and the picks already made; without a
    correlation source, a second position is blocked rather than assumed
    uncorrelated.
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
        if correlation_block(
            candidate.symbol,
            [*open_symbols, *(pick.symbol for pick in picks)],
            correlation,
            config=config,
        ):
            continue
        taken[candidate.sector] = taken.get(candidate.sector, 0) + 1
        picks.append(candidate)
    return picks


@dataclass(frozen=True)
class MarketLeg:
    """Capital one position had in the market during one session.

    Attributes:
        session: The session.
        notional: The position's capital at the start of the leg: its cost
            basis on the entry session, its market value at the previous close
            afterwards.
        from_open: True on the entry session, whose leg runs from the official
            open to the close; otherwise the leg runs close to close.
    """

    session: dt.date
    notional: float
    from_open: bool


def exposure_matched_series(
    *,
    name: str,
    sessions: Sequence[dt.date],
    legs: Sequence[MarketLeg],
    equity_before: Mapping[dt.date, float],
    market_open: Mapping[dt.date, float],
    market_close: Mapping[dt.date, float],
) -> DailySeries:
    """The market, held with the candidate's capital over the candidate's intervals.

    Every leg of every position is replicated in the market proxy: the same
    capital, from the official open on the entry session and close to close
    afterwards. A D1 entry is decided at D0's after-close run, so its exposure
    is known before the open and matching it is not look-ahead -- whereas
    lagging the exposure by a session would leave every entry day unmatched and
    credit a zero-skill strategy with the market's drift.

    This control exists to answer one question: whether market exposure, rather
    than forecasting, explains the result. A missing proxy price contributes
    zero for that leg.
    """
    index = {session: position for position, session in enumerate(sessions)}
    returns = np.zeros(len(sessions), dtype=float)
    for leg in legs:
        position = index.get(leg.session)
        if position is None:
            raise VariantError(f"leg on {leg.session.isoformat()} is outside the date index")
        if leg.from_open:
            start = market_open.get(leg.session)
        else:
            start = market_close.get(sessions[position - 1]) if position > 0 else None
        end = market_close.get(leg.session)
        base = equity_before.get(leg.session)
        if not start or not end or not base or start <= 0.0 or base <= 0.0:
            continue
        returns[position] += leg.notional / base * (end / start - 1.0)
    return DailySeries(name=name, sessions=tuple(sessions), returns=returns)


def cash_series(name: str, sessions: Sequence[dt.date]) -> DailySeries:
    """Zero-return cash, treated identically to every other system."""
    return DailySeries(
        name=name, sessions=tuple(sessions), returns=np.zeros(len(sessions), dtype=float)
    )
