"""Corporate actions, split consistency and wealth accounting.

Two conversions are kept strictly apart:

* **Feature units.** The canonical series is split-consistent, which means a
  split leaves no artificial jump in the price path. Dividend-adjusted closes
  are never used here, so a distribution cannot masquerade as momentum.
* **As-of execution units.** Eligibility screens and the paper ledger use the
  share and cash units that were current at the origin. A stock that has split
  three times since is not screened at today's nominal price.

The provider's own convention is *verified*, never assumed. ``auto_adjust=False``
does not establish that a returned field is an untouched historical quote, so
each split is audited against the price and volume series. If the audit cannot
decide, the episode is quarantined: a manufactured return is worse than a
missing one.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "ActionError",
    "QuarantinedEpisode",
    "CorporateAction",
    "ActionLedger",
    "SplitVerdict",
    "SplitAudit",
    "audit_split_convention",
    "to_split_consistent",
    "to_asof_units",
    "HoldingCosts",
    "reprice_net_return",
    "HoldingAccount",
    "label_net_return",
    "max_adverse_excursion",
]


class ActionError(RuntimeError):
    """Raised on an inconsistent or impossible corporate-action record."""


class QuarantinedEpisode(ActionError):
    """Raised when an episode cannot be reconstructed honestly.

    Callers drop the affected origins from the study and record the reason.
    They must not fall back to unadjusted prices.
    """


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """One split or cash distribution.

    Attributes:
        symbol: Issuer ticker.
        ex_date: Ex-date, i.e. the first session on which a buyer is not
            entitled to the distribution.
        action_type: ``"split"`` or ``"dividend"``.
        value: Split ratio (``2.0`` for a two-for-one) or cash per share.
        available_at: When the system first observed the record. A record the
            system had not seen is not usable at an earlier origin.
    """

    symbol: str
    ex_date: dt.date
    action_type: str
    value: float
    available_at: dt.datetime | None = None

    def __post_init__(self) -> None:
        if self.action_type not in {"split", "dividend"}:
            raise ActionError(f"unknown action type {self.action_type!r}")
        if not math.isfinite(self.value):
            raise ActionError(f"non-finite action value for {self.symbol} {self.ex_date}")
        if self.action_type == "split" and self.value <= 0.0:
            raise ActionError(
                f"split ratio must be positive, got {self.value} for {self.symbol}"
            )
        if self.action_type == "dividend" and self.value < 0.0:
            raise ActionError(
                f"dividend must be non-negative, got {self.value} for {self.symbol}"
            )


@dataclass(frozen=True)
class ActionLedger:
    """All known actions for one symbol, with point-in-time filtering."""

    symbol: str
    actions: tuple[CorporateAction, ...] = ()

    @classmethod
    def from_rows(cls, symbol: str, rows: Iterable[Mapping[str, object]]) -> "ActionLedger":
        parsed: list[CorporateAction] = []
        for row in rows:
            ex_date = row["ex_date"]
            available = row.get("available_at")
            parsed.append(
                CorporateAction(
                    symbol=symbol,
                    ex_date=(
                        ex_date
                        if isinstance(ex_date, dt.date) and not isinstance(ex_date, dt.datetime)
                        else dt.date.fromisoformat(str(ex_date)[:10])
                    ),
                    action_type=str(row["action_type"]),
                    value=float(row["value"]),  # type: ignore[arg-type]
                    available_at=(
                        available
                        if isinstance(available, dt.datetime) or available is None
                        else dt.datetime.fromisoformat(str(available))
                    ),
                )
            )
        return cls(symbol=symbol, actions=tuple(sorted(parsed, key=lambda a: (a.ex_date, a.action_type))))

    def known_at(self, as_of: dt.datetime) -> "ActionLedger":
        """Actions the system had already observed at ``as_of``."""
        return ActionLedger(
            symbol=self.symbol,
            actions=tuple(
                action
                for action in self.actions
                if action.available_at is None or action.available_at <= as_of
            ),
        )

    def splits(self) -> tuple[CorporateAction, ...]:
        return tuple(a for a in self.actions if a.action_type == "split")

    def dividends(self) -> tuple[CorporateAction, ...]:
        return tuple(a for a in self.actions if a.action_type == "dividend")

    def split_factor_after(self, session: dt.date) -> float:
        """Product of split ratios with an ex-date strictly after ``session``.

        Multiplying a series that is restated to the latest units by this
        factor recovers the nominal prices quoted as of ``session``.
        """
        factor = 1.0
        for split in self.splits():
            if split.ex_date > session:
                factor *= split.value
        return factor

    def split_factor_in(self, start_exclusive: dt.date, end_inclusive: dt.date) -> float:
        """Product of split ratios with ``start < ex_date <= end``."""
        factor = 1.0
        for split in self.splits():
            if start_exclusive < split.ex_date <= end_inclusive:
                factor *= split.value
        return factor

    def dividends_in(
        self, start_exclusive: dt.date, end_inclusive: dt.date
    ) -> tuple[CorporateAction, ...]:
        """Dividends a holder entitled by an entry at ``start`` actually earns.

        A buyer on the ex-date is not entitled, so the window is open at the
        start. A holder still on the books at the ex-date is entitled even if
        the position closes later that session, so the window is closed at the
        end.
        """
        return tuple(
            dividend
            for dividend in self.dividends()
            if start_exclusive < dividend.ex_date <= end_inclusive
        )

    def has_action_between(self, start_exclusive: dt.date, end_inclusive: dt.date) -> bool:
        return bool(
            self.split_factor_in(start_exclusive, end_inclusive) != 1.0
            or self.dividends_in(start_exclusive, end_inclusive)
        )


class SplitVerdict(str, Enum):
    """Whether the provider series already reflects a split."""

    #: The series is restated: no artificial jump at the ex-date.
    ALREADY_APPLIED = "already_applied"
    #: The series shows the raw quoted drop; a factor must be applied.
    NOT_APPLIED = "not_applied"
    #: Cannot be decided from the data. Quarantine the episode.
    AMBIGUOUS = "ambiguous"
    #: Both a price jump and a further factor would be needed: a double count.
    DOUBLE_APPLIED = "double_applied"


@dataclass(frozen=True)
class SplitAudit:
    """Result of auditing one split against the price and volume series."""

    action: CorporateAction
    verdict: SplitVerdict
    price_ratio: float | None
    volume_ratio: float | None
    detail: str

    @property
    def usable(self) -> bool:
        return self.verdict in (SplitVerdict.ALREADY_APPLIED, SplitVerdict.NOT_APPLIED)


def audit_split_convention(
    split: CorporateAction,
    sessions: Sequence[dt.date],
    closes: Sequence[float],
    volumes: Sequence[float] | None = None,
    *,
    tolerance: float = 0.08,
) -> SplitAudit:
    """Decide whether ``closes`` already reflects ``split``.

    The test is the close-to-close step across the ex-date. A restated series
    steps by roughly one; a raw quoted series steps by roughly the split ratio.
    A ratio that is close to neither, or a split too small to separate from an
    ordinary move, returns :attr:`SplitVerdict.AMBIGUOUS`.

    ``tolerance`` is a relative band around each hypothesis. It is an
    engineering starting assumption, not a measured property of the feed.
    """
    index = {session: position for position, session in enumerate(sessions)}
    position = index.get(split.ex_date)
    if position is None or position == 0:
        return SplitAudit(
            action=split,
            verdict=SplitVerdict.AMBIGUOUS,
            price_ratio=None,
            volume_ratio=None,
            detail="ex-date not inside the supplied session window with a predecessor",
        )
    previous_close = float(closes[position - 1])
    ex_close = float(closes[position])
    if not (math.isfinite(previous_close) and math.isfinite(ex_close)) or ex_close <= 0.0:
        return SplitAudit(
            action=split,
            verdict=SplitVerdict.AMBIGUOUS,
            price_ratio=None,
            volume_ratio=None,
            detail="non-finite or non-positive closes around the ex-date",
        )

    price_ratio = previous_close / ex_close
    volume_ratio: float | None = None
    if volumes is not None:
        previous_volume = float(volumes[position - 1])
        ex_volume = float(volumes[position])
        if math.isfinite(previous_volume) and math.isfinite(ex_volume) and previous_volume > 0:
            volume_ratio = ex_volume / previous_volume

    ratio = split.value
    if abs(ratio - 1.0) < 2.0 * tolerance:
        return SplitAudit(
            action=split,
            verdict=SplitVerdict.AMBIGUOUS,
            price_ratio=price_ratio,
            volume_ratio=volume_ratio,
            detail=(
                f"split ratio {ratio} is too close to 1 to separate from an ordinary "
                "price move at this tolerance"
            ),
        )

    near_one = abs(price_ratio - 1.0) <= tolerance
    near_ratio = abs(price_ratio / ratio - 1.0) <= tolerance
    near_squared = abs(price_ratio / (ratio * ratio) - 1.0) <= tolerance

    if near_one and not near_ratio:
        return SplitAudit(
            action=split,
            verdict=SplitVerdict.ALREADY_APPLIED,
            price_ratio=price_ratio,
            volume_ratio=volume_ratio,
            detail="no price step at the ex-date: the series is restated",
        )
    if near_ratio and not near_one:
        return SplitAudit(
            action=split,
            verdict=SplitVerdict.NOT_APPLIED,
            price_ratio=price_ratio,
            volume_ratio=volume_ratio,
            detail=f"price stepped by about the split ratio {ratio}",
        )
    if near_squared:
        return SplitAudit(
            action=split,
            verdict=SplitVerdict.DOUBLE_APPLIED,
            price_ratio=price_ratio,
            volume_ratio=volume_ratio,
            detail=(
                f"price stepped by about the squared ratio {ratio * ratio}: a split "
                "appears to have been applied twice"
            ),
        )
    return SplitAudit(
        action=split,
        verdict=SplitVerdict.AMBIGUOUS,
        price_ratio=price_ratio,
        volume_ratio=volume_ratio,
        detail=(
            f"price ratio {price_ratio:.4f} matches neither 1 nor the split ratio "
            f"{ratio} within {tolerance}"
        ),
    )


def to_split_consistent(
    sessions: Sequence[dt.date],
    values: Sequence[float],
    audits: Sequence[SplitAudit],
    *,
    kind: str = "price",
) -> np.ndarray:
    """Return a split-consistent copy of ``values``.

    Splits the provider has already applied are left alone -- applying them a
    second time is the classic way to manufacture a return. Splits the provider
    has not applied are restated to the latest units. An ambiguous or
    double-applied audit raises :class:`QuarantinedEpisode`.

    ``kind`` selects the direction: prices divide by the split factor, share
    quantities and volumes multiply by it.
    """
    if kind not in {"price", "quantity"}:
        raise ActionError(f"unknown series kind {kind!r}")
    result = np.asarray(values, dtype=float).copy()
    if result.size != len(sessions):
        raise ActionError("sessions and values must have the same length")

    for audit in audits:
        if not audit.usable:
            raise QuarantinedEpisode(
                f"{audit.action.symbol}: split on {audit.action.ex_date.isoformat()} "
                f"is {audit.verdict.value} ({audit.detail}); quarantine this episode "
                "rather than reconstruct a return from it"
            )
        if audit.verdict is SplitVerdict.ALREADY_APPLIED:
            continue
        ratio = audit.action.value
        # Rows strictly before the ex-date are still quoted in pre-split units.
        mask = np.asarray([session < audit.action.ex_date for session in sessions], dtype=bool)
        if kind == "price":
            result[mask] = result[mask] / ratio
        else:
            result[mask] = result[mask] * ratio
    return result


def to_asof_units(
    values: Sequence[float],
    ledger: ActionLedger,
    asof_session: dt.date,
    *,
    kind: str = "price",
) -> np.ndarray:
    """Convert a latest-units series into the units quoted at ``asof_session``.

    Screens must use as-of units. A USD 12 stock that later split four-for-one
    shows as USD 3 in restated history and would fail a USD 10 price floor it
    actually passed at the time.
    """
    if kind not in {"price", "quantity"}:
        raise ActionError(f"unknown series kind {kind!r}")
    factor = ledger.split_factor_after(asof_session)
    array = np.asarray(values, dtype=float)
    return array * factor if kind == "price" else array / factor


@dataclass(frozen=True)
class HoldingCosts:
    """Cost assumptions for one round trip.

    Test assumptions, not measured spreads and not guaranteed execution.
    """

    buy_slippage: float
    sell_slippage: float
    explicit_fee_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("buy_slippage", "sell_slippage", "explicit_fee_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ActionError(f"{name} must be a finite non-negative fraction")


def reprice_net_return(
    net_return: float,
    *,
    base: HoldingCosts,
    target: HoldingCosts,
) -> float:
    """Restate a net return under a different cost scenario.

    Recovers the implied gross price ratio from ``net_return`` under ``base``
    and reapplies ``target``. Exact for a single realised trade, and the
    documented approximation when applied to an estimate -- which is how the
    stress-cost alert condition is evaluated without refitting a second model.
    """
    if not math.isfinite(net_return):
        return float("nan")
    gross_factor = (net_return + 1.0 + base.explicit_fee_fraction) * (
        (1.0 + base.buy_slippage) / (1.0 - base.sell_slippage)
    )
    repriced = gross_factor * (
        (1.0 - target.sell_slippage) / (1.0 + target.buy_slippage)
    )
    return repriced - 1.0 - target.explicit_fee_fraction


@dataclass(frozen=True)
class HoldingAccount:
    """Complete accounting for one reference holding period.

    Every number here comes from the accounting engine, never from a forecast
    quantile.

    Prices and ``restated_shares`` are in split-consistent (latest) units, in
    which a split moves neither the price path nor the position's economic
    size. The nominal quantities record what a brokerage statement would have
    shown at each end of the hold, which is what
    :mod:`chronos2_signal.portfolio` keeps its cash ledger in.
    """

    entry_session: dt.date
    exit_session: dt.date
    entry_reference_price: float
    exit_reference_price: float
    entry_fill_price: float
    exit_fill_price: float
    restated_shares: float
    nominal_shares_at_entry: float
    nominal_shares_at_exit: float
    share_adjustment: float
    cash_dividends: float
    cost_basis: float
    proceeds: float
    net_return: float
    gross_return: float
    explicit_fee_fraction: float
    breakeven_total_cost_fraction: float
    actions_applied: tuple[str, ...] = ()

    def reconciles(self, *, tolerance: float = 1e-12) -> bool:
        """Whether wealth in and out agree with the reported net return."""
        wealth_change = self.proceeds + self.cash_dividends - self.cost_basis
        implied = wealth_change / self.cost_basis - self.explicit_fee_fraction
        return abs(implied - self.net_return) <= tolerance * max(1.0, abs(implied))

    def nominal_split_consistent(self, *, tolerance: float = 1e-12) -> bool:
        """Whether the nominal share counts agree with the split factor.

        This is the statement that a split creates no wealth: the quantity
        changes by exactly the ratio and nothing else does.
        """
        expected = self.nominal_shares_at_entry * self.share_adjustment
        return abs(expected - self.nominal_shares_at_exit) <= tolerance * max(
            1.0, abs(expected)
        )


def label_net_return(
    *,
    entry_session: dt.date,
    exit_session: dt.date,
    entry_open_price: float,
    exit_close_price: float,
    costs: HoldingCosts,
    ledger: ActionLedger | None = None,
    shares: float = 1.0,
) -> HoldingAccount:
    """Realised net return of the reference policy for one holding period.

    Prices are in split-consistent units, so a split inside the hold does not
    move the price path. The share count is restated by the same factor, which
    keeps the notional unchanged; that is the check that a split creates no
    wealth. Dividends are added once, and only for the ex-dates a holder
    entering at ``entry_session`` actually earns.

    With no corporate action this reduces exactly to the registered label

    ``R_net = C_exit*(1-s_sell) / (O_entry*(1+s_buy)) - 1 - fee``.
    """
    if exit_session < entry_session:
        raise ActionError("exit session precedes entry session")
    if not (math.isfinite(entry_open_price) and math.isfinite(exit_close_price)):
        raise ActionError("entry and exit prices must be finite")
    if entry_open_price <= 0.0 or exit_close_price <= 0.0:
        raise ActionError("entry and exit prices must be positive")
    if shares <= 0.0:
        raise ActionError("share quantity must be positive")

    ledger = ledger or ActionLedger(symbol="", actions=())
    split_factor = ledger.split_factor_in(entry_session, exit_session)
    earned = ledger.dividends_in(entry_session, exit_session)

    entry_fill = entry_open_price * (1.0 + costs.buy_slippage)
    exit_fill = exit_close_price * (1.0 - costs.sell_slippage)
    cost_basis = shares * entry_fill

    # In split-consistent units the quoted price already reflects every split,
    # so a split changes neither the price path nor the position's economic
    # size: the restated share count is constant across the hold.
    proceeds = shares * exit_fill

    # The nominal counts are what a statement would show. They exist only to be
    # reconciled against the portfolio ledger, which works in as-of units.
    nominal_at_entry = shares / ledger.split_factor_after(entry_session)
    nominal_at_exit = shares / ledger.split_factor_after(exit_session)

    # A dividend per share is quoted in the units current at its ex-date, and is
    # paid on the nominal quantity held going into that date -- which is before
    # any split sharing the same ex-date.
    dividend_cash = 0.0
    for dividend in earned:
        record_date_factor = ledger.split_factor_after(
            dividend.ex_date - dt.timedelta(days=1)
        )
        nominal_shares_on_record = shares / record_date_factor
        dividend_cash += nominal_shares_on_record * dividend.value

    net_return = (
        proceeds + dividend_cash - cost_basis
    ) / cost_basis - costs.explicit_fee_fraction
    gross_return = (
        exit_close_price / entry_open_price
        - 1.0
        + dividend_cash / (shares * entry_open_price)
    )
    # Total round-trip cost, as a fraction of the entry notional, that would
    # exactly consume this trade's gross profit. Reported so a thin margin is
    # visible instead of implied.
    breakeven = gross_return

    applied: list[str] = []
    if split_factor != 1.0:
        applied.append(f"split x{split_factor:g}")
    if earned:
        applied.append(f"{len(earned)} dividend(s) totalling {dividend_cash:.6f}")

    return HoldingAccount(
        entry_session=entry_session,
        exit_session=exit_session,
        entry_reference_price=entry_open_price,
        exit_reference_price=exit_close_price,
        entry_fill_price=entry_fill,
        exit_fill_price=exit_fill,
        restated_shares=shares,
        nominal_shares_at_entry=nominal_at_entry,
        nominal_shares_at_exit=nominal_at_exit,
        share_adjustment=split_factor,
        cash_dividends=dividend_cash,
        cost_basis=cost_basis,
        proceeds=proceeds,
        net_return=net_return,
        gross_return=gross_return,
        explicit_fee_fraction=costs.explicit_fee_fraction,
        breakeven_total_cost_fraction=breakeven,
        actions_applied=tuple(applied),
    )


def max_adverse_excursion(
    entry_fill_price: float,
    lows: Sequence[float],
) -> float:
    """Worst drawdown from the entry fill reached inside the hold.

    Recorded for diagnosis only. The protocol has no stop in version 1, so this
    value must never be used as an exit price: doing so would credit the
    strategy with an exit it never took.
    """
    finite = [float(value) for value in lows if math.isfinite(value)]
    if not finite or entry_fill_price <= 0.0:
        return float("nan")
    return min(0.0, min(finite) / entry_fill_price - 1.0)


@dataclass
class QuarantineLog:
    """Episodes dropped because they could not be reconstructed honestly."""

    entries: list[dict[str, object]] = field(default_factory=list)

    def add(self, symbol: str, session: dt.date, reason: str) -> None:
        self.entries.append(
            {"symbol": symbol, "session": session.isoformat(), "reason": reason}
        )

    def symbols(self) -> set[str]:
        return {str(entry["symbol"]) for entry in self.entries}

    def __len__(self) -> int:
        return len(self.entries)
