"""The reference paper portfolio.

These are evaluation allocations, not a recommendation for anyone's account.
The rules are the registered ones and nothing else:

* A new position targets 10% of current equity, bounded by remaining cash and
  by a 30% gross-exposure limit on **new** allocations. Existing
  marked-to-market exposure may drift above the limit, in which case no new
  position is added -- there is no untested rebalancing exit to enforce a
  continuous cap.
* Entry is the next session's official open, exit the second following
  session's official close. No leverage, no shorting, no pyramiding.
* Idle cash earns zero, and every baseline is treated identically.
* A drawdown pause is absorbing inside an evaluation run: it is not reset at a
  fold or report boundary, and existing positions keep their registered time
  exits rather than closing at an imaginary protected price.

Prices are split-consistent. A position's economic size is therefore unchanged
by a split, and the nominal share count is tracked alongside purely so the
ledger reconciles against a brokerage view. Every price supplied for one
position must be in the units it was entered in; when the data's units change
mid-hold, the session engine re-expresses the position first
(:meth:`ReferencePortfolio.restate_units`).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .actions import ActionLedger, HoldingCosts, label_net_return
from .config import DesignConfig

__all__ = [
    "PortfolioError",
    "PaperPosition",
    "PortfolioDay",
    "EntryRequest",
    "ReferencePortfolio",
]


class PortfolioError(RuntimeError):
    """Raised on an accounting rule violation."""


@dataclass
class PaperPosition:
    """One open or closed reference position.

    ``restated_shares`` is the quantity in split-consistent units and is
    constant across the hold. ``nominal_shares`` is what a statement would
    show, and changes by exactly the split ratio.

    ``unit_restatement`` is the factor by which the position has been
    re-expressed because later quotes arrived in different units than its
    entry (see :meth:`ReferencePortfolio.restate_units`). It is 1.0 whenever
    the data kept one set of units across the hold.
    """

    position_id: str
    signal_id: str
    symbol: str
    issuer_id: str
    sector: str
    signal_session: dt.date
    entry_session: dt.date
    planned_exit_session: dt.date
    restated_shares: float
    nominal_shares: float
    entry_reference_price: float
    entry_fill_price: float
    cost_basis: float
    cost_scenario: str
    exit_session: dt.date | None = None
    exit_reference_price: float | None = None
    exit_fill_price: float | None = None
    proceeds: float | None = None
    cash_dividends: float = 0.0
    share_adjustment: float = 1.0
    unit_restatement: float = 1.0
    max_adverse_excursion: float | None = None
    status: str = "open"

    def market_value(self, price: float) -> float:
        """Split-consistent market value at ``price``."""
        if not math.isfinite(price) or price <= 0.0:
            raise PortfolioError(
                f"{self.symbol}: cannot mark a position at price {price!r}"
            )
        return self.restated_shares * price

    def realised_net_return(self) -> float:
        """Net return actually realised, from the ledger's own numbers."""
        if self.proceeds is None:
            raise PortfolioError(f"{self.symbol}: position is still open")
        return (self.proceeds + self.cash_dividends - self.cost_basis) / self.cost_basis


@dataclass(frozen=True)
class PortfolioDay:
    """One marked session of the reference portfolio."""

    session: dt.date
    equity: float
    cash: float
    gross_exposure: float
    open_positions: int
    net_return: float
    peak_equity: float
    drawdown: float
    paused: bool


@dataclass(frozen=True)
class EntryRequest:
    """An alert promoted to a reference entry at the next open."""

    signal_id: str
    symbol: str
    issuer_id: str
    sector: str
    signal_session: dt.date
    entry_session: dt.date
    planned_exit_session: dt.date
    reference_open_price: float
    action_ledger: ActionLedger | None = None


@dataclass
class ReferencePortfolio:
    """Cash, positions and daily marks for one variant.

    Args:
        config: Registered configuration.
        variant: Which candidate or baseline this portfolio belongs to. Every
            baseline runs through this same class so that fills, costs and
            capital accounting are identical by construction.
        cost_scenario: ``"base"``, ``"stress"`` or ``"severe"``.
    """

    config: DesignConfig
    variant: str
    cost_scenario: str = "base"
    cash: float = field(init=False)
    positions: list[PaperPosition] = field(default_factory=list, init=False)
    closed: list[PaperPosition] = field(default_factory=list, init=False)
    days: list[PortfolioDay] = field(default_factory=list, init=False)
    peak_equity: float = field(init=False)
    paused: bool = field(default=False, init=False)
    pause_session: dt.date | None = field(default=None, init=False)
    _last_equity: float = field(init=False)
    _counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cash = self.config.portfolio.initial_equity
        self.peak_equity = self.config.portfolio.initial_equity
        self._last_equity = self.config.portfolio.initial_equity

    # -- costs ------------------------------------------------------------- #

    @property
    def costs(self) -> HoldingCosts:
        slippage = self.config.execution.slippage_fraction(self.cost_scenario)
        return HoldingCosts(
            buy_slippage=slippage,
            sell_slippage=slippage,
            explicit_fee_fraction=self.config.execution.explicit_fee_fraction,
        )

    # -- state ------------------------------------------------------------- #

    def open_symbols(self) -> set[str]:
        return {position.symbol for position in self.positions}

    def open_issuers(self) -> set[str]:
        return {position.issuer_id for position in self.positions}

    def sector_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for position in self.positions:
            counts[position.sector] = counts.get(position.sector, 0) + 1
        return counts

    def gross_value(self, prices: Mapping[str, float]) -> float:
        total = 0.0
        for position in self.positions:
            price = prices.get(position.symbol)
            if price is None:
                raise PortfolioError(
                    f"{position.symbol}: no price supplied to mark an open position"
                )
            total += position.market_value(price)
        return total

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.gross_value(prices)

    # -- entries ----------------------------------------------------------- #

    def allocate(
        self,
        request: EntryRequest,
        *,
        equity_at_open: float,
        gross_value_at_open: float,
    ) -> PaperPosition | None:
        """Size and open one position at the reference opening fill.

        Returns ``None`` when the registered limits leave no room. That is a
        capacity outcome, not an error, and it is recorded rather than worked
        around.
        """
        settings = self.config.portfolio
        if self.paused:
            return None
        if len(self.positions) >= settings.max_open_positions:
            return None
        if request.issuer_id in self.open_issuers():
            raise PortfolioError(
                f"{request.symbol}: issuer {request.issuer_id} already has an open "
                "position; the policy engine should have suppressed this alert"
            )
        if self.sector_counts().get(request.sector, 0) >= settings.max_open_per_sector:
            return None

        price = request.reference_open_price
        if not math.isfinite(price) or price <= 0.0:
            raise PortfolioError(f"{request.symbol}: invalid reference open {price!r}")

        target = settings.position_fraction_current_equity_at_entry * equity_at_open
        exposure_room = settings.max_gross_exposure * equity_at_open - gross_value_at_open
        notional = min(target, self.cash, max(0.0, exposure_room))
        if notional <= 0.0:
            return None

        fill_price = price * (1.0 + self.costs.buy_slippage)
        restated_shares = notional / fill_price
        if restated_shares <= 0.0:
            return None

        ledger = request.action_ledger or ActionLedger(symbol=request.symbol)
        nominal = restated_shares / ledger.split_factor_after(request.entry_session)

        self._counter += 1
        position = PaperPosition(
            position_id=f"{self.variant}-{request.entry_session.isoformat()}-{self._counter:04d}",
            signal_id=request.signal_id,
            symbol=request.symbol,
            issuer_id=request.issuer_id,
            sector=request.sector,
            signal_session=request.signal_session,
            entry_session=request.entry_session,
            planned_exit_session=request.planned_exit_session,
            restated_shares=restated_shares,
            nominal_shares=nominal,
            entry_reference_price=price,
            entry_fill_price=fill_price,
            cost_basis=restated_shares * fill_price,
            cost_scenario=self.cost_scenario,
        )
        self.cash -= position.cost_basis
        if self.cash < -1e-9:
            raise PortfolioError(
                f"{request.symbol}: allocation drove cash negative; the reference "
                "portfolio does not borrow"
            )
        self.cash = max(0.0, self.cash)
        self.positions.append(position)
        return position

    def enter_all(
        self,
        requests: Sequence[EntryRequest],
        *,
        equity_at_open: float,
        gross_value_at_open: float,
    ) -> list[PaperPosition]:
        """Open as many requests as the limits allow, in the given order.

        Equity and existing exposure are evaluated once, at the open, so that
        two entries in the same session cannot each claim the whole allowance.
        """
        opened: list[PaperPosition] = []
        running_gross = gross_value_at_open
        for request in requests:
            position = self.allocate(
                request,
                equity_at_open=equity_at_open,
                gross_value_at_open=running_gross,
            )
            if position is not None:
                opened.append(position)
                running_gross += position.cost_basis
        return opened

    # -- during the hold --------------------------------------------------- #

    def credit_dividends(
        self, session: dt.date, ledgers: Mapping[str, ActionLedger]
    ) -> float:
        """Credit cash dividends whose ex-date is ``session``.

        Entitlement requires having held through the ex-date boundary: a
        position entered on the ex-date itself earns nothing.
        """
        credited = 0.0
        for position in self.positions:
            ledger = ledgers.get(position.symbol)
            if ledger is None:
                continue
            for dividend in ledger.dividends():
                if dividend.ex_date != session:
                    continue
                if dividend.ex_date <= position.entry_session:
                    continue
                record_factor = ledger.split_factor_after(
                    dividend.ex_date - dt.timedelta(days=1)
                )
                shares_on_record = position.restated_shares / record_factor
                amount = shares_on_record * dividend.value
                position.cash_dividends += amount
                self.cash += amount
                credited += amount
        return credited

    def apply_splits(
        self, session: dt.date, ledgers: Mapping[str, ActionLedger]
    ) -> None:
        """Restate nominal share counts for splits with an ex-date of ``session``.

        The split-consistent quantity and the market value are untouched: a
        split creates no wealth, and this is where that is enforced rather than
        assumed.
        """
        for position in self.positions:
            ledger = ledgers.get(position.symbol)
            if ledger is None:
                continue
            for split in ledger.splits():
                if split.ex_date != session or split.ex_date <= position.entry_session:
                    continue
                position.nominal_shares *= split.value
                position.share_adjustment *= split.value

    def restate_units(self, position: PaperPosition, factor: float) -> None:
        """Re-express an open position in price units ``factor`` times smaller.

        Needed when later quotes arrive in different units than the entry: a
        point-in-time view after an ex-date quotes post-split prices, while the
        entry was filled from pre-split ones. The quantity grows by exactly the
        factor and every recorded price shrinks by it, so the cost basis, the
        market value and the realised return are unchanged -- a change of units
        creates no wealth. The nominal count is not touched here; the split
        itself is booked by :meth:`apply_splits`.
        """
        if not math.isfinite(factor) or factor <= 0.0:
            raise PortfolioError(f"{position.symbol}: invalid unit factor {factor!r}")
        if position not in self.positions:
            raise PortfolioError(f"{position.symbol}: only an open position can be restated")
        position.restated_shares *= factor
        position.entry_reference_price /= factor
        position.entry_fill_price /= factor
        position.unit_restatement *= factor

    def record_excursion(self, lows: Mapping[str, float]) -> None:
        """Track the worst drawdown from entry, for diagnosis only.

        Never used as an exit price: version 1 has no stop, and crediting the
        strategy with an exit it did not take would be fiction.
        """
        for position in self.positions:
            low = lows.get(position.symbol)
            if low is None or not math.isfinite(low):
                continue
            excursion = min(0.0, low / position.entry_fill_price - 1.0)
            if (
                position.max_adverse_excursion is None
                or excursion < position.max_adverse_excursion
            ):
                position.max_adverse_excursion = excursion

    # -- exits ------------------------------------------------------------- #

    def close_due(
        self,
        session: dt.date,
        prices: Mapping[str, float],
        ledgers: Mapping[str, ActionLedger] | None = None,
    ) -> list[PaperPosition]:
        """Close every position whose registered exit session is ``session``."""
        ledgers = ledgers or {}
        closing = [
            position
            for position in self.positions
            if position.planned_exit_session == session
        ]
        for position in closing:
            price = prices.get(position.symbol)
            if price is None or not math.isfinite(price) or price <= 0.0:
                # A missing exit price stays in the ledger as an unresolved
                # exit. A failed trade is never dropped because its data
                # disappeared.
                position.status = "exit_price_unavailable"
                position.exit_session = session
                continue
            fill = price * (1.0 - self.costs.sell_slippage)
            position.exit_session = session
            position.exit_reference_price = price
            position.exit_fill_price = fill
            # The explicit fee is a fraction of the entry notional, charged once
            # per round trip -- exactly as the registered label charges it.
            position.proceeds = (
                position.restated_shares * fill
                - self.costs.explicit_fee_fraction * position.cost_basis
            )
            position.status = "closed"
            ledger = ledgers.get(position.symbol)
            if ledger is not None:
                position.share_adjustment = ledger.split_factor_in(
                    position.entry_session, session
                )
                position.nominal_shares = (
                    position.restated_shares / ledger.split_factor_after(session)
                )
            self.cash += position.proceeds
        self.positions = [
            position for position in self.positions if position not in closing
        ]
        self.closed.extend(closing)
        return closing

    def accounting_for(
        self, position: PaperPosition, ledger: ActionLedger | None = None
    ) -> object:
        """Re-derive the holding account for a closed position.

        Used to cross-check the ledger against :func:`label_net_return`: the
        two are computed by different code paths and must agree.
        """
        if position.exit_reference_price is None or position.exit_session is None:
            raise PortfolioError(f"{position.symbol}: position has no recorded exit")
        return label_net_return(
            entry_session=position.entry_session,
            exit_session=position.exit_session,
            entry_open_price=position.entry_reference_price,
            exit_close_price=position.exit_reference_price,
            costs=self.costs,
            ledger=ledger,
            shares=position.restated_shares,
        )

    # -- daily marks ------------------------------------------------------- #

    def mark(self, session: dt.date, prices: Mapping[str, float]) -> PortfolioDay:
        """Record one marked session and update the drawdown state.

        Overlapping holds are included, and cash exposure is reported, because
        a trade-level average return and a portfolio return are different
        outputs.
        """
        gross = self.gross_value(prices)
        equity = self.cash + gross
        if equity <= 0.0:
            raise PortfolioError(f"{session.isoformat()}: equity is non-positive")
        net_return = equity / self._last_equity - 1.0 if self._last_equity > 0 else 0.0
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1.0

        limit = self.config.portfolio.pause_new_alerts_at_portfolio_drawdown
        if not self.paused and drawdown <= -limit:
            self.paused = True
            self.pause_session = session

        day = PortfolioDay(
            session=session,
            equity=equity,
            cash=self.cash,
            gross_exposure=gross / equity,
            open_positions=len(self.positions),
            net_return=net_return,
            peak_equity=self.peak_equity,
            drawdown=drawdown,
            paused=self.paused,
        )
        self.days.append(day)
        self._last_equity = equity
        return day

    # -- reporting --------------------------------------------------------- #

    def daily_returns(self) -> list[float]:
        return [day.net_return for day in self.days]

    def sessions(self) -> list[dt.date]:
        return [day.session for day in self.days]

    def summary(self) -> dict[str, float | int | bool | None]:
        equity = self.days[-1].equity if self.days else self.config.portfolio.initial_equity
        exposures = [day.gross_exposure for day in self.days]
        drawdowns = [day.drawdown for day in self.days]
        resolved = [p for p in self.closed if p.status == "closed"]
        unresolved = [p for p in self.closed if p.status != "closed"]
        return {
            "variant": self.variant,
            "cost_scenario": self.cost_scenario,
            "sessions_marked": len(self.days),
            "final_equity": equity,
            "total_return": equity / self.config.portfolio.initial_equity - 1.0,
            "max_drawdown": min(drawdowns) if drawdowns else 0.0,
            "mean_gross_exposure": (sum(exposures) / len(exposures)) if exposures else 0.0,
            "closed_positions": len(resolved),
            "unresolved_exits": len(unresolved),
            "open_positions": len(self.positions),
            "paused": self.paused,
            "pause_session": self.pause_session.isoformat() if self.pause_session else None,
        }
