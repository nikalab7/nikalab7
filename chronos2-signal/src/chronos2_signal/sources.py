"""Origin-bounded, split-consistent market data.

A :class:`MarketSource` does not hand out series. It hands out *views*, and a
view is bounded at one session: every panel it returns ends at or before that
session's close. The pipeline asks for ``source.view(origin)`` and can only see
what that view contains, so "no look-ahead" is a property of the data layer
rather than a convention the feature code has to remember.

Every view also enforces the section-6 adjustment contract before a panel
leaves it:

* each split inside the view's data is **audited** against the raw price and
  volume series;
* a split the provider did not restate is restated here, exactly once;
* a split that cannot be resolved is reported as an unusable audit, which the
  eligibility gate turns into a quarantine instead of a manufactured return.

Two implementations exist. :class:`FixtureMarketSource` wraps the synthetic
market used by the tests. :class:`LedgerMarketSource` reads the SQLite ledger
the collector writes, and adds a second bound -- the data **vintage** -- which
protects old origins from provider revisions made after them. The two are
interchangeable, which is what lets the integrity tests exercise the real data
path.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from .actions import (
    ActionLedger,
    SplitAudit,
    SplitVerdict,
    audit_split_convention,
)
from .calendar_spec import ExchangeCalendar
from .market import PRICE_COLUMNS, BarPanel, DailyPanel
from .provenance import stable_hash
from .storage import Ledger

__all__ = [
    "SourceError",
    "MarketView",
    "MarketSource",
    "FixtureMarketSource",
    "LedgerMarketSource",
    "Vintage",
]

#: How many views a source keeps. A walk-forward run revisits recent sessions
#: repeatedly (marks, labels, entries), so a small cache removes most rebuilds.
_VIEW_CACHE_SIZE = 64


class SourceError(RuntimeError):
    """Raised when a view cannot be constructed."""


class MarketView(Protocol):
    """Everything known about the market after one session's close."""

    session: dt.date

    def hourly_panel(self, symbol: str, count: int) -> BarPanel:  # pragma: no cover
        ...

    def daily_panel(self, symbol: str) -> DailyPanel:  # pragma: no cover
        ...

    def action_ledger(self, symbol: str) -> ActionLedger:  # pragma: no cover
        ...

    def split_audits(self, symbol: str) -> tuple[SplitAudit, ...]:  # pragma: no cover
        ...

    def provenance(self) -> dict[str, Any]:  # pragma: no cover
        ...


class MarketSource(Protocol):
    """A factory of session-bounded views."""

    def view(self, session: dt.date) -> MarketView:  # pragma: no cover
        ...

    def describe(self) -> dict[str, Any]:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# Shared split-consistency logic
# --------------------------------------------------------------------------- #


class _SplitConsistentView:
    """Audits and restates the raw series a subclass provides.

    Subclasses implement ``_raw_daily``, ``_raw_hourly`` and ``_raw_ledger``,
    all bounded at :attr:`session`. Everything public on this class returns
    split-consistent data.
    """

    session: dt.date

    def __init__(self, session: dt.date) -> None:
        self.session = session
        self._daily_cache: dict[str, DailyPanel] = {}
        self._raw_daily_cache: dict[str, DailyPanel] = {}
        self._audit_cache: dict[str, tuple[SplitAudit, ...]] = {}
        self._ledger_cache: dict[str, ActionLedger] = {}

    # -- subclass hooks ---------------------------------------------------- #

    def _raw_daily(self, symbol: str) -> DailyPanel:  # pragma: no cover
        raise NotImplementedError

    def _raw_hourly(self, symbol: str, count: int) -> BarPanel:  # pragma: no cover
        raise NotImplementedError

    def _raw_ledger(self, symbol: str) -> ActionLedger:  # pragma: no cover
        raise NotImplementedError

    # -- public ------------------------------------------------------------ #

    def action_ledger(self, symbol: str) -> ActionLedger:
        if symbol not in self._ledger_cache:
            self._ledger_cache[symbol] = self._raw_ledger(symbol)
        return self._ledger_cache[symbol]

    def split_audits(self, symbol: str) -> tuple[SplitAudit, ...]:
        """Audit every split that falls inside this view's raw daily data.

        A split whose ex-date is at or before the first raw session leaves no
        step inside the data -- every row is already post-split -- so it has
        nothing to audit. A split after this view's session is not in the data
        yet. Everything in between is audited against the *raw* series.
        """
        if symbol in self._audit_cache:
            return self._audit_cache[symbol]
        raw = self._raw_daily_panel(symbol)
        audits: list[SplitAudit] = []
        if len(raw):
            sessions = raw.sessions
            first, last = sessions[0], sessions[-1]
            closes = raw.column("close")
            volumes = raw.column("volume") if "volume" in raw.frame.columns else None
            for split in self.action_ledger(symbol).splits():
                if not (first < split.ex_date <= min(last, self.session)):
                    continue
                audits.append(
                    audit_split_convention(split, sessions, closes, volumes)
                )
        result = tuple(audits)
        self._audit_cache[symbol] = result
        return result

    def daily_panel(self, symbol: str) -> DailyPanel:
        """Split-consistent daily panel, bounded at this view's session."""
        if symbol in self._daily_cache:
            return self._daily_cache[symbol]
        raw = self._raw_daily_panel(symbol)
        audits = self._restatable(symbol)
        if not audits or not len(raw):
            panel = raw
        else:
            frame = raw.frame.copy()
            sessions = list(frame.index)
            for column in PRICE_COLUMNS:
                if column in frame.columns:
                    frame[column] = _restate(sessions, frame[column].to_numpy(dtype=float), audits, "price")
            if "volume" in frame.columns:
                frame["volume"] = _restate(sessions, frame["volume"].to_numpy(dtype=float), audits, "quantity")
            panel = DailyPanel(
                symbol=raw.symbol,
                frame=frame,
                snapshot_hash=_restated_hash(raw.snapshot_hash, audits),
            )
        self._daily_cache[symbol] = panel
        return panel

    def hourly_panel(self, symbol: str, count: int) -> BarPanel:
        """Split-consistent hourly panel of ``count`` bars ending at the session."""
        raw = self._raw_hourly(symbol, count)
        if raw.bars and raw.bars[-1].session != self.session:
            raise SourceError(
                f"{symbol}: hourly panel ends on {raw.bars[-1].session.isoformat()}, "
                f"not on the view's session {self.session.isoformat()}"
            )
        audits = self._restatable(symbol)
        if not audits:
            return raw
        sessions = [bar.session for bar in raw.bars]
        replacements = {
            column: _restate(sessions, raw.column(column), audits, "price")
            for column in PRICE_COLUMNS
        }
        replacements["volume"] = _restate(sessions, raw.column("volume"), audits, "quantity")
        restated = raw.with_prices(**replacements)
        return BarPanel(
            symbol=restated.symbol,
            interval=restated.interval,
            bars=restated.bars,
            frame=restated.frame,
            snapshot_hash=_restated_hash(raw.snapshot_hash, audits),
        )

    # -- internals --------------------------------------------------------- #

    def _raw_daily_panel(self, symbol: str) -> DailyPanel:
        if symbol not in self._raw_daily_cache:
            self._raw_daily_cache[symbol] = self._raw_daily(symbol)
        return self._raw_daily_cache[symbol]

    def _restatable(self, symbol: str) -> list[SplitAudit]:
        """Usable audits that call for a restatement.

        An unusable audit is *not* restated: there is no honest factor to apply.
        It is reported through :meth:`split_audits` so that eligibility can
        quarantine the affected origins.
        """
        return [
            audit
            for audit in self.split_audits(symbol)
            if audit.verdict is SplitVerdict.NOT_APPLIED
        ]


def _restate(
    sessions: Sequence[dt.date],
    values: np.ndarray,
    audits: Sequence[SplitAudit],
    kind: str,
) -> np.ndarray:
    """Restate rows before each ex-date to the post-split units, once."""
    result = np.asarray(values, dtype=float).copy()
    for audit in audits:
        ratio = audit.action.value
        before = np.asarray([session < audit.action.ex_date for session in sessions], dtype=bool)
        if kind == "price":
            result[before] = result[before] / ratio
        else:
            result[before] = result[before] * ratio
    return result


def _restated_hash(raw_hash: str, audits: Sequence[SplitAudit]) -> str:
    """Identity of a restated panel: the raw vintage plus what was applied."""
    return stable_hash(
        {
            "raw": raw_hash,
            "restated": [
                [audit.action.ex_date.isoformat(), audit.action.value, audit.verdict.value]
                for audit in audits
            ],
        }
    )[:32]


def _cache_view(cache: dict[dt.date, Any], session: dt.date, view: Any) -> Any:
    if len(cache) >= _VIEW_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    cache[session] = view
    return view


# --------------------------------------------------------------------------- #
# Fixture source
# --------------------------------------------------------------------------- #


class _FixtureView(_SplitConsistentView):
    def __init__(self, market: Any, session: dt.date) -> None:
        super().__init__(session)
        self._market = market

    def _raw_daily(self, symbol: str) -> DailyPanel:
        return self._market.daily(symbol).up_to(self.session)

    def _raw_hourly(self, symbol: str, count: int) -> BarPanel:
        return self._market.panel_ending_at(symbol, self.session, count)

    def _raw_ledger(self, symbol: str) -> ActionLedger:
        return self._market.action_ledger(symbol)

    def provenance(self) -> dict[str, Any]:
        return {
            "source": "fixture",
            "session": self.session.isoformat(),
            "spec_hash": self._market.spec_hash,
            "vintage": "deterministic",
        }


@dataclass
class FixtureMarketSource:
    """Views over a :class:`~chronos2_signal.fixtures.SyntheticMarket`.

    The fixture has no vintages -- it is deterministic -- so a view is bounded
    by session only.
    """

    market: Any
    _views: dict[dt.date, _FixtureView] = field(default_factory=dict, init=False, repr=False)

    def view(self, session: dt.date) -> MarketView:
        cached = self._views.get(session)
        if cached is not None:
            return cached
        return _cache_view(self._views, session, _FixtureView(self.market, session))

    def invalidate(self) -> None:
        """Drop cached views after the underlying fixture was modified."""
        self._views.clear()

    def describe(self) -> dict[str, Any]:
        return {
            "source": "fixture",
            "spec_hash": self.market.spec_hash,
            "vintage": "deterministic",
        }


# --------------------------------------------------------------------------- #
# Ledger source
# --------------------------------------------------------------------------- #


class Vintage:
    """How a ledger view chooses between snapshots of the same bar."""

    #: Only snapshots retrieved by the origin's data deadline. For data that was
    #: collected forward, day by day: this is what the decision actually saw.
    POINT_IN_TIME = "point_in_time"
    #: The most recent snapshot of every bar. For a historical backfill fetched
    #: in one go, where every row was retrieved today and a point-in-time read
    #: would find nothing. Provider revisions after each origin are then not
    #: excluded, and every report produced this way says so.
    LATEST = "latest"

    ALL = (POINT_IN_TIME, LATEST)

    LATEST_LIMITATION = (
        "historical backfill read at the latest vendor vintage: provider revisions "
        "made after each origin are not excluded, so this does not reproduce what "
        "a decision on that date would have seen"
    )


class _LedgerView(_SplitConsistentView):
    def __init__(
        self,
        *,
        ledger: Ledger,
        calendar: ExchangeCalendar,
        session: dt.date,
        as_of: dt.datetime | None,
        vintage: str,
        hourly_interval: str,
        daily_interval: str,
    ) -> None:
        super().__init__(session)
        self._ledger = ledger
        self._calendar = calendar
        self._as_of = as_of
        self._vintage = vintage
        self._hourly_interval = hourly_interval
        self._daily_interval = daily_interval

    def _raw_daily(self, symbol: str) -> DailyPanel:
        rows = self._ledger.read_bars(
            symbol, self._daily_interval, end=self.session, as_of=self._as_of
        )
        records: dict[dt.date, dict[str, float]] = {}
        for row in rows:
            if not row.get("observed"):
                continue
            session = dt.date.fromisoformat(str(row["session"])[:10])
            records[session] = {
                column: float(row[column])
                for column in ("open", "high", "low", "close", "volume", "adj_close")
                if row.get(column) is not None
            }
        return DailyPanel.from_records(
            symbol, records, snapshot_hash=_snapshot_identity(rows, self._vintage)
        )

    def _raw_hourly(self, symbol: str, count: int) -> BarPanel:
        bars = self._calendar.bars_ending_at(self.session, count)
        rows = self._ledger.read_bars(
            symbol,
            self._hourly_interval,
            start=bars[0].session,
            end=self.session,
            as_of=self._as_of,
        )
        # The first session of the window is usually entered mid-session; its
        # earlier bars are outside the requested schedule and must not be
        # mistaken for unscheduled provider bars.
        wanted = {bar.start for bar in bars}
        in_window = [
            row
            for row in rows
            if pd.Timestamp(str(row["bar_start"])).tz_convert("UTC") in wanted
        ]
        return BarPanel.from_ledger_rows(
            symbol,
            self._hourly_interval,
            bars,
            in_window,
            snapshot_hash=_snapshot_identity(in_window, self._vintage),
        )

    def _raw_ledger(self, symbol: str) -> ActionLedger:
        return ActionLedger.from_rows(
            symbol, self._ledger.corporate_actions(symbol, as_of=self._as_of)
        )

    def provenance(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "source": "ledger",
            "session": self.session.isoformat(),
            "vintage": self._vintage,
            "as_of": self._as_of.isoformat() if self._as_of else None,
        }
        if self._vintage == Vintage.LATEST:
            record["limitation"] = Vintage.LATEST_LIMITATION
        return record


def _snapshot_identity(rows: Sequence[dict[str, Any]], vintage: str) -> str:
    """Identity of the exact rows a panel was built from."""
    return stable_hash(
        {"vintage": vintage, "snapshots": sorted({str(row["snapshot_id"]) for row in rows})}
    )[:32]


@dataclass
class LedgerMarketSource:
    """Views over the bars and actions the collector wrote to the ledger.

    Args:
        vintage: :attr:`Vintage.POINT_IN_TIME` for forward-collected data, or
            :attr:`Vintage.LATEST` for a one-shot historical backfill. There is
            no default: which one applies is a fact about how the data was
            collected, and choosing wrongly either finds no data or hides a
            leak.
        max_data_delay_minutes: The registered snapshot deadline after the close.
            A point-in-time view accepts snapshots retrieved up to that deadline,
            because a snapshot retrieved later could not have fed that origin.
    """

    ledger: Ledger
    calendar: ExchangeCalendar
    vintage: str
    max_data_delay_minutes: int
    hourly_interval: str = "1h"
    daily_interval: str = "1d"
    _views: dict[dt.date, _LedgerView] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.vintage not in Vintage.ALL:
            raise SourceError(
                f"unknown vintage {self.vintage!r}; expected one of {Vintage.ALL}"
            )

    def as_of(self, session: dt.date) -> dt.datetime | None:
        """The retrieval cut-off a view at ``session`` reads under."""
        if self.vintage == Vintage.LATEST:
            return None
        deadline = self.calendar.data_deadline(
            session, max_delay_minutes=self.max_data_delay_minutes
        )
        return deadline.to_pydatetime()

    def view(self, session: dt.date) -> MarketView:
        cached = self._views.get(session)
        if cached is not None:
            return cached
        view = _LedgerView(
            ledger=self.ledger,
            calendar=self.calendar,
            session=session,
            as_of=self.as_of(session),
            vintage=self.vintage,
            hourly_interval=self.hourly_interval,
            daily_interval=self.daily_interval,
        )
        return _cache_view(self._views, session, view)

    def describe(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "source": "ledger",
            "path": str(self.ledger.path),
            "vintage": self.vintage,
            "max_data_delay_minutes": self.max_data_delay_minutes,
        }
        if self.vintage == Vintage.LATEST:
            record["limitation"] = Vintage.LATEST_LIMITATION
        return record
