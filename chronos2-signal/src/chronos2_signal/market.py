"""Canonical in-memory market series.

Three representations of the same prices are kept apart on purpose, as the
protocol requires:

1. the immutable provider response (handled by :mod:`chronos2_signal.collector`),
2. the canonical, split-consistent **feature** series built here,
3. the **execution/accounting** series in as-of share and cash units
   (:mod:`chronos2_signal.actions`).

A panel is always aligned to the *expected* bar schedule from the exchange
calendar. A bar the provider did not return becomes a masked row with ``NaN``
prices -- never a synthesised flat price, and never a silently dropped row that
would make an unequal time step look contiguous.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .calendar_spec import BarSpec, ExchangeCalendar

__all__ = [
    "MarketDataError",
    "BarPanel",
    "DailyPanel",
    "OHLCV_COLUMNS",
]


class MarketDataError(RuntimeError):
    """Raised when a series cannot be aligned or fails a structural check."""


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _as_utc_index(values: Iterable) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(list(values), utc=True))
    if index.has_duplicates:
        raise MarketDataError("duplicate timestamps in provider series")
    if not index.is_monotonic_increasing:
        raise MarketDataError("provider series is not sorted by timestamp")
    return index


@dataclass(frozen=True)
class BarPanel:
    """Intraday bars for one symbol, aligned to a scheduled bar index.

    Attributes:
        symbol: Ticker as requested from the provider.
        interval: Provider interval string, e.g. ``"1h"``.
        bars: The expected schedule this panel is aligned to.
        frame: Rows in ``bars`` order, indexed by bar start (UTC), with
            ``open``, ``high``, ``low``, ``close``, ``volume``, ``adj_close``,
            ``observed`` and ``session`` columns.
        snapshot_hash: Identifier of the data vintage behind the rows.
    """

    symbol: str
    interval: str
    bars: tuple[BarSpec, ...]
    frame: pd.DataFrame
    snapshot_hash: str = ""

    # -- construction ------------------------------------------------------ #

    @classmethod
    def aligned(
        cls,
        symbol: str,
        interval: str,
        bars: Sequence[BarSpec],
        observations: Mapping[pd.Timestamp, Mapping[str, float]] | pd.DataFrame | None = None,
        *,
        snapshot_hash: str = "",
    ) -> "BarPanel":
        """Align ``observations`` onto the expected ``bars`` schedule.

        Observations keyed by a timestamp that is not a scheduled bar start are
        rejected rather than snapped to the nearest bar: a differently anchored
        provider series is a quality event, not a rounding problem.
        """
        index = pd.DatetimeIndex([bar.start for bar in bars], name="bar_start")
        frame = pd.DataFrame(
            {
                "open": np.full(len(bars), np.nan),
                "high": np.full(len(bars), np.nan),
                "low": np.full(len(bars), np.nan),
                "close": np.full(len(bars), np.nan),
                "volume": np.full(len(bars), np.nan),
                "adj_close": np.full(len(bars), np.nan),
                "observed": np.zeros(len(bars), dtype=bool),
                "session": [bar.session for bar in bars],
            },
            index=index,
        )
        if observations is not None:
            source = (
                observations
                if isinstance(observations, pd.DataFrame)
                else pd.DataFrame.from_dict(dict(observations), orient="index")
            )
            if len(source):
                source = source.copy()
                source.index = _as_utc_index(source.index)
                unexpected = source.index.difference(index)
                if len(unexpected):
                    raise MarketDataError(
                        f"{symbol} {interval}: {len(unexpected)} observation(s) at "
                        f"unscheduled timestamps, first {unexpected[0].isoformat()}; "
                        "validate the provider's bar anchoring before use"
                    )
                for column in (*OHLCV_COLUMNS, "adj_close"):
                    if column in source.columns:
                        frame.loc[source.index, column] = pd.to_numeric(
                            source[column], errors="coerce"
                        ).to_numpy(dtype=float)
                frame.loc[source.index, "observed"] = True
                # A row whose close is not finite carries no usable price.
                bad = frame.index.isin(source.index) & ~np.isfinite(frame["close"].to_numpy())
                frame.loc[bad, "observed"] = False
        return cls(
            symbol=symbol,
            interval=interval,
            bars=tuple(bars),
            frame=frame,
            snapshot_hash=snapshot_hash,
        )

    @classmethod
    def from_ledger_rows(
        cls,
        symbol: str,
        interval: str,
        bars: Sequence[BarSpec],
        rows: Iterable[Mapping[str, object]],
        *,
        snapshot_hash: str = "",
    ) -> "BarPanel":
        records: dict[pd.Timestamp, dict[str, float]] = {}
        for row in rows:
            if not row.get("observed", True):
                continue
            key = pd.Timestamp(str(row["bar_start"]))
            key = key.tz_convert("UTC") if key.tzinfo else key.tz_localize("UTC")
            records[key] = {
                column: float(row[column])  # type: ignore[arg-type]
                for column in (*OHLCV_COLUMNS, "adj_close")
                if row.get(column) is not None
            }
        return cls.aligned(
            symbol, interval, bars, records, snapshot_hash=snapshot_hash
        )

    # -- access ------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index  # type: ignore[return-value]

    @property
    def observed(self) -> np.ndarray:
        return self.frame["observed"].to_numpy(dtype=bool)

    def column(self, name: str) -> np.ndarray:
        if name not in self.frame.columns:
            raise MarketDataError(f"{self.symbol}: no column {name!r}")
        return self.frame[name].to_numpy(dtype=float)

    @property
    def close(self) -> np.ndarray:
        return self.column("close")

    @property
    def sessions(self) -> list[dt.date]:
        return list(self.frame["session"])

    def observed_fraction(self, last_n: int | None = None) -> float:
        """Share of scheduled bars actually observed, over the whole panel or its tail."""
        mask = self.observed if last_n is None else self.observed[-last_n:]
        if mask.size == 0:
            return 0.0
        return float(mask.mean())

    def session_slice(self, session: dt.date) -> pd.DataFrame:
        return self.frame[self.frame["session"] == session]

    def session_complete(self, session: dt.date) -> bool:
        rows = self.session_slice(session)
        return bool(len(rows)) and bool(rows["observed"].all())

    def last_observed_close(self) -> float:
        closes = self.close
        mask = np.isfinite(closes)
        if not mask.any():
            raise MarketDataError(f"{self.symbol}: no observed close in panel")
        return float(closes[mask][-1])

    def tail(self, count: int) -> "BarPanel":
        if count > len(self.bars):
            raise MarketDataError(
                f"{self.symbol}: requested tail of {count} from a panel of {len(self.bars)}"
            )
        return BarPanel(
            symbol=self.symbol,
            interval=self.interval,
            bars=self.bars[-count:],
            frame=self.frame.iloc[-count:],
            snapshot_hash=self.snapshot_hash,
        )

    def with_prices(self, **columns: np.ndarray) -> "BarPanel":
        """Return a copy with some price columns replaced.

        Used when converting between the feature and execution representations,
        which must never be mutated in place into each other.
        """
        frame = self.frame.copy()
        for name, values in columns.items():
            if len(values) != len(frame):
                raise MarketDataError(
                    f"{self.symbol}: replacement column {name!r} has length "
                    f"{len(values)}, panel has {len(frame)}"
                )
            frame[name] = np.asarray(values, dtype=float)
        return BarPanel(
            symbol=self.symbol,
            interval=self.interval,
            bars=self.bars,
            frame=frame,
            snapshot_hash=self.snapshot_hash,
        )


@dataclass(frozen=True)
class DailyPanel:
    """Daily bars for one symbol, indexed by exchange session date.

    Daily history drives the slow features, the volatility scale, beta and the
    liquidity screens. It is a diagnostic and risk input, not a second
    Transformer input in version 1.
    """

    symbol: str
    frame: pd.DataFrame
    snapshot_hash: str = ""

    @classmethod
    def from_records(
        cls,
        symbol: str,
        records: Mapping[dt.date, Mapping[str, float]] | pd.DataFrame,
        *,
        snapshot_hash: str = "",
    ) -> "DailyPanel":
        frame = (
            records.copy()
            if isinstance(records, pd.DataFrame)
            else pd.DataFrame.from_dict(dict(records), orient="index")
        )
        if len(frame):
            sessions = [
                value.date() if isinstance(value, dt.datetime) else value
                for value in frame.index
            ]
            frame.index = pd.Index(sessions, name="session")
            if frame.index.has_duplicates:
                raise MarketDataError(f"{symbol}: duplicate daily sessions")
            frame = frame.sort_index()
            for column in (*OHLCV_COLUMNS, "adj_close"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame = pd.DataFrame(
                columns=[*OHLCV_COLUMNS, "adj_close"], index=pd.Index([], name="session")
            )
        return cls(symbol=symbol, frame=frame, snapshot_hash=snapshot_hash)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def sessions(self) -> list[dt.date]:
        return list(self.frame.index)

    def up_to(self, session: dt.date) -> "DailyPanel":
        """Rows at or before ``session``.

        Every daily feature goes through this so that an origin can only see
        completed sessions.
        """
        mask = [value <= session for value in self.frame.index]
        return DailyPanel(
            symbol=self.symbol,
            frame=self.frame.loc[mask],
            snapshot_hash=self.snapshot_hash,
        )

    def column(self, name: str) -> np.ndarray:
        if name not in self.frame.columns:
            raise MarketDataError(f"{self.symbol}: no daily column {name!r}")
        return self.frame[name].to_numpy(dtype=float)

    @property
    def close(self) -> np.ndarray:
        return self.column("close")

    def valid_observations(self) -> int:
        if not len(self.frame):
            return 0
        finite = np.isfinite(self.column("close"))
        return int(finite.sum())

    def log_returns(self, *, column: str = "close") -> np.ndarray:
        values = self.column(column)
        if values.size < 2:
            return np.zeros(0, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.diff(np.log(values))

    def row(self, session: dt.date) -> pd.Series:
        if session not in self.frame.index:
            raise MarketDataError(f"{self.symbol}: no daily row for {session.isoformat()}")
        return self.frame.loc[session]


def expected_bar_panel(
    calendar: ExchangeCalendar,
    symbol: str,
    interval: str,
    origin_session: dt.date,
    count: int,
) -> BarPanel:
    """An all-masked panel on the expected schedule ending at ``origin_session``.

    Useful as the starting point for tests and for reporting a symbol whose
    provider data is entirely missing without inventing rows.
    """
    bars = calendar.bars_ending_at(origin_session, count)
    return BarPanel.aligned(symbol, interval, bars, None)
