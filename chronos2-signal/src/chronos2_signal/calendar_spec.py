"""Exchange session and bar arithmetic.

Everything downstream of this module counts *scheduled exchange bars*, never
wall-clock hours and never calendar days. Two trading sessions are not 13 hours
and not 48 hours: on a normal pair of sessions they are 14 hourly bars, and
around a half day they are fewer.

Conventions fixed here, matching the protocol document:

* Internal timestamps are timezone-aware UTC. Session logic derives from the
  exchange calendar in ``America/New_York``; display formatting happens at the
  edges of the system only.
* A bar is keyed by its **start**. A normal session yields seven bars starting
  09:30, 10:30, 11:30, 12:30, 13:30, 14:30 and 15:30, the last of which lasts
  30 minutes and ends at 16:00.
* Real session gaps are preserved. There are no synthetic overnight or weekend
  bars; unequal elapsed time is exposed to the model through a calendar
  channel instead.
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CalendarError",
    "BarSpec",
    "Horizon",
    "ExchangeCalendar",
    "calendar_channel_matrix",
    "CALENDAR_CHANNEL_NAMES",
    "REGULAR_SESSION_MINUTES",
    "NOMINAL_BAR_MINUTES",
]

UTC = dt.timezone.utc

#: Length of a full regular session in minutes (09:30-16:00 New York).
REGULAR_SESSION_MINUTES = 390.0

#: Nominal bar length in minutes. The final bar of a session is shorter.
NOMINAL_BAR_MINUTES = 60.0

#: Names of the five known-future calendar channels, in channel order.
CALENDAR_CHANNEL_NAMES = (
    "cal_minutes_since_open_frac",
    "cal_bar_duration_frac",
    "cal_gap_hours",
    "cal_weekday_sin",
    "cal_weekday_cos",
)

# Padding applied when the internal session index has to be extended. Chosen so
# that ordinary walk-forward use extends the index a handful of times, not once
# per origin.
_INDEX_PAD = dt.timedelta(days=540)


class CalendarError(RuntimeError):
    """Raised when a session or bar request cannot be answered exactly."""


@dataclass(frozen=True, slots=True)
class BarSpec:
    """One scheduled regular-session bar.

    Attributes:
        session: Exchange session date the bar belongs to.
        start: Bar start, timezone-aware UTC. This is the bar's key.
        end: Bar end, timezone-aware UTC.
        slot: Zero-based position of the bar inside its session.
        gap_hours: Hours between the end of the previous scheduled bar and this
            bar's start. ``0.0`` between contiguous bars inside a session,
            larger across an overnight or weekend break, and ``nan`` when the
            predecessor falls outside the requested window.
    """

    session: dt.date
    start: pd.Timestamp
    end: pd.Timestamp
    slot: int
    gap_hours: float

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0

    @property
    def minutes_since_open(self) -> float:
        """Minutes from the session open to this bar's start.

        Bars step from the open in fixed nominal increments, so the slot index
        determines this exactly; only the final bar's *duration* is shortened.
        """
        return self.slot * NOMINAL_BAR_MINUTES

    def calendar_channels(self) -> tuple[float, float, float, float, float]:
        """The five known-future calendar values for this bar.

        The first channel divides by the fixed full-session length, so a half
        day is visibly compressed rather than rescaled to look normal.
        """
        weekday_angle = 2.0 * math.pi * self.session.weekday() / 7.0
        return (
            self.minutes_since_open / REGULAR_SESSION_MINUTES,
            self.duration_minutes / NOMINAL_BAR_MINUTES,
            self.gap_hours,
            math.sin(weekday_angle),
            math.cos(weekday_angle),
        )


@dataclass(frozen=True, slots=True)
class Horizon:
    """The future bars of a fixed multi-session holding period.

    Attributes:
        origin_session: The completed signal session, D0.
        sessions: Future sessions covered, in order (D1, D2, ...).
        bars: Every future bar, in order.
        terminal_indices: Index into ``bars`` of the last bar of each session.
    """

    origin_session: dt.date
    sessions: tuple[dt.date, ...]
    bars: tuple[BarSpec, ...]
    terminal_indices: tuple[int, ...]

    @property
    def prediction_length(self) -> int:
        return len(self.bars)

    def terminal_index(self, session_offset: int) -> int:
        """Index of the terminal bar of ``D{session_offset}`` (1-based offset)."""
        if not 1 <= session_offset <= len(self.terminal_indices):
            raise CalendarError(
                f"session offset {session_offset} outside horizon of "
                f"{len(self.terminal_indices)} sessions"
            )
        return self.terminal_indices[session_offset - 1]

    def session_of(self, session_offset: int) -> dt.date:
        if not 1 <= session_offset <= len(self.sessions):
            raise CalendarError(
                f"session offset {session_offset} outside horizon of "
                f"{len(self.sessions)} sessions"
            )
        return self.sessions[session_offset - 1]


class ExchangeCalendar:
    """Session and bar schedule for one exchange.

    The calendar package version is pinned by the dependency lock and recorded
    in the release manifest: holiday and early-close tables change over time, so
    a result is only reproducible together with the calendar that produced it.

    Sessions are resolved through one internally cached, lazily widened index so
    that a walk-forward backtest does not rebuild a schedule per origin.
    """

    def __init__(
        self,
        name: str = "XNYS",
        *,
        exchange_timezone: str = "America/New_York",
    ) -> None:
        import pandas_market_calendars as mcal

        self.name = name
        self.exchange_timezone = exchange_timezone
        self._calendar = mcal.get_calendar(name)
        self._package_version = getattr(mcal, "__version__", "unknown")
        self._sessions: list[dt.date] = []
        self._windows: dict[dt.date, tuple[pd.Timestamp, pd.Timestamp]] = {}
        self._covered: tuple[dt.date, dt.date] | None = None
        # Bar starts are a pure function of the session window, and a
        # walk-forward run asks for the same sessions thousands of times.
        self._bar_starts: dict[dt.date, list[pd.Timestamp]] = {}

    # -- provenance -------------------------------------------------------- #

    @property
    def package_version(self) -> str:
        return self._package_version

    def provenance(self) -> dict[str, str]:
        return {
            "calendar_name": self.name,
            "calendar_package_version": self._package_version,
            "exchange_timezone": self.exchange_timezone,
        }

    # -- internal session index -------------------------------------------- #

    def _ensure_covered(self, lo: dt.date, hi: dt.date) -> None:
        if self._covered is not None:
            covered_lo, covered_hi = self._covered
            if covered_lo <= lo and hi <= covered_hi:
                return
            lo = min(lo, covered_lo)
            hi = max(hi, covered_hi)
        want_lo, want_hi = lo - _INDEX_PAD, hi + _INDEX_PAD
        schedule = self._calendar.schedule(start_date=want_lo, end_date=want_hi)
        if "market_open" not in schedule.columns or "market_close" not in schedule.columns:
            raise CalendarError("calendar schedule is missing market_open/market_close")
        sessions: list[dt.date] = []
        windows: dict[dt.date, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for index_ts, row in schedule.iterrows():
            session = index_ts.date()
            open_ts = pd.Timestamp(row["market_open"]).tz_convert("UTC")
            close_ts = pd.Timestamp(row["market_close"]).tz_convert("UTC")
            if close_ts <= open_ts:
                raise CalendarError(f"degenerate session window on {session.isoformat()}")
            sessions.append(session)
            windows[session] = (open_ts, close_ts)
        self._sessions = sessions
        self._windows = windows
        self._covered = (want_lo, want_hi)
        # Rebuilt windows invalidate any derived bar schedule.
        self._bar_starts.clear()

    def _position(self, session: dt.date) -> int:
        """Index of ``session`` in the internal session list."""
        self._ensure_covered(session, session)
        pos = bisect.bisect_left(self._sessions, session)
        if pos >= len(self._sessions) or self._sessions[pos] != session:
            raise CalendarError(f"{session.isoformat()} is not a trading session")
        return pos

    # -- sessions ---------------------------------------------------------- #

    def sessions(self, start: dt.date, end: dt.date) -> list[dt.date]:
        """Trading sessions in ``[start, end]``, ascending."""
        if end < start:
            return []
        self._ensure_covered(start, end)
        lo = bisect.bisect_left(self._sessions, start)
        hi = bisect.bisect_right(self._sessions, end)
        return self._sessions[lo:hi]

    def is_session(self, day: dt.date) -> bool:
        self._ensure_covered(day, day)
        pos = bisect.bisect_left(self._sessions, day)
        return pos < len(self._sessions) and self._sessions[pos] == day

    def session_window(self, session: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
        """UTC open and close timestamps of one session."""
        self._position(session)
        return self._windows[session]

    def session_open(self, session: dt.date) -> pd.Timestamp:
        return self.session_window(session)[0]

    def session_close(self, session: dt.date) -> pd.Timestamp:
        return self.session_window(session)[1]

    def session_minutes(self, session: dt.date) -> float:
        open_ts, close_ts = self.session_window(session)
        return (close_ts - open_ts).total_seconds() / 60.0

    def is_early_close(self, session: dt.date) -> bool:
        return self.session_minutes(session) < REGULAR_SESSION_MINUTES - 1e-9

    def next_session(self, after: dt.date, count: int = 1) -> dt.date:
        """The ``count``-th trading session strictly after ``after``.

        ``after`` itself need not be a session, which makes this usable for
        "the session that follows this calendar date".
        """
        if count < 1:
            raise CalendarError("count must be at least 1")
        self._ensure_covered(after, after + dt.timedelta(days=30 * count + 30))
        pos = bisect.bisect_right(self._sessions, after) + count - 1
        if pos >= len(self._sessions):
            raise CalendarError(
                f"session index does not reach {count} sessions after {after.isoformat()}"
            )
        return self._sessions[pos]

    def previous_sessions(self, before: dt.date, count: int) -> list[dt.date]:
        """The ``count`` trading sessions strictly before ``before``, ascending."""
        if count < 0:
            raise CalendarError("count must be non-negative")
        if count == 0:
            return []
        self._ensure_covered(before - dt.timedelta(days=30 * count + 30), before)
        hi = bisect.bisect_left(self._sessions, before)
        lo = hi - count
        if lo < 0:
            raise CalendarError(
                f"only {hi} sessions available before {before.isoformat()}, need {count}"
            )
        return self._sessions[lo:hi]

    def sessions_between(self, start: dt.date, end: dt.date) -> int:
        """Number of sessions strictly after ``start`` up to and including ``end``."""
        return len(self.sessions(start + dt.timedelta(days=1), end))

    # -- bars -------------------------------------------------------------- #

    def session_bar_starts(self, session: dt.date) -> list[pd.Timestamp]:
        """Expected provider bar starts for one session, in order."""
        cached = self._bar_starts.get(session)
        if cached is not None:
            return cached
        open_ts, close_ts = self.session_window(session)
        starts: list[pd.Timestamp] = []
        cursor = open_ts
        step = pd.Timedelta(minutes=NOMINAL_BAR_MINUTES)
        while cursor < close_ts:
            starts.append(cursor)
            cursor = cursor + step
        self._bar_starts[session] = starts
        return starts

    def bars_per_session(self, session: dt.date) -> int:
        return len(self.session_bar_starts(session))

    def session_bars(
        self,
        session: dt.date,
        *,
        previous_end: pd.Timestamp | None = None,
    ) -> list[BarSpec]:
        """Scheduled bars of one session.

        The final bar is truncated at the official close, which is why a normal
        session ends with a 30-minute bar rather than a 60-minute one.
        """
        _open_ts, close_ts = self.session_window(session)
        bars: list[BarSpec] = []
        for slot, start in enumerate(self.session_bar_starts(session)):
            end = min(start + pd.Timedelta(minutes=NOMINAL_BAR_MINUTES), close_ts)
            if slot == 0:
                gap = (
                    float("nan")
                    if previous_end is None
                    else (start - previous_end).total_seconds() / 3600.0
                )
            else:
                gap = (start - bars[-1].end).total_seconds() / 3600.0
            bars.append(
                BarSpec(session=session, start=start, end=end, slot=slot, gap_hours=gap)
            )
        return bars

    def bar_index(
        self,
        sessions: Sequence[dt.date],
        *,
        previous_end: pd.Timestamp | None = None,
    ) -> list[BarSpec]:
        """Flatten the bar schedule of consecutive ``sessions`` into one series.

        ``previous_end`` seeds the overnight gap of the first session so that
        the leading bar's calendar channel is defined. Without it that one value
        is ``nan``.
        """
        bars: list[BarSpec] = []
        tail_end = previous_end
        for session in sessions:
            session_bars = self.session_bars(session, previous_end=tail_end)
            if not session_bars:
                raise CalendarError(f"{session.isoformat()} has no scheduled bars")
            bars.extend(session_bars)
            tail_end = session_bars[-1].end
        return bars

    def bars_ending_at(self, origin_session: dt.date, count: int) -> list[BarSpec]:
        """The last ``count`` scheduled bars up to and including ``origin_session``.

        Used to build a model context window. One extra session is consulted
        internally so that every returned bar has a defined overnight gap. The
        request fails loudly rather than returning a short window: a silently
        truncated context changes what the model sees.
        """
        if count < 1:
            raise CalendarError("count must be at least 1")
        # Sessions hold at most ceil(390/60) = 7 bars, so count//4 + 4 sessions
        # is a generous first guess even with half days in the window.
        needed = count // 4 + 4
        # The session index covers whatever range earlier lookups asked for,
        # which may begin just before this origin. Reach back far enough first
        # -- two calendar days per session clears weekends and holidays -- so
        # the answer cannot depend on the order of earlier calls.
        self._ensure_covered(
            origin_session - dt.timedelta(days=2 * needed + 14), origin_session
        )
        pos = self._position(origin_session)
        while True:
            if needed >= pos:
                needed = pos  # all available history before the origin
            start_pos = pos - needed
            seed_session = self._sessions[start_pos]
            window = self._sessions[start_pos + 1 : pos + 1]
            seed_bars = self.session_bars(seed_session)
            bars = self.bar_index(window, previous_end=seed_bars[-1].end)
            if len(bars) >= count:
                return bars[-count:]
            if needed >= pos:
                raise CalendarError(
                    f"only {len(bars)} scheduled bars exist before "
                    f"{origin_session.isoformat()}, need {count}"
                )
            needed *= 2

    # -- horizons ---------------------------------------------------------- #

    def horizon(self, origin_session: dt.date, horizon_sessions: int = 2) -> Horizon:
        """Future bars of the holding period that starts after ``origin_session``.

        Both the prediction length and the per-session terminal indices come
        from the calendar, so an early close shortens the horizon instead of
        shifting a terminal bar onto the wrong day.
        """
        if horizon_sessions < 1:
            raise CalendarError("horizon_sessions must be at least 1")
        self._ensure_covered(
            origin_session, origin_session + dt.timedelta(days=30 * horizon_sessions + 30)
        )
        pos = self._position(origin_session)
        if pos + horizon_sessions >= len(self._sessions):
            raise CalendarError(
                f"calendar does not cover {horizon_sessions} sessions after "
                f"{origin_session.isoformat()}"
            )
        future = self._sessions[pos + 1 : pos + 1 + horizon_sessions]
        origin_bars = self.session_bars(origin_session)
        bars = self.bar_index(future, previous_end=origin_bars[-1].end)

        terminal: list[int] = []
        cursor = 0
        for session in future:
            cursor += self.bars_per_session(session)
            terminal.append(cursor - 1)
        return Horizon(
            origin_session=origin_session,
            sessions=tuple(future),
            bars=tuple(bars),
            terminal_indices=tuple(terminal),
        )

    # -- operational timestamps -------------------------------------------- #

    def signal_time(self, session: dt.date, *, delay_minutes: int = 30) -> pd.Timestamp:
        """When the after-close run is scheduled for ``session``."""
        return self.session_close(session) + pd.Timedelta(minutes=delay_minutes)

    def data_deadline(self, session: dt.date, *, max_delay_minutes: int = 120) -> pd.Timestamp:
        """Latest acceptable snapshot time before the origin is DATA_UNAVAILABLE."""
        return self.session_close(session) + pd.Timedelta(minutes=max_delay_minutes)

    # -- validation -------------------------------------------------------- #

    def validate_provider_bars(
        self,
        session: dt.date,
        observed_starts: Iterable[pd.Timestamp],
    ) -> dict[str, list[str]]:
        """Compare provider bar starts against the expected schedule.

        Returns unexpected and missing starts as ISO strings. Callers treat a
        non-empty result as a quality event: differently anchored bars must not
        be used silently, and missing bars become masks rather than prices.
        """
        expected = set(self.session_bar_starts(session))
        observed = {pd.Timestamp(ts).tz_convert("UTC") for ts in observed_starts}
        return {
            "unexpected": sorted(ts.isoformat() for ts in observed - expected),
            "missing": sorted(ts.isoformat() for ts in expected - observed),
        }


def calendar_channel_matrix(bars: Sequence[BarSpec]) -> np.ndarray:
    """Stack the five calendar channels for ``bars`` into ``(n_bars, 5)``."""
    if not bars:
        return np.zeros((0, len(CALENDAR_CHANNEL_NAMES)), dtype=np.float32)
    return np.asarray([bar.calendar_channels() for bar in bars], dtype=np.float32)
