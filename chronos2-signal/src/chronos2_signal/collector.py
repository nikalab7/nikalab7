"""Bounded provider fetches with immutable provenance.

Access to Yahoo through ``yfinance`` is unofficial. Gaps, revisions and access
limits are expected, so this module is built around three rules:

* Requests stay inside the provider's real retention window. Splitting a
  request into smaller pieces cannot recover history the provider no longer
  keeps, so a wider request is not attempted as a workaround.
* Every response is stored once, content-addressed, with its retrieval time and
  declared adjustment mode. A later revision becomes a new snapshot.
* A failure never becomes a price. If a required input is missing the affected
  scope is suppressed; a stale snapshot is never presented as current data.

The collector is an interface with two implementations: the live Yahoo client,
and a fixture collector used by the integrity tests and by offline development.
Nothing downstream can tell them apart, which is what makes the invariants
testable without a network.
"""

from __future__ import annotations

import datetime as dt
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence

import pandas as pd

from .calendar_spec import ExchangeCalendar
from .provenance import stable_hash
from .storage import Ledger, SnapshotStore

__all__ = [
    "ProviderError",
    "RateLimited",
    "FetchRequest",
    "ProviderResponse",
    "Collector",
    "RetryPolicy",
    "fetch_with_retry",
    "YahooCollector",
    "FixtureCollector",
    "Ingestor",
    "IngestionRecord",
    "ScanReadiness",
    "ReadinessStatus",
]


class ProviderError(RuntimeError):
    """A transient or permanent provider failure."""


class RateLimited(ProviderError):
    """The provider signalled rate limiting; stop retrying aggressively."""


@dataclass(frozen=True)
class FetchRequest:
    """One bounded request for a symbol and interval."""

    symbol: str
    interval: str
    start: dt.date | None = None
    end: dt.date | None = None
    prepost: bool = False
    auto_adjust: bool = False
    actions: bool = True
    repair: bool = False

    def params(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "prepost": self.prepost,
            "auto_adjust": self.auto_adjust,
            "actions": self.actions,
            "repair": self.repair,
        }


@dataclass(frozen=True)
class ProviderResponse:
    """A provider response together with everything needed to trust it later."""

    request: FetchRequest
    frame: pd.DataFrame
    retrieved_at: dt.datetime
    declared_adjustment_mode: str
    provider: str
    provider_version: str
    actions: pd.DataFrame | None = None
    status: str = "ok"

    def payload_hash(self) -> str:
        """Content hash over the returned rows and the request that produced them."""
        frame = self.frame
        payload = {
            "request": self.request.params(),
            "index": [str(value) for value in frame.index],
            "columns": list(map(str, frame.columns)),
            "values": frame.to_numpy(dtype=object).tolist() if len(frame) else [],
        }
        if self.actions is not None and len(self.actions):
            payload["actions"] = {
                "index": [str(value) for value in self.actions.index],
                "columns": list(map(str, self.actions.columns)),
                "values": self.actions.to_numpy(dtype=object).tolist(),
            }
        return stable_hash(payload)

    def snapshot_id(self) -> str:
        """Snapshot identity: the content hash, prefixed for readability."""
        return f"{self.request.symbol}-{self.request.interval}-{self.payload_hash()[:24]}"


class Collector(Protocol):
    """Anything that can answer a bounded fetch request."""

    name: str
    version: str

    def fetch(self, request: FetchRequest) -> ProviderResponse:  # pragma: no cover
        ...


@dataclass(frozen=True)
class RetryPolicy:
    """Retry budget for transient failures.

    Deliberately small. On rate limiting there is no retry at all: hammering an
    unofficial endpoint is how access disappears.
    """

    max_retries: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 32.0
    jitter_seconds: float = 0.5
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=lambda: random.Random(42))

    def delay_for(self, attempt: int) -> float:
        raw = self.base_delay_seconds * (2 ** max(0, attempt - 1))
        return min(raw, self.max_delay_seconds) + self.rng.uniform(0.0, self.jitter_seconds)


def fetch_with_retry(
    collector: Collector,
    request: FetchRequest,
    policy: RetryPolicy | None = None,
) -> ProviderResponse:
    """Fetch with bounded exponential backoff.

    Raises the last :class:`ProviderError` if the budget is exhausted. Callers
    convert that into a suppression decision; they never substitute a price.
    """
    policy = policy or RetryPolicy()
    last: Exception | None = None
    for attempt in range(1, policy.max_retries + 1):
        try:
            return collector.fetch(request)
        except RateLimited:
            raise
        except ProviderError as exc:
            last = exc
            if attempt < policy.max_retries:
                policy.sleep(policy.delay_for(attempt))
    raise ProviderError(
        f"{request.symbol} {request.interval}: {policy.max_retries} attempts failed "
        f"({last})"
    ) from last


class YahooCollector:
    """Live Yahoo client.

    Concurrency is capped by a semaphore rather than by hope; the default of two
    in-flight requests matches the registered configuration.
    """

    name = "yahoo_via_yfinance"

    def __init__(self, *, max_parallel_requests: int = 2) -> None:
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProviderError(
                "yfinance is not installed; install the 'provider' extra before "
                "attempting a live fetch"
            ) from exc
        self._yfinance = yfinance
        self.version = getattr(yfinance, "__version__", "unknown")
        self._semaphore = threading.BoundedSemaphore(max_parallel_requests)

    def fetch(self, request: FetchRequest) -> ProviderResponse:
        with self._semaphore:
            return self._fetch_locked(request)

    def _fetch_locked(self, request: FetchRequest) -> ProviderResponse:
        retrieved_at = dt.datetime.now(dt.timezone.utc)
        ticker = self._yfinance.Ticker(request.symbol)
        try:
            frame = ticker.history(
                interval=request.interval,
                start=request.start,
                end=request.end,
                prepost=request.prepost,
                auto_adjust=request.auto_adjust,
                actions=request.actions,
                repair=request.repair,
                raise_errors=True,
            )
        except Exception as exc:  # provider exceptions are not a stable taxonomy
            message = str(exc).lower()
            if "too many requests" in message or "rate limit" in message or "429" in message:
                raise RateLimited(f"{request.symbol}: provider rate limited") from exc
            raise ProviderError(f"{request.symbol} {request.interval}: {exc}") from exc

        if frame is None or not len(frame):
            raise ProviderError(
                f"{request.symbol} {request.interval}: provider returned no rows"
            )
        frame = frame.rename(columns={str(column): str(column).lower() for column in frame.columns})
        action_columns = [c for c in ("dividends", "stock splits") if c in frame.columns]
        actions = frame[action_columns].copy() if action_columns else None
        if actions is not None:
            actions = actions.loc[(actions != 0).any(axis=1)]
        return ProviderResponse(
            request=request,
            frame=frame,
            retrieved_at=retrieved_at,
            # Recorded, not trusted: the adjustment convention is verified
            # against known actions before the series is used for features.
            declared_adjustment_mode=(
                "auto_adjust=True" if request.auto_adjust else "auto_adjust=False"
            ),
            provider=self.name,
            provider_version=self.version,
            actions=actions,
        )


@dataclass
class FixtureCollector:
    """Deterministic collector backed by in-memory frames.

    Used by the integrity tests and offline development. ``failures`` makes a
    symbol raise, which is how the suppression rules are exercised without
    depending on a real outage.
    """

    frames: Mapping[tuple[str, str], pd.DataFrame]
    action_frames: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    failures: Mapping[tuple[str, str], Exception] = field(default_factory=dict)
    retrieved_at: dt.datetime = field(
        default_factory=lambda: dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    )
    name: str = "fixture"
    version: str = "0"
    calls: list[FetchRequest] = field(default_factory=list)

    def fetch(self, request: FetchRequest) -> ProviderResponse:
        key = (request.symbol, request.interval)
        self.calls.append(request)
        if key in self.failures:
            raise self.failures[key]
        if key not in self.frames:
            raise ProviderError(f"fixture has no series for {key}")
        frame = self.frames[key]
        if request.start is not None:
            frame = frame.loc[[value.date() >= request.start for value in frame.index]]
        if request.end is not None:
            frame = frame.loc[[value.date() < request.end for value in frame.index]]
        return ProviderResponse(
            request=request,
            frame=frame.copy(),
            retrieved_at=self.retrieved_at,
            declared_adjustment_mode="auto_adjust=False",
            provider=self.name,
            provider_version=self.version,
            actions=self.action_frames.get(request.symbol),
        )


class ReadinessStatus(str, Enum):
    """Outcome of the pre-scan data gate."""

    OK = "OK"
    #: A required market-wide input is missing: the whole scan is suppressed.
    SUPPRESS_SCAN = "SUPPRESS_SCAN"
    #: Sector inputs failed: only those sectors are suppressed.
    SUPPRESS_SECTORS = "SUPPRESS_SECTORS"
    #: No valid snapshot by the deadline.
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


@dataclass(frozen=True)
class IngestionRecord:
    """What one ingest attempt produced."""

    symbol: str
    interval: str
    status: str
    snapshot_id: str | None = None
    rows: int = 0
    anchoring: Mapping[str, Sequence[str]] | None = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class ScanReadiness:
    """Whether an origin may produce signals, and for which sectors."""

    origin_session: dt.date
    status: ReadinessStatus
    suppressed_sectors: frozenset[str] = frozenset()
    reasons: tuple[str, ...] = ()

    @property
    def may_scan(self) -> bool:
        return self.status in (ReadinessStatus.OK, ReadinessStatus.SUPPRESS_SECTORS)

    def sector_allowed(self, sector: str) -> bool:
        return self.may_scan and sector not in self.suppressed_sectors


@dataclass
class Ingestor:
    """Fetch, verify anchoring, and persist snapshots and bars.

    The ingestor does not decide anything about trading. It produces an
    auditable record per symbol, and the scan gate turns those records into a
    suppression decision.
    """

    calendar: ExchangeCalendar
    ledger: Ledger
    snapshots: SnapshotStore
    collector: Collector
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def ingest(self, request: FetchRequest) -> IngestionRecord:
        try:
            response = fetch_with_retry(self.collector, request, self.retry_policy)
        except RateLimited as exc:
            return IngestionRecord(
                symbol=request.symbol,
                interval=request.interval,
                status="rate_limited",
                message=str(exc),
            )
        except ProviderError as exc:
            return IngestionRecord(
                symbol=request.symbol,
                interval=request.interval,
                status="failed",
                message=str(exc),
            )

        snapshot_id = response.snapshot_id()
        payload_hash = response.payload_hash()
        try:
            payload_path = self.snapshots.write(
                response.frame, request.symbol, request.interval, snapshot_id
            )
            path_text: str | None = str(payload_path)
        except Exception:
            # An identical snapshot is already on disk. Content addressing means
            # the bytes match, so reuse them rather than overwrite.
            path_text = str(
                self.snapshots.path_for(request.symbol, request.interval, snapshot_id)
            )

        self.ledger.record_snapshot(
            snapshot_id=snapshot_id,
            provider=response.provider,
            provider_version=response.provider_version,
            symbol=request.symbol,
            interval=request.interval,
            requested_start=request.start,
            requested_end=request.end,
            retrieved_at=response.retrieved_at,
            declared_adjustment_mode=response.declared_adjustment_mode,
            row_count=int(len(response.frame)),
            payload_path=path_text,
            payload_sha256=payload_hash,
            request_params=request.params(),
            status=response.status,
        )

        anchoring = self._persist_bars(response, snapshot_id)
        self._persist_actions(response, snapshot_id)
        return IngestionRecord(
            symbol=request.symbol,
            interval=request.interval,
            status="ok",
            snapshot_id=snapshot_id,
            rows=int(len(response.frame)),
            anchoring=anchoring,
        )

    def _persist_bars(
        self, response: ProviderResponse, snapshot_id: str
    ) -> dict[str, list[str]]:
        request = response.request
        frame = response.frame
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        intraday = request.interval not in {"1d", "1wk", "1mo"}

        # "in_progress": bars that had not ended when the response was retrieved.
        # Storing one would let a forming bar be read later as finished.
        anchoring: dict[str, list[str]] = {"unexpected": [], "missing": [], "in_progress": []}
        retrieved = pd.Timestamp(response.retrieved_at)
        rows = []
        if intraday:
            by_session: dict[dt.date, list[pd.Timestamp]] = {}
            for timestamp in index:
                by_session.setdefault(
                    timestamp.tz_convert(self.calendar.exchange_timezone).date(), []
                ).append(timestamp)
            for session, starts in sorted(by_session.items()):
                if not self.calendar.is_session(session):
                    anchoring["unexpected"].extend(ts.isoformat() for ts in starts)
                    continue
                report = self.calendar.validate_provider_bars(session, starts)
                anchoring["unexpected"].extend(report["unexpected"])
                anchoring["missing"].extend(report["missing"])
                schedule = {
                    bar.start: bar for bar in self.calendar.session_bars(session)
                }
                for start in starts:
                    bar = schedule.get(start)
                    if bar is None:
                        continue  # already reported as unexpected anchoring
                    if bar.end > retrieved:
                        anchoring["in_progress"].append(start.isoformat())
                        continue
                    record = frame.loc[index == start]
                    rows.append(
                        self._bar_row(
                            request,
                            record,
                            session=session,
                            bar_start=bar.start,
                            bar_end=bar.end,
                            snapshot_id=snapshot_id,
                            retrieved_at=response.retrieved_at,
                        )
                    )
        else:
            for timestamp in index:
                session = timestamp.tz_convert(self.calendar.exchange_timezone).date()
                if not self.calendar.is_session(session):
                    anchoring["unexpected"].append(timestamp.isoformat())
                    continue
                open_ts, close_ts = self.calendar.session_window(session)
                if close_ts > retrieved:
                    anchoring["in_progress"].append(timestamp.isoformat())
                    continue
                record = frame.loc[index == timestamp]
                rows.append(
                    self._bar_row(
                        request,
                        record,
                        session=session,
                        bar_start=open_ts,
                        bar_end=close_ts,
                        snapshot_id=snapshot_id,
                        retrieved_at=response.retrieved_at,
                    )
                )
        self.ledger.record_bars(rows)
        return anchoring

    @staticmethod
    def _bar_row(
        request: FetchRequest,
        record: pd.DataFrame,
        *,
        session: dt.date,
        bar_start: pd.Timestamp,
        bar_end: pd.Timestamp,
        snapshot_id: str,
        retrieved_at: dt.datetime,
    ) -> dict[str, object]:
        def value(column: str) -> float | None:
            if column not in record.columns or not len(record):
                return None
            raw = record.iloc[0][column]
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return None
            return number

        close = value("close")
        adjusted = value("adj close")
        if adjusted is None:
            adjusted = value("adj_close")
        return {
            "symbol": request.symbol,
            "interval": request.interval,
            "bar_start": bar_start,
            "bar_end": bar_end,
            "session": session,
            "open": value("open"),
            "high": value("high"),
            "low": value("low"),
            "close": close,
            "volume": value("volume"),
            "adj_close": adjusted,
            "observed": close is not None,
            "snapshot_id": snapshot_id,
            "retrieved_at": retrieved_at,
        }

    def _persist_actions(self, response: ProviderResponse, snapshot_id: str) -> None:
        actions = response.actions
        if actions is None or not len(actions):
            return
        index = pd.DatetimeIndex(pd.to_datetime(actions.index, utc=True))
        for position, timestamp in enumerate(index):
            row = actions.iloc[position]
            ex_date = timestamp.tz_convert(self.calendar.exchange_timezone).date()
            for column, action_type in (
                ("dividends", "dividend"),
                ("stock splits", "split"),
                ("splits", "split"),
            ):
                if column not in actions.columns:
                    continue
                try:
                    magnitude = float(row[column])
                except (TypeError, ValueError):
                    continue
                if magnitude == 0.0:
                    continue
                self.ledger.record_corporate_action(
                    symbol=response.request.symbol,
                    ex_date=ex_date,
                    action_type=action_type,
                    value=magnitude,
                    snapshot_id=snapshot_id,
                    retrieved_at=response.retrieved_at,
                    # A provider action row is observable once retrieved; the
                    # ex-date itself is not evidence of advance knowledge.
                    available_at=response.retrieved_at,
                )


def assess_readiness(
    *,
    origin_session: dt.date,
    now: dt.datetime,
    calendar: ExchangeCalendar,
    market_record: IngestionRecord,
    sector_records: Mapping[str, IngestionRecord],
    max_delay_minutes: int,
) -> ScanReadiness:
    """Turn ingestion records into a scan decision.

    The rules are the registered ones: a failed market proxy suppresses the
    entire scan, a failed sector input suppresses that sector, and no valid
    snapshot by close plus ``max_delay_minutes`` makes the origin
    ``DATA_UNAVAILABLE``. None of these paths substitutes a stale snapshot.
    """
    deadline = calendar.data_deadline(origin_session, max_delay_minutes=max_delay_minutes)
    past_deadline = pd.Timestamp(now) > deadline
    reasons: list[str] = []

    if not market_record.ok:
        reasons.append(
            f"market proxy {market_record.symbol} unavailable ({market_record.status})"
        )
        if past_deadline:
            # No valid snapshot by close plus the allowed delay. The origin is
            # closed out; a later run does not backdate the missed alert.
            reasons.append(
                f"past data deadline {deadline.isoformat()}; missed alerts are not backdated"
            )
            status = ReadinessStatus.DATA_UNAVAILABLE
        else:
            # Still inside the window, so a later attempt at this origin may
            # succeed. Suppressing is not the same as declaring the origin dead.
            status = ReadinessStatus.SUPPRESS_SCAN
        return ScanReadiness(
            origin_session=origin_session, status=status, reasons=tuple(reasons)
        )

    if past_deadline:
        # Required data did arrive, only late. That is a monitoring event, not a
        # reason to discard a valid snapshot, so the scan proceeds and the
        # lateness is recorded on the signal.
        reasons.append(
            f"snapshot accepted after the {deadline.isoformat()} deadline; "
            "provider lag exceeded the buffer"
        )

    failed_sectors = frozenset(
        sector for sector, record in sector_records.items() if not record.ok
    )
    if failed_sectors:
        reasons.extend(
            f"sector input {sector_records[sector].symbol} unavailable "
            f"({sector_records[sector].status})"
            for sector in sorted(failed_sectors)
        )
        return ScanReadiness(
            origin_session=origin_session,
            status=ReadinessStatus.SUPPRESS_SECTORS,
            suppressed_sectors=failed_sectors,
            reasons=tuple(reasons),
        )
    return ScanReadiness(
        origin_session=origin_session,
        status=ReadinessStatus.OK,
        reasons=tuple(reasons),
    )
