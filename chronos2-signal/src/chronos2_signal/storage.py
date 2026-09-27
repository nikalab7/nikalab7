"""Persistence: an append-only SQLite ledger plus Parquet snapshot files.

Design constraints this module enforces rather than documents:

* Every price and event record carries both an observation time and a
  retrieval/availability time, so a read can be replayed as of an old vintage.
* A provider revision creates a *new* snapshot. It never overwrites the rows an
  earlier decision was made from.
* The ``forecasts`` table is append-only at the database level: updates and
  deletes abort. Forecasts are written before outcomes exist and are never
  edited once an outcome is known.
* ``signals`` is unique on the deduplication key
  ``(strategy_version, issuer_id, signal_session, holding_period)``, so a
  retried run cannot emit a second alert.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .holdout import HoldoutGuard

__all__ = ["StorageError", "Ledger", "SnapshotStore", "LEDGER_TABLES"]

#: The minimum table set required by the protocol, plus the daily portfolio
#: marks the block bootstrap needs.
LEDGER_TABLES = (
    "instruments",
    "source_snapshots",
    "bars",
    "corporate_actions",
    "event_snapshots",
    "origin_features",
    "forecasts",
    "fit_manifests",
    "signals",
    "paper_positions",
    "fills",
    "outcomes",
    "run_log",
    "portfolio_daily",
)

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Frozen watchlist manifest. `selection_timestamp` is when the issuer entered
-- the manifest; it is fixed before any performance metric is produced.
CREATE TABLE IF NOT EXISTS instruments (
    symbol              TEXT PRIMARY KEY,
    issuer_id           TEXT NOT NULL,
    exchange            TEXT NOT NULL,
    sector              TEXT NOT NULL,
    sector_etf          TEXT NOT NULL,
    share_class         TEXT,
    listing_history     TEXT,
    selection_timestamp TEXT NOT NULL,
    candidate_roster    TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    notes               TEXT
);

-- Immutable provider responses. `payload_sha256` is over the stored file.
CREATE TABLE IF NOT EXISTS source_snapshots (
    snapshot_id              TEXT PRIMARY KEY,
    provider                 TEXT NOT NULL,
    provider_version         TEXT NOT NULL,
    symbol                   TEXT NOT NULL,
    interval                 TEXT NOT NULL,
    requested_start          TEXT,
    requested_end            TEXT,
    retrieved_at             TEXT NOT NULL,
    declared_adjustment_mode TEXT NOT NULL,
    row_count                INTEGER NOT NULL,
    payload_path             TEXT,
    payload_sha256           TEXT,
    request_params           TEXT NOT NULL,
    status                   TEXT NOT NULL
);

-- Canonical bars. Keyed by snapshot as well as bar start so a later revision
-- adds rows instead of mutating the vintage an old decision used.
CREATE TABLE IF NOT EXISTS bars (
    symbol       TEXT NOT NULL,
    interval     TEXT NOT NULL,
    bar_start    TEXT NOT NULL,
    bar_end      TEXT NOT NULL,
    session      TEXT NOT NULL,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,
    adj_close    REAL,
    observed     INTEGER NOT NULL,
    snapshot_id  TEXT NOT NULL REFERENCES source_snapshots(snapshot_id),
    retrieved_at TEXT NOT NULL,
    PRIMARY KEY (symbol, interval, bar_start, snapshot_id)
);
CREATE INDEX IF NOT EXISTS bars_lookup ON bars (symbol, interval, session, bar_start);

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol       TEXT NOT NULL,
    ex_date      TEXT NOT NULL,
    action_type  TEXT NOT NULL,
    value        REAL NOT NULL,
    snapshot_id  TEXT NOT NULL REFERENCES source_snapshots(snapshot_id),
    retrieved_at TEXT NOT NULL,
    available_at TEXT,
    PRIMARY KEY (symbol, ex_date, action_type, snapshot_id)
);

-- Event annotations. `first_seen_at` is what the system knew and when; it is
-- never backfilled from a later calendar.
CREATE TABLE IF NOT EXISTS event_snapshots (
    event_id      TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    symbol        TEXT,
    published_at  TEXT,
    event_time    TEXT,
    first_seen_at TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL,
    revision      TEXT,
    status        TEXT NOT NULL,
    payload       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS origin_features (
    origin_session TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    feature_set    TEXT NOT NULL,
    payload        TEXT NOT NULL,
    features_hash  TEXT NOT NULL,
    snapshot_hash  TEXT NOT NULL,
    built_at       TEXT NOT NULL,
    PRIMARY KEY (origin_session, symbol, feature_set)
);

-- Append-only: see the triggers below.
CREATE TABLE IF NOT EXISTS forecasts (
    cache_key            TEXT PRIMARY KEY,
    symbol               TEXT NOT NULL,
    origin_session       TEXT NOT NULL,
    variant              TEXT NOT NULL,
    checkpoint_id        TEXT NOT NULL,
    checkpoint_revision  TEXT NOT NULL,
    context_length       INTEGER NOT NULL,
    prediction_length    INTEGER NOT NULL,
    quantile_levels      TEXT NOT NULL,
    quantiles            TEXT NOT NULL,
    terminal_indices     TEXT NOT NULL,
    horizon_sessions     TEXT NOT NULL,
    origin_close         REAL,
    status               TEXT NOT NULL,
    diagnostics          TEXT,
    snapshot_hash        TEXT NOT NULL,
    generated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS forecasts_origin ON forecasts (origin_session, variant, symbol);

CREATE TRIGGER IF NOT EXISTS forecasts_no_update
BEFORE UPDATE ON forecasts
BEGIN
    SELECT RAISE(ABORT, 'forecasts is append-only: a stored forecast may not be revised');
END;

CREATE TRIGGER IF NOT EXISTS forecasts_no_delete
BEFORE DELETE ON forecasts
BEGIN
    SELECT RAISE(ABORT, 'forecasts is append-only: a stored forecast may not be deleted');
END;

CREATE TABLE IF NOT EXISTS fit_manifests (
    fit_id                    TEXT PRIMARY KEY,
    variant                   TEXT NOT NULL,
    fold_index                INTEGER NOT NULL,
    fit_start_session         TEXT NOT NULL,
    fit_end_session           TEXT NOT NULL,
    calibration_start_session TEXT NOT NULL,
    calibration_end_session   TEXT NOT NULL,
    purge_sessions            INTEGER NOT NULL,
    n_fit_rows                INTEGER NOT NULL,
    n_fit_dates               INTEGER NOT NULL,
    n_calibration_rows        INTEGER NOT NULL,
    n_calibration_dates       INTEGER NOT NULL,
    label_timestamps_checked  INTEGER NOT NULL,
    clip_fraction             REAL,
    design_fingerprint        TEXT NOT NULL,
    environment_digest        TEXT NOT NULL,
    code_revision             TEXT,
    artifact_path             TEXT,
    artifact_hash             TEXT,
    created_at                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id                 TEXT PRIMARY KEY,
    dedup_key                 TEXT NOT NULL UNIQUE,
    strategy_version          TEXT NOT NULL,
    issuer_id                 TEXT NOT NULL,
    symbol                    TEXT NOT NULL,
    sector                    TEXT NOT NULL,
    signal_session            TEXT NOT NULL,
    holding_period            TEXT NOT NULL,
    variant                   TEXT NOT NULL,
    generated_at              TEXT NOT NULL,
    data_timestamp            TEXT NOT NULL,
    estimated_net_return      REAL NOT NULL,
    estimated_net_return_stress REAL NOT NULL,
    calibrated_probability    REAL NOT NULL,
    sigma_2d                  REAL NOT NULL,
    rank_score                REAL NOT NULL,
    rank_position             INTEGER NOT NULL,
    forecast_cache_key        TEXT,
    fit_id                    TEXT,
    decision                  TEXT NOT NULL,
    suppression_reason        TEXT,
    output_label              TEXT NOT NULL,
    payload                   TEXT NOT NULL,
    notified_at               TEXT,
    expires_at                TEXT
);
CREATE INDEX IF NOT EXISTS signals_session ON signals (signal_session, variant);

CREATE TABLE IF NOT EXISTS paper_positions (
    position_id          TEXT PRIMARY KEY,
    signal_id            TEXT NOT NULL REFERENCES signals(signal_id),
    variant              TEXT NOT NULL,
    symbol               TEXT NOT NULL,
    issuer_id            TEXT NOT NULL,
    sector               TEXT NOT NULL,
    entry_session        TEXT NOT NULL,
    planned_exit_session TEXT NOT NULL,
    exit_session         TEXT,
    shares               REAL NOT NULL,
    entry_reference_price REAL NOT NULL,
    entry_fill_price     REAL NOT NULL,
    entry_notional       REAL NOT NULL,
    exit_reference_price REAL,
    exit_fill_price      REAL,
    cash_dividends       REAL NOT NULL DEFAULT 0.0,
    share_adjustment     REAL NOT NULL DEFAULT 1.0,
    status               TEXT NOT NULL,
    cost_scenario        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id           TEXT PRIMARY KEY,
    position_id       TEXT NOT NULL REFERENCES paper_positions(position_id),
    kind              TEXT NOT NULL,
    session           TEXT NOT NULL,
    reference_price   REAL NOT NULL,
    executed_price    REAL NOT NULL,
    shares            REAL NOT NULL,
    slippage_fraction REAL NOT NULL,
    fee               REAL NOT NULL,
    cost_scenario     TEXT NOT NULL,
    source            TEXT NOT NULL,
    recorded_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id             TEXT PRIMARY KEY,
    signal_id              TEXT NOT NULL REFERENCES signals(signal_id),
    position_id            TEXT REFERENCES paper_positions(position_id),
    variant                TEXT NOT NULL,
    symbol                 TEXT NOT NULL,
    origin_session         TEXT NOT NULL,
    entry_session          TEXT NOT NULL,
    exit_session           TEXT NOT NULL,
    r_net_base             REAL NOT NULL,
    r_net_stress           REAL NOT NULL,
    r_net_severe           REAL NOT NULL,
    r_gross                REAL NOT NULL,
    max_adverse_excursion  REAL,
    label_available_at     TEXT NOT NULL,
    matured_at             TEXT NOT NULL,
    accounting             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outcomes_origin ON outcomes (origin_session, variant);

CREATE TABLE IF NOT EXISTS portfolio_daily (
    variant        TEXT NOT NULL,
    session        TEXT NOT NULL,
    equity         REAL NOT NULL,
    cash           REAL NOT NULL,
    gross_exposure REAL NOT NULL,
    open_positions INTEGER NOT NULL,
    net_return     REAL NOT NULL,
    peak_equity    REAL NOT NULL,
    drawdown       REAL NOT NULL,
    paused         INTEGER NOT NULL,
    PRIMARY KEY (variant, session)
);

CREATE TABLE IF NOT EXISTS run_log (
    run_id             TEXT PRIMARY KEY,
    run_kind           TEXT NOT NULL,
    origin_session     TEXT,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    status             TEXT NOT NULL,
    message            TEXT,
    design_fingerprint TEXT NOT NULL,
    environment_digest TEXT NOT NULL,
    counts             TEXT
);
"""


class StorageError(RuntimeError):
    """Raised on a persistence rule violation."""


def _iso(value: Any) -> str | None:
    """Render a timestamp or date as an ISO string for storage."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            raise StorageError(f"naive datetime {value!r}: internal timestamps must be UTC-aware")
        return value.astimezone(dt.timezone.utc).isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    raise StorageError(f"cannot store {value!r} as a timestamp")


def _date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    raise StorageError(f"cannot interpret {value!r} as a session date")


class Ledger:
    """SQLite ledger for manifests, run state, signals, positions and outcomes."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_version: int = 1,
        guard: HoldoutGuard | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema_version = schema_version
        self.guard = guard or HoldoutGuard.open()
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._assert_schema_version()

    # -- lifecycle --------------------------------------------------------- #

    def _assert_schema_version(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(self.schema_version),),
            )
            return
        stored = int(row["value"])
        if stored != self.schema_version:
            raise StorageError(
                f"ledger at {self.path} has schema version {stored}, code expects "
                f"{self.schema_version}; migrate deliberately rather than in place"
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def tables(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        return [row["name"] for row in rows]

    # -- generic helpers --------------------------------------------------- #

    def _insert(self, table: str, row: Mapping[str, Any], *, mode: str = "strict") -> None:
        columns = list(row)
        placeholders = ", ".join("?" for _ in columns)
        verb = {
            "strict": "INSERT",
            "ignore": "INSERT OR IGNORE",
            "replace": "INSERT OR REPLACE",
        }[mode]
        sql = (
            f"{verb} INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        )
        try:
            self._conn.execute(sql, [row[column] for column in columns])
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"{table}: {exc}") from exc

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql, list(params)).fetchall()
        return [dict(row) for row in rows]

    # -- instruments ------------------------------------------------------- #

    def upsert_instrument(self, **fields: Any) -> None:
        """Register a watchlist member.

        Replacement is allowed only for descriptive columns; the manifest
        freeze itself is enforced by :mod:`chronos2_signal.universe`, which
        refuses to re-select after performance has been produced.
        """
        row = dict(fields)
        row["selection_timestamp"] = _iso(row["selection_timestamp"])
        self._insert("instruments", row, mode="replace")

    def instruments(self, *, status: str | None = "active") -> list[dict[str, Any]]:
        if status is None:
            return self.query("SELECT * FROM instruments ORDER BY symbol")
        return self.query(
            "SELECT * FROM instruments WHERE status = ? ORDER BY symbol", [status]
        )

    # -- snapshots and bars ------------------------------------------------ #

    def record_snapshot(self, **fields: Any) -> None:
        row = dict(fields)
        row["retrieved_at"] = _iso(row["retrieved_at"])
        row["requested_start"] = _iso(row.get("requested_start"))
        row["requested_end"] = _iso(row.get("requested_end"))
        if isinstance(row.get("request_params"), Mapping):
            row["request_params"] = json.dumps(row["request_params"], sort_keys=True)
        self._insert("source_snapshots", row, mode="ignore")

    def record_bars(self, rows: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for row in rows:
            payload = dict(row)
            payload["bar_start"] = _iso(payload["bar_start"])
            payload["bar_end"] = _iso(payload["bar_end"])
            payload["session"] = _iso(payload["session"])
            payload["retrieved_at"] = _iso(payload["retrieved_at"])
            payload["observed"] = int(bool(payload["observed"]))
            self._insert("bars", payload, mode="ignore")
            count += 1
        return count

    def read_bars(
        self,
        symbol: str,
        interval: str,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        as_of: dt.datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        """Point-in-time bar read.

        For each bar start, the row from the most recent snapshot retrieved at
        or before ``as_of`` is returned. Passing ``as_of`` is how a backtest
        replays the vintage a decision actually saw instead of today's
        corrected history.
        """
        clauses = ["symbol = ?", "interval = ?"]
        params: list[Any] = [symbol, interval]
        if start is not None:
            clauses.append("session >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("session <= ?")
            params.append(end.isoformat())
        if as_of is not None:
            clauses.append("retrieved_at <= ?")
            params.append(_iso(as_of))
        where = " AND ".join(clauses)
        sql = f"""
            SELECT b.* FROM bars b
            JOIN (
                SELECT bar_start, MAX(retrieved_at) AS latest
                FROM bars WHERE {where}
                GROUP BY bar_start
            ) pick
            ON b.bar_start = pick.bar_start AND b.retrieved_at = pick.latest
            WHERE {where}
            ORDER BY b.bar_start
        """
        return self.query(sql, params + params)

    def record_corporate_action(self, **fields: Any) -> None:
        row = dict(fields)
        row["ex_date"] = _iso(row["ex_date"])
        row["retrieved_at"] = _iso(row["retrieved_at"])
        row["available_at"] = _iso(row.get("available_at"))
        self._insert("corporate_actions", row, mode="ignore")

    def corporate_actions(
        self,
        symbol: str,
        *,
        as_of: dt.datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["symbol = ?"]
        params: list[Any] = [symbol]
        if as_of is not None:
            clauses.append("retrieved_at <= ?")
            params.append(_iso(as_of))
        return self.query(
            f"SELECT * FROM corporate_actions WHERE {' AND '.join(clauses)} "
            "ORDER BY ex_date, action_type",
            params,
        )

    # -- events ------------------------------------------------------------ #

    def record_event(self, **fields: Any) -> None:
        row = dict(fields)
        for key in ("published_at", "event_time", "first_seen_at", "retrieved_at"):
            if key in row:
                row[key] = _iso(row[key])
        if isinstance(row.get("payload"), Mapping):
            row["payload"] = json.dumps(row["payload"], sort_keys=True)
        self._insert("event_snapshots", row, mode="ignore")

    def events_known_at(
        self, as_of: dt.datetime | str, *, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """Events whose first observation precedes ``as_of``.

        An event the system had not yet seen is invisible here, which is what
        keeps an earnings annotation from becoming retrospective knowledge.
        """
        clauses = ["first_seen_at <= ?"]
        params: list[Any] = [_iso(as_of)]
        if symbol is not None:
            clauses.append("(symbol = ? OR symbol IS NULL)")
            params.append(symbol)
        return self.query(
            f"SELECT * FROM event_snapshots WHERE {' AND '.join(clauses)} "
            "ORDER BY first_seen_at",
            params,
        )

    # -- features and forecasts -------------------------------------------- #

    def record_origin_features(self, **fields: Any) -> None:
        row = dict(fields)
        row["origin_session"] = _iso(row["origin_session"])
        row["built_at"] = _iso(row["built_at"])
        if isinstance(row.get("payload"), Mapping):
            row["payload"] = json.dumps(row["payload"], sort_keys=True, default=str)
        self._insert("origin_features", row, mode="replace")

    def record_forecast(self, **fields: Any) -> str:
        """Append one forecast.

        Re-recording an identical forecast under the same cache key is a no-op,
        which keeps retries idempotent. Re-recording *different* numbers under
        the same key raises: that would mean the key does not describe the
        inputs.
        """
        row = dict(fields)
        row["origin_session"] = _iso(row["origin_session"])
        row["generated_at"] = _iso(row["generated_at"])
        for key in ("quantile_levels", "quantiles", "terminal_indices", "horizon_sessions", "diagnostics"):
            if key in row and not isinstance(row[key], (str, type(None))):
                row[key] = json.dumps(row[key], default=str)

        existing = self._conn.execute(
            "SELECT * FROM forecasts WHERE cache_key = ?", (row["cache_key"],)
        ).fetchone()
        if existing is not None:
            for column, value in row.items():
                if existing[column] != value:
                    raise StorageError(
                        f"forecast {row['cache_key']} already stored with a different "
                        f"{column}; the cache key must cover every input that changes "
                        "the numbers"
                    )
            return str(row["cache_key"])
        self._insert("forecasts", row)
        return str(row["cache_key"])

    def read_forecast(self, cache_key: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM forecasts WHERE cache_key = ?", [cache_key])
        return rows[0] if rows else None

    def forecasts_for_origin(
        self, origin_session: dt.date, variant: str
    ) -> list[dict[str, Any]]:
        self.guard.check(origin_session, purpose="forecast read")
        return self.query(
            "SELECT * FROM forecasts WHERE origin_session = ? AND variant = ? "
            "ORDER BY symbol",
            [origin_session.isoformat(), variant],
        )

    # -- fits -------------------------------------------------------------- #

    def record_fit_manifest(self, **fields: Any) -> None:
        row = dict(fields)
        for key in (
            "fit_start_session",
            "fit_end_session",
            "calibration_start_session",
            "calibration_end_session",
            "created_at",
        ):
            row[key] = _iso(row[key])
        row["label_timestamps_checked"] = int(bool(row["label_timestamps_checked"]))
        self._insert("fit_manifests", row, mode="replace")

    # -- signals ----------------------------------------------------------- #

    @staticmethod
    def dedup_key(
        strategy_version: str,
        issuer_id: str,
        signal_session: dt.date,
        holding_period: str,
    ) -> str:
        """The protocol's deduplication key, rendered as a string."""
        return "|".join(
            [strategy_version, issuer_id, _iso(signal_session) or "", holding_period]
        )

    def record_signal(self, **fields: Any) -> str:
        """Persist a signal row.

        Commit precedes notification, so a retry after a delivery failure finds
        the row already present and does not emit a duplicate alert.
        """
        row = dict(fields)
        row["signal_session"] = _iso(row["signal_session"])
        row["generated_at"] = _iso(row["generated_at"])
        row["data_timestamp"] = _iso(row["data_timestamp"])
        row["expires_at"] = _iso(row.get("expires_at"))
        row["notified_at"] = _iso(row.get("notified_at"))
        if isinstance(row.get("payload"), Mapping):
            row["payload"] = json.dumps(row["payload"], sort_keys=True, default=str)
        existing = self._conn.execute(
            "SELECT signal_id FROM signals WHERE dedup_key = ?", (row["dedup_key"],)
        ).fetchone()
        if existing is not None:
            return str(existing["signal_id"])
        self._insert("signals", row)
        return str(row["signal_id"])

    def mark_notified(self, signal_id: str, when: dt.datetime) -> None:
        self._conn.execute(
            "UPDATE signals SET notified_at = ? WHERE signal_id = ? AND notified_at IS NULL",
            (_iso(when), signal_id),
        )

    def signals_for_session(
        self, signal_session: dt.date, variant: str | None = None
    ) -> list[dict[str, Any]]:
        self.guard.check(signal_session, purpose="signal read")
        if variant is None:
            return self.query(
                "SELECT * FROM signals WHERE signal_session = ? ORDER BY rank_position",
                [signal_session.isoformat()],
            )
        return self.query(
            "SELECT * FROM signals WHERE signal_session = ? AND variant = ? "
            "ORDER BY rank_position",
            [signal_session.isoformat(), variant],
        )

    # -- positions, fills, outcomes ---------------------------------------- #

    def record_position(self, **fields: Any) -> None:
        row = dict(fields)
        for key in ("entry_session", "planned_exit_session", "exit_session"):
            row[key] = _iso(row.get(key))
        self._insert("paper_positions", row, mode="replace")

    def record_fill(self, **fields: Any) -> None:
        row = dict(fields)
        row["session"] = _iso(row["session"])
        row["recorded_at"] = _iso(row["recorded_at"])
        self._insert("fills", row, mode="replace")

    def record_outcome(self, **fields: Any) -> None:
        row = dict(fields)
        for key in ("origin_session", "entry_session", "exit_session", "label_available_at", "matured_at"):
            row[key] = _iso(row[key])
        if isinstance(row.get("accounting"), Mapping):
            row["accounting"] = json.dumps(row["accounting"], sort_keys=True, default=str)
        self._insert("outcomes", row, mode="replace")

    def matured_outcomes(
        self,
        *,
        variant: str,
        as_of: dt.datetime | str | None = None,
        visible_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Outcomes whose labels were observable by ``as_of``.

        A scheduled refit queries only matured labels: an outcome that had not
        yet been observed at fit time must not enter that fit, even though it
        sits in the same table today.
        """
        clauses = ["variant = ?"]
        params: list[Any] = [variant]
        if as_of is not None:
            clauses.append("label_available_at <= ?")
            params.append(_iso(as_of))
        rows = self.query(
            f"SELECT * FROM outcomes WHERE {' AND '.join(clauses)} ORDER BY origin_session, symbol",
            params,
        )
        if not visible_only:
            return rows
        visible = set(self.guard.filter_visible({_date(row["origin_session"]) for row in rows}))
        return [row for row in rows if _date(row["origin_session"]) in visible]

    def record_portfolio_day(self, **fields: Any) -> None:
        row = dict(fields)
        row["session"] = _iso(row["session"])
        row["paused"] = int(bool(row["paused"]))
        self._insert("portfolio_daily", row, mode="replace")

    def portfolio_days(self, variant: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM portfolio_daily WHERE variant = ? ORDER BY session", [variant]
        )

    # -- run log ----------------------------------------------------------- #

    def start_run(self, **fields: Any) -> str:
        row = dict(fields)
        row["started_at"] = _iso(row["started_at"])
        row["origin_session"] = _iso(row.get("origin_session"))
        if isinstance(row.get("counts"), Mapping):
            row["counts"] = json.dumps(row["counts"], sort_keys=True)
        self._insert("run_log", row, mode="replace")
        return str(row["run_id"])

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: dt.datetime,
        message: str | None = None,
        counts: Mapping[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE run_log SET status = ?, finished_at = ?, message = ?, counts = ? "
            "WHERE run_id = ?",
            (
                status,
                _iso(finished_at),
                message,
                json.dumps(counts, sort_keys=True, default=str) if counts else None,
                run_id,
            ),
        )


@dataclass
class SnapshotStore:
    """Immutable Parquet store for provider responses and numeric series.

    Snapshot files are written once under a content-addressed name. A write
    that would change existing bytes raises instead of overwriting, because an
    earlier decision was made from those bytes.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, interval: str, snapshot_id: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.root / interval / safe_symbol / f"{snapshot_id}.parquet"

    def write(self, frame: Any, symbol: str, interval: str, snapshot_id: str) -> Path:
        target = self.path_for(symbol, interval, snapshot_id)
        if target.exists():
            raise StorageError(
                f"snapshot {target} already exists; provider revisions are stored as "
                "new snapshots, never as overwrites"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=True)
        return target

    def read(self, symbol: str, interval: str, snapshot_id: str) -> Any:
        import pandas as pd

        target = self.path_for(symbol, interval, snapshot_id)
        if not target.exists():
            raise StorageError(f"snapshot {target} not found")
        return pd.read_parquet(target)

    def list_snapshots(self, symbol: str, interval: str) -> list[str]:
        directory = self.root / interval / symbol.replace("/", "_")
        if not directory.exists():
            return []
        return sorted(path.stem for path in directory.glob("*.parquet"))
