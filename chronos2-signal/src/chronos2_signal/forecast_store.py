"""Ledger-backed forecast cache: forecasts are recorded before outcomes exist.

Section 15 requires forecasts to be saved before their outcomes, in an
append-only table distinct from the outcomes attached later. This module is
where that happens: the pipeline routes every forecast through
:meth:`ForecastCache.predict`, which records each new forecast the moment it is
produced -- at batch time, before any label for that origin is computed -- and
serves an identical request from the ledger instead of recomputing it.

The key covers everything that can change the numbers: the forecaster's
identity (checkpoint revision, package version, input schema), the symbol and
origin, context and prediction lengths, the exact channel set, the
cross-learning flag and the hash of the source data. It deliberately excludes
the batch budget, because batching must not change a prediction -- that is
invariant 8 -- and a key that included it would hide a violation instead of
exposing one.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .features import ChronosTask
from .forecaster import Forecaster, ForecastResult, ForecastStatus
from .provenance import forecast_cache_key, stable_hash
from .storage import Ledger

__all__ = ["ForecastCache"]


@dataclass
class ForecastCache:
    """Append-only forecast store in front of a forecaster.

    Attributes:
        ledger: Where forecasts are written and read.
        schema_version: Ledger schema version, part of every key.
        hits: Forecasts served from the ledger in this process.
        misses: Forecasts computed and recorded in this process.
    """

    ledger: Ledger
    schema_version: int = 1
    hits: int = 0
    misses: int = 0

    # -- keys -------------------------------------------------------------- #

    def key(
        self, forecaster: Forecaster, task: ChronosTask, snapshot_hash: str
    ) -> str:
        identity = dict(forecaster.cache_identity())
        return forecast_cache_key(
            checkpoint_revision=str(identity.get("revision") or identity["forecaster"]),
            package_version=stable_hash(identity),
            schema_version=self.schema_version,
            symbol=task.symbol,
            origin_session=task.origin_session.isoformat(),
            context_length=task.context_length,
            prediction_length=task.prediction_length,
            source_snapshot_hash=snapshot_hash,
            channel_set=task.channel_names,
            cross_learning=bool(identity.get("cross_learning", False)),
        )

    # -- read and write ---------------------------------------------------- #

    def get(self, key: str) -> ForecastResult | None:
        row = self.ledger.read_forecast(key)
        if row is None:
            return None
        diagnostics = json.loads(row["diagnostics"]) if row["diagnostics"] else {}
        detail = str(diagnostics.pop("detail", ""))
        return ForecastResult(
            symbol=str(row["symbol"]),
            origin_session=dt.date.fromisoformat(str(row["origin_session"])[:10]),
            quantile_levels=tuple(float(level) for level in json.loads(row["quantile_levels"])),
            paths=np.asarray(json.loads(row["quantiles"]), dtype=float),
            terminal_indices=tuple(int(index) for index in json.loads(row["terminal_indices"])),
            status=ForecastStatus(str(row["status"])),
            diagnostics={name: float(value) for name, value in diagnostics.items()},
            detail=detail,
        )

    def put(
        self,
        key: str,
        result: ForecastResult,
        task: ChronosTask,
        *,
        variant: str,
        forecaster: Forecaster,
        snapshot_hash: str,
        generated_at: dt.datetime,
    ) -> None:
        identity = forecaster.cache_identity()
        diagnostics: dict[str, Any] = dict(result.diagnostics)
        if result.detail:
            diagnostics["detail"] = result.detail
        self.ledger.record_forecast(
            cache_key=key,
            symbol=task.symbol,
            origin_session=task.origin_session,
            variant=variant,
            checkpoint_id=str(identity.get("model_id") or identity["forecaster"]),
            checkpoint_revision=str(identity.get("revision") or identity["forecaster"]),
            context_length=task.context_length,
            prediction_length=task.prediction_length,
            quantile_levels=list(result.quantile_levels),
            quantiles=np.asarray(result.paths, dtype=float).tolist(),
            terminal_indices=list(result.terminal_indices),
            horizon_sessions=sorted({bar.session.isoformat() for bar in task.horizon_bars}),
            origin_close=task.origin_close,
            status=result.status.value,
            diagnostics=diagnostics,
            snapshot_hash=snapshot_hash,
            generated_at=generated_at,
        )

    # -- the one entry point the pipeline uses ----------------------------- #

    def predict(
        self,
        forecaster: Forecaster,
        tasks: Sequence[ChronosTask],
        *,
        variant: str,
        snapshot_hashes: Mapping[str, str],
        now: dt.datetime | None = None,
    ) -> list[tuple[ForecastResult, str]]:
        """Forecasts for ``tasks`` in order, each with its cache key.

        Cached forecasts are returned as stored. Only the misses reach the
        forecaster, in a single call so that batching is unchanged, and every
        one of them is recorded before this method returns.
        """
        now = now or dt.datetime.now(dt.timezone.utc)
        keys = [self.key(forecaster, task, snapshot_hashes[task.symbol]) for task in tasks]
        found: dict[int, ForecastResult] = {}
        for position, key in enumerate(keys):
            cached = self.get(key)
            if cached is not None:
                found[position] = cached
        self.hits += len(found)

        missing = [position for position in range(len(tasks)) if position not in found]
        if missing:
            fresh = forecaster.predict([tasks[position] for position in missing])
            if len(fresh) != len(missing):
                raise RuntimeError(
                    f"forecaster returned {len(fresh)} results for {len(missing)} tasks"
                )
            with self.ledger.transaction():
                for position, result in zip(missing, fresh, strict=True):
                    self.put(
                        keys[position],
                        result,
                        tasks[position],
                        variant=variant,
                        forecaster=forecaster,
                        snapshot_hash=snapshot_hashes[tasks[position].symbol],
                        generated_at=now,
                    )
                    found[position] = result
            self.misses += len(missing)
        return [(found[position], keys[position]) for position in range(len(tasks))]
