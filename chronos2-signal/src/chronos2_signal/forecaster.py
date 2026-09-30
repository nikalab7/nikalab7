"""Adapter over the frozen Chronos-2 checkpoint.

The checkpoint is a fixed input, not a component of this system: 120M
parameters, encoder-only, direct quantile outputs, patch size and stride 16,
maximum context 8192. None of that is evidence of financial predictive power,
and nothing here fine-tunes it.

Implementation traps this adapter exists to avoid:

* The output the library calls ``mean`` is the **median**. It is stored as
  ``forecast_median`` and never as an expected profit.
* Batch size counts *series*, target and covariates together, not symbols. The
  budget here is therefore expressed in channels.
* ``cross_learning=True`` shares information across tasks, which makes a
  prediction depend on which other stocks happen to be in the same batch. It is
  off, and the cache key records that it was off.
* The DataFrame convenience path assumes a regular timestamp grid. Exchange
  bars are not regular -- overnight and weekend breaks are real -- so the
  array/dictionary interface is used with our explicit bar calendar.
* A crossed or non-finite quantile **blocks** the forecast. Sorting the values
  and carrying on would report an output the model did not produce.

The exact keyword mapping of ``Chronos2Pipeline.predict_quantiles`` is
introspected at load time rather than assumed. If the installed package does
not match, loading fails with the signature it actually found, which is the
compatibility problem the protocol requires resolving before any run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import importlib.metadata as importlib_metadata
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .features import ChronosTask

__all__ = [
    "ForecastError",
    "ForecastStatus",
    "ForecastResult",
    "Forecaster",
    "Chronos2Forecaster",
    "DeterministicStubForecaster",
    "batch_tasks_by_channels",
]


class ForecastError(RuntimeError):
    """Raised when a forecast cannot be produced or validated."""


class ForecastStatus(str, Enum):
    """Outcome of one forecast attempt."""

    OK = "ok"
    BLOCKED_CROSSED = "blocked_crossed_quantiles"
    BLOCKED_NONFINITE = "blocked_nonfinite"
    BLOCKED_SHAPE = "blocked_shape"
    FAILED = "failed"


@dataclass(frozen=True)
class ForecastResult:
    """Quantile paths for one task, in the rebased log target units.

    Attributes:
        symbol: Target symbol.
        origin_session: Origin the context ended on.
        quantile_levels: Requested levels, ascending.
        paths: ``(n_levels, prediction_length)`` matrix. Row order matches
            ``quantile_levels``.
        terminal_indices: Index of each holding session's terminal bar.
        status: Whether the forecast may be used.
        diagnostics: Values the monitor records (crossings, padding trimmed).
    """

    symbol: str
    origin_session: dt.date
    quantile_levels: tuple[float, ...]
    paths: np.ndarray
    terminal_indices: tuple[int, ...]
    status: ForecastStatus = ForecastStatus.OK
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status is ForecastStatus.OK

    @property
    def prediction_length(self) -> int:
        return int(self.paths.shape[1]) if self.paths.ndim == 2 else 0

    def level(self, quantile: float) -> np.ndarray:
        """The path for one requested quantile level."""
        try:
            row = self.quantile_levels.index(quantile)
        except ValueError as exc:
            raise ForecastError(
                f"{self.symbol}: quantile {quantile} was not requested; available "
                f"{self.quantile_levels}"
            ) from exc
        return self.paths[row]

    @property
    def forecast_median(self) -> np.ndarray:
        """The 0.50 quantile path.

        Named to make the misreading impossible: this is a median, not a mean,
        and not an expected profit.
        """
        return self.level(0.50)

    def as_mapping(self) -> dict[float, np.ndarray]:
        return {level: self.paths[row] for row, level in enumerate(self.quantile_levels)}

    def terminal_median(self, session_offset: int) -> float:
        index = self.terminal_indices[session_offset - 1]
        return float(self.forecast_median[index])


def validate_quantile_paths(
    symbol: str,
    origin_session: dt.date,
    levels: Sequence[float],
    raw: np.ndarray,
    prediction_length: int,
    terminal_indices: Sequence[int],
) -> ForecastResult:
    """Shape, padding, finiteness and monotonicity checks on a raw output.

    Padding beyond the requested horizon is discarded -- the patch stride can
    make the model emit more steps than were asked for. Crossed quantiles block
    the forecast instead of being repaired.
    """
    levels = tuple(float(level) for level in levels)
    array = np.asarray(raw, dtype=float)
    diagnostics: dict[str, float] = {}

    if array.ndim != 2 or array.shape[0] != len(levels):
        return ForecastResult(
            symbol=symbol,
            origin_session=origin_session,
            quantile_levels=levels,
            paths=array,
            terminal_indices=tuple(terminal_indices),
            status=ForecastStatus.BLOCKED_SHAPE,
            detail=(
                f"expected ({len(levels)}, >= {prediction_length}), got {array.shape}"
            ),
        )
    if array.shape[1] < prediction_length:
        return ForecastResult(
            symbol=symbol,
            origin_session=origin_session,
            quantile_levels=levels,
            paths=array,
            terminal_indices=tuple(terminal_indices),
            status=ForecastStatus.BLOCKED_SHAPE,
            detail=(
                f"model returned {array.shape[1]} steps, fewer than the "
                f"{prediction_length} calendar-derived bars in the horizon"
            ),
        )

    diagnostics["padding_trimmed"] = float(array.shape[1] - prediction_length)
    trimmed = array[:, :prediction_length]

    if not np.isfinite(trimmed).all():
        return ForecastResult(
            symbol=symbol,
            origin_session=origin_session,
            quantile_levels=levels,
            paths=trimmed,
            terminal_indices=tuple(terminal_indices),
            status=ForecastStatus.BLOCKED_NONFINITE,
            diagnostics=diagnostics,
            detail=f"{int((~np.isfinite(trimmed)).sum())} non-finite quantile values",
        )

    crossings = int((np.diff(trimmed, axis=0) < 0.0).sum())
    diagnostics["quantile_crossings"] = float(crossings)
    if crossings:
        return ForecastResult(
            symbol=symbol,
            origin_session=origin_session,
            quantile_levels=levels,
            paths=trimmed,
            terminal_indices=tuple(terminal_indices),
            status=ForecastStatus.BLOCKED_CROSSED,
            diagnostics=diagnostics,
            detail=(
                f"{crossings} crossed quantile value(s); the forecast is blocked rather "
                "than silently sorted"
            ),
        )

    for index in terminal_indices:
        if not 0 <= index < prediction_length:
            return ForecastResult(
                symbol=symbol,
                origin_session=origin_session,
                quantile_levels=levels,
                paths=trimmed,
                terminal_indices=tuple(terminal_indices),
                status=ForecastStatus.BLOCKED_SHAPE,
                diagnostics=diagnostics,
                detail=f"terminal index {index} outside horizon {prediction_length}",
            )

    return ForecastResult(
        symbol=symbol,
        origin_session=origin_session,
        quantile_levels=levels,
        paths=trimmed,
        terminal_indices=tuple(terminal_indices),
        status=ForecastStatus.OK,
        diagnostics=diagnostics,
    )


def batch_tasks_by_channels(
    tasks: Sequence[ChronosTask], budget: int
) -> list[list[ChronosTask]]:
    """Group tasks so that each batch fits a channel budget.

    The budget counts every series in the batch -- target and covariates alike --
    because that is what the library counts. A task whose own channel count
    exceeds the budget still forms a batch of one: splitting a task's channels
    would change the task.
    """
    if budget < 1:
        raise ForecastError("channel budget must be positive")
    batches: list[list[ChronosTask]] = []
    current: list[ChronosTask] = []
    used = 0
    for task in tasks:
        channels = task.n_channels
        if current and used + channels > budget:
            batches.append(current)
            current, used = [], 0
        current.append(task)
        used += channels
    if current:
        batches.append(current)
    return batches


class Forecaster(Protocol):
    """Anything that turns tasks into validated quantile paths."""

    def predict(self, tasks: Sequence[ChronosTask]) -> list[ForecastResult]:  # pragma: no cover
        ...

    def provenance(self) -> dict[str, Any]:  # pragma: no cover
        ...

    def cache_identity(self) -> dict[str, Any]:  # pragma: no cover
        """Everything about the forecaster that can change its numbers.

        Excludes the batch budget and the device on purpose: neither may change
        a prediction beyond numerical tolerance, and a cache key that included
        them would hide a violation of that rule instead of exposing it.
        """
        ...


#: Version of the mapping from a task to the library's input dictionary. Part
#: of the cache identity, so forecasts produced under an earlier mapping are
#: never served as if they came from the current one.
CHRONOS_INPUT_SCHEMA = 2


def _inference_context() -> Any:
    """``torch.inference_mode()`` when torch is present, else a no-op.

    The real pipeline needs torch anyway; the fallback only exists so that the
    adapter's own logic can be tested against a fake pipeline without it.
    """
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    return torch.inference_mode()


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "absent"


@dataclass
class Chronos2Forecaster:
    """The frozen checkpoint, loaded once and used in inference mode only.

    Args:
        model_id: Hub identifier, ``amazon/chronos-2``.
        revision: Pinned 40-character commit. A tag is not acceptable: a tag can
            move, and then a stored forecast no longer describes its inputs.
        expected_sha256: Recorded hash of the checkpoint weights. When the file
            is present locally it is verified; a mismatch raises rather than
            silently switching weights.
        batch_size_channels: Channel budget per batch.
        device: ``"cpu"``, ``"cuda"`` or ``None`` to select automatically.
    """

    model_id: str
    revision: str
    quantile_levels: tuple[float, ...]
    batch_size_channels: int = 128
    device: str | None = None
    dtype: str = "float32"
    cross_learning: bool = False
    expected_sha256: str | None = None
    _pipeline: Any = field(default=None, init=False, repr=False)
    _api: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    is_frozen_checkpoint = True

    # -- loading ----------------------------------------------------------- #

    def load(self) -> None:
        """Load the pipeline and verify it matches the expected interface."""
        if self._pipeline is not None:
            return
        if self.cross_learning:
            raise ForecastError(
                "cross_learning must be False: sharing across stock tasks makes a "
                "prediction depend on batch composition"
            )
        try:
            from chronos import Chronos2Pipeline  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ForecastError(
                "chronos-forecasting is not installed. Install the 'model' extra to run "
                "the compatibility smoke test; the research pipeline refuses to "
                "fabricate forecasts without it."
            ) from exc

        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            revision=self.revision,
            device_map=device,
            torch_dtype=getattr(torch, self.dtype),
        )
        inner = getattr(pipeline, "model", None)
        if inner is not None and hasattr(inner, "eval"):
            inner.eval()
        for parameter in getattr(inner, "parameters", lambda: [])():
            parameter.requires_grad_(False)

        self._pipeline = pipeline
        self._api = self._introspect(pipeline)
        self.device = device
        self._verify_weights()

    @staticmethod
    def _introspect(pipeline: Any) -> dict[str, Any]:
        """Record the real ``predict_quantiles`` signature.

        The protocol requires a compatibility smoke test before any run. This
        makes its result a stored fact rather than an assumption: the parameter
        names are read off the installed package.
        """
        method = getattr(pipeline, "predict_quantiles", None)
        if method is None:
            raise ForecastError(
                "installed pipeline has no predict_quantiles method; resolve the "
                "compatibility problem rather than switching to another entry point"
            )
        signature = inspect.signature(method)
        parameters = tuple(signature.parameters)
        required = {"quantile_levels"}
        missing = required - set(parameters)
        if missing:
            raise ForecastError(
                f"predict_quantiles is missing expected parameters {sorted(missing)}; "
                f"found {parameters}. Resolve compatibility before running."
            )
        return {
            "signature": str(signature),
            "parameters": parameters,
            "supports_cross_learning": "cross_learning" in parameters,
            "prediction_length_parameter": (
                "prediction_length" if "prediction_length" in parameters else None
            ),
        }

    def _verify_weights(self) -> None:
        """Verify the checkpoint hash when the weight file can be located."""
        if not self.expected_sha256:
            return
        from pathlib import Path

        from .provenance import sha256_file

        candidates: list[Path] = []
        config = getattr(getattr(self._pipeline, "model", None), "config", None)
        name_or_path = getattr(config, "_name_or_path", None)
        if name_or_path:
            candidates.append(Path(str(name_or_path)))
        for root in candidates:
            weight_file = root / "model.safetensors"
            if weight_file.is_file():
                digest = sha256_file(weight_file)
                if digest != self.expected_sha256:
                    raise ForecastError(
                        f"checkpoint hash mismatch at {weight_file}: expected "
                        f"{self.expected_sha256}, found {digest}. Stop and resolve; do "
                        "not silently switch weights."
                    )
                self._api["weights_verified"] = str(weight_file)
                return
        self._api["weights_verified"] = "not_located"

    def describe_pipeline_api(self) -> dict[str, Any]:
        """What the installed package actually exposes, for the run manifest."""
        self.load()
        return dict(self._api)

    # -- inference --------------------------------------------------------- #

    def predict(self, tasks: Sequence[ChronosTask]) -> list[ForecastResult]:
        self.load()
        results: list[ForecastResult] = []
        for batch in batch_tasks_by_channels(tasks, self.batch_size_channels):
            with _inference_context():
                results.extend(self._predict_batch(batch))
        return results

    def _predict_batch(self, batch: Sequence[ChronosTask]) -> list[ForecastResult]:
        payload = [self._build_pipeline_input(task) for task in batch]
        lengths = {task.prediction_length for task in batch}
        if len(lengths) != 1:
            # Different horizons in one call would make the padding ambiguous.
            # Early closes make this real, so such tasks are split, not padded.
            return [
                result
                for task in batch
                for result in self._predict_batch([task])
            ]
        prediction_length = lengths.pop()

        kwargs: dict[str, Any] = {"quantile_levels": list(self.quantile_levels)}
        if self._api.get("prediction_length_parameter"):
            kwargs["prediction_length"] = prediction_length
        if self._api.get("supports_cross_learning"):
            kwargs["cross_learning"] = self.cross_learning

        try:
            raw = self._pipeline.predict_quantiles(payload, **kwargs)
        except Exception as exc:
            return [
                ForecastResult(
                    symbol=task.symbol,
                    origin_session=task.origin_session,
                    quantile_levels=self.quantile_levels,
                    paths=np.zeros((0, 0)),
                    terminal_indices=(),
                    status=ForecastStatus.FAILED,
                    detail=f"pipeline raised {type(exc).__name__}: {exc}",
                )
                for task in batch
            ]

        per_task = self._to_arrays(raw, len(batch))
        results: list[ForecastResult] = []
        for task, array in zip(batch, per_task, strict=True):
            terminal = _terminal_indices(task)
            oriented, problem = _orient_levels_by_horizon(array, len(self.quantile_levels))
            if problem is not None:
                results.append(
                    ForecastResult(
                        symbol=task.symbol,
                        origin_session=task.origin_session,
                        quantile_levels=self.quantile_levels,
                        paths=np.asarray(array, dtype=float),
                        terminal_indices=terminal,
                        status=ForecastStatus.BLOCKED_SHAPE,
                        detail=problem,
                    )
                )
                continue
            results.append(
                validate_quantile_paths(
                    task.symbol,
                    task.origin_session,
                    self.quantile_levels,
                    oriented,
                    prediction_length,
                    terminal,
                )
            )
        return results

    def _build_pipeline_input(self, task: ChronosTask) -> dict[str, Any]:
        """One task in the array/dictionary interface.

        Kept deliberately small and in one place: it is the single point that
        depends on the library's input contract, and the compatibility smoke
        test exercises exactly this mapping.

        **Unverified against the installed package** -- the model cannot be
        downloaded in the environment this was written in. The mapping follows
        the documented pattern as understood: a covariate known in the future
        appears under the *same name* in ``past_covariates`` (its history) and in
        ``future_covariates`` (its values over the horizon); a covariate only in
        ``past_covariates`` is past-only. :meth:`channel_usage_check` exists to
        confirm, on the real checkpoint, that every channel actually reaches the
        model rather than being silently dropped.
        """
        from .calendar_spec import CALENDAR_CHANNEL_NAMES

        past: dict[str, np.ndarray] = {}
        for column, name in enumerate(task.channel_names[1:]):
            past[name] = np.asarray(task.past_covariates()[:, column], dtype=np.float32)
        for column, name in enumerate(CALENDAR_CHANNEL_NAMES):
            past[name] = np.asarray(task.past_calendar[:, column], dtype=np.float32)
        return {
            "target": np.asarray(task.target, dtype=np.float32),
            "past_covariates": past,
            "future_covariates": {
                name: np.asarray(task.future_calendar[:, column], dtype=np.float32)
                for column, name in enumerate(CALENDAR_CHANNEL_NAMES)
            },
        }

    @staticmethod
    def _to_arrays(raw: Any, expected: int) -> list[np.ndarray]:
        """Split the library's return value into one array per task.

        ``predict_quantiles`` returns ``(quantiles, mean)``. The mean is the
        library's point forecast, which is the median and is not used here.
        Unpacking that pair explicitly matters: treating it as a generic
        sequence would read a two-task batch as "task one = quantiles, task two
        = mean", silently.
        """
        if isinstance(raw, tuple) and len(raw) == 2:
            raw = raw[0]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        if isinstance(raw, (list, tuple)):
            items = [
                item.detach().cpu().numpy() if hasattr(item, "detach") else np.asarray(item)
                for item in raw
            ]
            if len(items) != expected:
                raise ForecastError(
                    f"pipeline returned {len(items)} per-task outputs for {expected} tasks"
                )
            return items
        array = np.asarray(raw)
        if array.ndim >= 3 and array.shape[0] == expected:
            return [np.asarray(item) for item in array]
        if array.ndim == 2 and expected == 1:
            return [array]
        raise ForecastError(
            f"cannot interpret pipeline output of shape {array.shape} for {expected} tasks"
        )

    def cache_identity(self) -> dict[str, Any]:
        return {
            "forecaster": "chronos2",
            "model_id": self.model_id,
            "revision": self.revision,
            "quantile_levels": list(self.quantile_levels),
            "cross_learning": self.cross_learning,
            "dtype": self.dtype,
            "package_version": _package_version("chronos-forecasting"),
            "input_schema": CHRONOS_INPUT_SCHEMA,
        }

    def channel_usage_check(
        self, task: ChronosTask, *, tolerance: float = 1e-7
    ) -> dict[str, Any]:
        """Confirm on the real checkpoint that each channel group reaches the model.

        Each group is perturbed in isolation -- the weekday channels of the
        calendar history, the same channels over the horizon, and the past-only
        covariates -- and the forecast is compared with the unperturbed one. A
        group whose perturbation leaves the output unchanged is not being
        consumed, which is how a wrong input mapping shows up: not as an error,
        but as a model quietly forecasting from less than it was given.

        This is a compatibility check with no performance meaning, and it must
        not be used to tune anything.
        """
        baseline = self.predict([task])[0]
        report: dict[str, Any] = {"baseline_status": baseline.status.value}
        if not baseline.usable:
            report["detail"] = baseline.detail
            return report

        def changed(candidate: ChronosTask) -> bool:
            result = self.predict([candidate])[0]
            if not result.usable or result.paths.shape != baseline.paths.shape:
                return True
            return not np.allclose(result.paths, baseline.paths, rtol=0.0, atol=tolerance)

        past_calendar = task.past_calendar.copy()
        past_calendar[:, 3:] = -past_calendar[:, 3:]
        report["past_calendar_used"] = changed(
            dataclasses.replace(task, past_calendar=past_calendar)
        )

        future_calendar = task.future_calendar.copy()
        future_calendar[:, 3:] = -future_calendar[:, 3:]
        report["future_calendar_used"] = changed(
            dataclasses.replace(task, future_calendar=future_calendar)
        )

        if task.past.shape[1] > 1:
            past = task.past.copy()
            finite = np.isfinite(past[:, 1:])
            past[:, 1:] = np.where(finite, -past[:, 1:], past[:, 1:])
            report["past_covariates_used"] = changed(dataclasses.replace(task, past=past))
        return report

    def provenance(self) -> dict[str, Any]:
        return {
            "forecaster": "chronos2",
            "is_frozen_checkpoint": True,
            "model_id": self.model_id,
            "revision": self.revision,
            "quantile_levels": list(self.quantile_levels),
            "cross_learning": self.cross_learning,
            "batch_size_channels": self.batch_size_channels,
            "device": self.device,
            "dtype": self.dtype,
            "pipeline_api": dict(self._api),
        }


@dataclass
class DeterministicStubForecaster:
    """A fixed arithmetic rule standing in for the checkpoint.

    This is **not a model**. It exists so the pipeline, the leakage invariants
    and the ledger can be tested without the 120M-parameter checkpoint or a
    network. Its ``provenance`` marks it as not a frozen checkpoint, and the
    evaluation layer refuses to attach performance claims to it.

    The rule: the median path continues the context's recent mean bar move, and
    the quantile band widens with the square root of the horizon step, scaled by
    the context's realised bar volatility. Deterministic, ordered by
    construction, and with no predictive content whatsoever.
    """

    quantile_levels: tuple[float, ...]
    batch_size_channels: int = 128
    momentum_bars: int = 14
    band_scale: float = 1.0

    is_frozen_checkpoint = False

    def predict(self, tasks: Sequence[ChronosTask]) -> list[ForecastResult]:
        results: list[ForecastResult] = []
        # Batched exactly like the real adapter so the batching invariant is
        # exercised on the same code path.
        for batch in batch_tasks_by_channels(tasks, self.batch_size_channels):
            for task in batch:
                results.append(self._predict_one(task))
        return results

    def _predict_one(self, task: ChronosTask) -> ForecastResult:
        target = np.asarray(task.target, dtype=float)
        finite = target[np.isfinite(target)]
        if finite.size < 2:
            return ForecastResult(
                symbol=task.symbol,
                origin_session=task.origin_session,
                quantile_levels=self.quantile_levels,
                paths=np.zeros((len(self.quantile_levels), 0)),
                terminal_indices=_terminal_indices(task),
                status=ForecastStatus.BLOCKED_NONFINITE,
                detail="context has fewer than two finite target values",
            )
        steps = np.diff(finite)
        window = steps[-self.momentum_bars :]
        drift = float(np.mean(window)) if window.size else 0.0
        scale = float(np.std(steps, ddof=1)) if steps.size > 1 else 0.1
        scale = max(scale, 1e-6) * self.band_scale

        horizon = task.prediction_length
        offsets = np.arange(1, horizon + 1, dtype=float)
        median = drift * offsets
        spread = scale * np.sqrt(offsets)

        # A fixed standard-normal-like ladder keeps the band ordered by
        # construction, which is what makes this a plumbing fixture.
        ladder = {0.10: -1.2816, 0.25: -0.6745, 0.50: 0.0, 0.75: 0.6745, 0.90: 1.2816}
        rows = []
        for level in self.quantile_levels:
            multiplier = ladder.get(level)
            if multiplier is None:
                multiplier = float(np.sign(level - 0.5)) * abs(level - 0.5) * 4.0
            rows.append(median + multiplier * spread)
        return validate_quantile_paths(
            task.symbol,
            task.origin_session,
            self.quantile_levels,
            np.vstack(rows),
            horizon,
            _terminal_indices(task),
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "forecaster": "deterministic_stub",
            "is_frozen_checkpoint": False,
            "warning": (
                "arithmetic fixture with no predictive content; results produced with "
                "it are plumbing checks, not performance"
            ),
            "quantile_levels": list(self.quantile_levels),
            "momentum_bars": self.momentum_bars,
            "band_scale": self.band_scale,
        }

    def cache_identity(self) -> dict[str, Any]:
        return {
            "forecaster": "deterministic_stub",
            "quantile_levels": list(self.quantile_levels),
            "momentum_bars": self.momentum_bars,
            "band_scale": self.band_scale,
        }


def _orient_levels_by_horizon(
    array: Any, n_levels: int
) -> tuple[np.ndarray, str | None]:
    """Bring one task's output into ``(levels, horizon)`` order.

    Libraries differ on whether the quantile axis comes first or last, and may
    add a leading axis for the number of target series. The orientation is
    decided from the known number of requested levels rather than assumed.
    When both axes have that length the orientation cannot be decided, and the
    forecast is blocked rather than guessed.
    """
    oriented = np.asarray(array, dtype=float)
    while oriented.ndim > 2 and oriented.shape[0] == 1:
        oriented = oriented[0]
    if oriented.ndim != 2:
        return oriented, None  # validate_quantile_paths reports the shape
    rows, columns = oriented.shape
    if rows == n_levels and columns != n_levels:
        return oriented, None
    if columns == n_levels and rows != n_levels:
        return oriented.T, None
    if rows == columns == n_levels:
        return oriented, (
            f"output is {rows}x{columns} with {n_levels} requested levels: the "
            "quantile axis cannot be told apart from the horizon axis"
        )
    return oriented, None


def _terminal_indices(task: ChronosTask) -> tuple[int, ...]:
    """Terminal bar index of each holding session, from the task's horizon."""
    indices: list[int] = []
    for position, bar in enumerate(task.horizon_bars):
        is_last = (
            position + 1 == len(task.horizon_bars)
            or task.horizon_bars[position + 1].session != bar.session
        )
        if is_last:
            indices.append(position)
    return tuple(indices)
