"""The small decision model: a regularised return head and a probability head.

With roughly two years of hourly history -- and far less after the frozen
checkpoint existed -- independent market dates are scarce. So the second stage
starts regularised and linear. A flexible nonlinear model is a challenger for a
later registered version, not an automatic upgrade.

Both heads train on exactly the same origin rows:

* **Return head.** Ridge, alpha 10, intercept on. Target ``R_net / sigma_2d``;
  the prediction is multiplied back by the current ``sigma_2d`` to give an
  estimated net return. The volatility scale is a feature normalisation, not a
  claim that returns are Gaussian.
* **Probability head.** Logistic regression, L2, ``C=1``, lbfgs, 2000
  iterations, no class rebalancing, on ``1[R_net > 0]``. Its scores are then
  Platt-calibrated on a **later, disjoint** block of dates. Random-split
  calibration would leak across the time boundary.

Preprocessing statistics -- medians and standardisation -- are fitted on each
head's own training window and never on the full dataset. Only the optional
sector-breadth feature may be imputed; any other missing required feature
disqualifies the row rather than being filled in.
"""

from __future__ import annotations

import datetime as dt
import inspect
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from .actions import HoldingCosts, reprice_net_return
from .config import DesignConfig
from .features import (
    BREADTH_FEATURE_NAME,
    BREADTH_MISSING_FLAG,
    NON_FORECAST_FEATURE_NAMES,
    PREPROCESSING_COLUMNS,
)

__all__ = [
    "DecisionError",
    "OriginRow",
    "Preprocessor",
    "PlattCalibrator",
    "DecisionModel",
    "DecisionEstimate",
    "FitReport",
    "fit_decision_model",
    "column_set_for_variant",
    "date_normalised_weights",
]

#: Columns that may be median-imputed. Everything else must be valid.
IMPUTABLE_COLUMNS = frozenset({BREADTH_FEATURE_NAME})


class DecisionError(RuntimeError):
    """Raised on an invalid training set, feature row or fitted artefact."""


def _l2_logistic(*, C: float, max_iter: int) -> LogisticRegression:
    """An explicitly L2-penalised logistic regression, across sklearn versions.

    scikit-learn 1.8 deprecated the ``penalty`` keyword in favour of
    ``l1_ratio``. Rather than guessing from a version string, the constructor
    signature is inspected so the penalty stays *explicit* on both old and new
    installations. Relying on the library default would leave the registered
    "L2 regularisation" as an assumption about someone else's default.
    """
    parameters = inspect.signature(LogisticRegression.__init__).parameters
    common = {
        "C": C,
        "solver": "lbfgs",
        "max_iter": max_iter,
        # No class rebalancing: the registered probability head is fitted on
        # the observed base rate.
        "class_weight": None,
    }
    if "l1_ratio" in parameters:
        return LogisticRegression(l1_ratio=0.0, **common)
    return LogisticRegression(penalty="l2", **common)


@dataclass(frozen=True)
class OriginRow:
    """One (origin, symbol) training or scoring row.

    Attributes:
        origin_session: The completed signal session the features end on.
        symbol: Stock symbol.
        features: Values keyed by preprocessing column name.
        sigma_2d: Two-session volatility scale at this origin.
        r_net: Realised net return under base costs. ``None`` while unlabelled.
        label_available_at: When the outcome became observable. A row may only
            enter a fit whose training ends at or after this moment.
        eligible_count: Eligible symbols at this origin, recorded for audit.
    """

    origin_session: dt.date
    symbol: str
    features: Mapping[str, float]
    sigma_2d: float
    r_net: float | None = None
    label_available_at: dt.datetime | None = None
    eligible_count: int | None = None

    @property
    def labelled(self) -> bool:
        return self.r_net is not None and math.isfinite(self.r_net)

    def scaled_target(self) -> float:
        if self.r_net is None:
            raise DecisionError(f"{self.symbol} {self.origin_session}: row is unlabelled")
        if not math.isfinite(self.sigma_2d) or self.sigma_2d <= 0.0:
            raise DecisionError(
                f"{self.symbol} {self.origin_session}: sigma_2d must be positive"
            )
        return self.r_net / self.sigma_2d


def column_set_for_variant(variant: str) -> tuple[str, ...]:
    """Preprocessing columns for a registered variant.

    ``B0`` drops the six forecast features, leaving the fourteen non-forecast
    features and the breadth indicator: fifteen columns. Every Chronos variant
    uses all twenty-one.
    """
    if variant == "B0":
        return (*NON_FORECAST_FEATURE_NAMES, BREADTH_MISSING_FLAG)
    if variant in {"C128", "C256", "C512", "U256"}:
        return PREPROCESSING_COLUMNS
    raise DecisionError(
        f"unknown variant {variant!r}; the registered set is B0, C128, C256, C512, U256"
    )


def date_normalised_weights(
    sessions: Sequence[dt.date],
) -> np.ndarray:
    """Weights that give every distinct date a total weight of one.

    This stops a date with more eligible symbols from dominating the fit. It
    does **not** make correlated stocks statistically independent -- that is
    what the date-block bootstrap is for.

    The divisor is the number of rows actually present for the date in this
    fitting set, which is what makes each date's total exactly one. When rows
    were dropped for invalid features, the eligible count recorded on the row
    would no longer sum to one, so the present-row count is used and the
    eligible count is kept for audit only.
    """
    counts: dict[dt.date, int] = {}
    for session in sessions:
        counts[session] = counts.get(session, 0) + 1
    return np.asarray([1.0 / counts[session] for session in sessions], dtype=float)


@dataclass(frozen=True)
class Preprocessor:
    """Median imputation, standardisation and clipping, fitted on one window.

    Attributes:
        columns: Column order.
        medians: Training-window medians, used only for imputable columns.
        centers: Training-window means.
        scales: Training-window standard deviations, floored away from zero.
        clip_abs: Absolute bound applied to standardised values.
    """

    columns: tuple[str, ...]
    medians: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    clip_abs: float

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, float]],
        columns: Sequence[str],
        *,
        clip_abs: float,
    ) -> "Preprocessor":
        if not rows:
            raise DecisionError("cannot fit a preprocessor on an empty window")
        matrix = _raw_matrix(rows, columns)
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = _impute(matrix, columns, medians)
        centers = imputed.mean(axis=0)
        scales = imputed.std(axis=0, ddof=0)
        # A constant column carries no information; a unit scale keeps it at
        # zero after centring instead of producing a division by zero.
        scales = np.where(scales > 1e-12, scales, 1.0)
        return cls(
            columns=tuple(columns),
            medians=medians,
            centers=centers,
            scales=scales,
            clip_abs=float(clip_abs),
        )

    def transform(
        self, rows: Sequence[Mapping[str, float]]
    ) -> tuple[np.ndarray, float]:
        """Standardise and clip. Returns the matrix and the clipped fraction."""
        matrix = _raw_matrix(rows, self.columns)
        imputed = _impute(matrix, self.columns, self.medians)
        standardised = (imputed - self.centers) / self.scales
        clipped = np.clip(standardised, -self.clip_abs, self.clip_abs)
        touched = int(np.sum(clipped != standardised))
        fraction = touched / max(1, clipped.size)
        return clipped, fraction

    def transform_one(self, row: Mapping[str, float]) -> np.ndarray:
        matrix, _fraction = self.transform([row])
        return matrix[0]


def _raw_matrix(
    rows: Sequence[Mapping[str, float]], columns: Sequence[str]
) -> np.ndarray:
    matrix = np.empty((len(rows), len(columns)), dtype=float)
    for position, row in enumerate(rows):
        for column_index, name in enumerate(columns):
            if name not in row:
                raise DecisionError(f"feature row is missing required column {name!r}")
            matrix[position, column_index] = float(row[name])
    return matrix


def _impute(
    matrix: np.ndarray, columns: Sequence[str], medians: np.ndarray
) -> np.ndarray:
    """Fill the one optional column; reject a missing required value."""
    out = matrix.copy()
    for column_index, name in enumerate(columns):
        column = out[:, column_index]
        missing = ~np.isfinite(column)
        if not missing.any():
            continue
        if name not in IMPUTABLE_COLUMNS:
            raise DecisionError(
                f"required feature {name!r} is missing in {int(missing.sum())} row(s); "
                "such rows are skipped rather than silently imputed"
            )
        column[missing] = medians[column_index]
        out[:, column_index] = column
    return out


@dataclass(frozen=True)
class PlattCalibrator:
    """Sigmoid calibration ``p = 1 / (1 + exp(-(a*s + b)))``.

    Fitted on a later, disjoint block of dates. The scores it maps come from
    the probability head; the head itself is never refitted on the calibration
    block, which is what keeps the two sets of dates disjoint.
    """

    slope: float
    intercept: float
    n_rows: int
    n_dates: int
    first_session: dt.date
    last_session: dt.date

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        sessions: Sequence[dt.date],
        weights: np.ndarray | None = None,
    ) -> "PlattCalibrator":
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels, dtype=int)
        if scores.shape[0] != labels.size:
            raise DecisionError("calibration scores and labels must align")
        if labels.size == 0:
            raise DecisionError("cannot calibrate on an empty block")
        unique = set(np.unique(labels).tolist())
        if unique == {0} or unique == {1}:
            raise DecisionError(
                "calibration block contains a single class; extend the block rather "
                "than reusing training dates"
            )
        # A very large C approximates the unregularised Platt fit while keeping
        # the dependency surface to one estimator.
        model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
        model.fit(scores, labels, sample_weight=weights)
        return cls(
            slope=float(model.coef_[0][0]),
            intercept=float(model.intercept_[0]),
            n_rows=int(labels.size),
            n_dates=len(set(sessions)),
            first_session=min(sessions),
            last_session=max(sessions),
        )

    def probability(self, score: float | np.ndarray) -> np.ndarray:
        value = np.asarray(score, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.slope * value + self.intercept)))


@dataclass(frozen=True)
class DecisionEstimate:
    """The two head outputs for one scoring row."""

    symbol: str
    origin_session: dt.date
    estimated_net_return: float
    estimated_net_return_stress: float
    calibrated_probability: float
    raw_probability_score: float
    sigma_2d: float
    scaled_return_prediction: float

    def as_payload(self) -> dict[str, float | str]:
        return {
            "symbol": self.symbol,
            "origin_session": self.origin_session.isoformat(),
            # Named so it cannot be read as a promise.
            "estimated_net_return_model_estimate": self.estimated_net_return,
            "estimated_net_return_stress_cost": self.estimated_net_return_stress,
            "calibrated_probability_model_estimate": self.calibrated_probability,
            "sigma_2d": self.sigma_2d,
        }


@dataclass(frozen=True)
class FitReport:
    """Everything a fit manifest needs to record about one fit."""

    variant: str
    columns: tuple[str, ...]
    fit_first_session: dt.date
    fit_last_session: dt.date
    fit_rows: int
    fit_dates: int
    calibration_first_session: dt.date
    calibration_last_session: dt.date
    calibration_rows: int
    calibration_dates: int
    clip_fraction: float
    positive_label_rate: float
    label_timestamps_checked: bool
    sample_weight_total: float = 0.0
    dropped_rows: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "columns": list(self.columns),
            "fit_first_session": self.fit_first_session.isoformat(),
            "fit_last_session": self.fit_last_session.isoformat(),
            "fit_rows": self.fit_rows,
            "fit_dates": self.fit_dates,
            "calibration_first_session": self.calibration_first_session.isoformat(),
            "calibration_last_session": self.calibration_last_session.isoformat(),
            "calibration_rows": self.calibration_rows,
            "calibration_dates": self.calibration_dates,
            "clip_fraction": self.clip_fraction,
            "positive_label_rate": self.positive_label_rate,
            "label_timestamps_checked": self.label_timestamps_checked,
            "sample_weight_total": self.sample_weight_total,
            "dropped_rows": self.dropped_rows,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DecisionModel:
    """A fitted return head, probability head and calibrator.

    The model never sees the future opening price: entry happens at the D1 open,
    which is unknown when this model scores an origin.
    """

    variant: str
    columns: tuple[str, ...]
    preprocessor: Preprocessor
    ridge: Ridge
    logistic: LogisticRegression
    calibrator: PlattCalibrator
    base_costs: HoldingCosts
    stress_costs: HoldingCosts
    report: FitReport
    training_base_rate: float

    def estimate(self, row: OriginRow) -> DecisionEstimate:
        """Score one origin row.

        Raises:
            DecisionError: If ``sigma_2d`` is not usable or a required feature
                is missing. A row that cannot be scored is skipped, never
                scored with a filled-in value.
        """
        if not math.isfinite(row.sigma_2d) or row.sigma_2d <= 0.0:
            raise DecisionError(
                f"{row.symbol} {row.origin_session}: sigma_2d must be positive to "
                "rescale the return estimate"
            )
        design = self.preprocessor.transform_one(row.features).reshape(1, -1)
        scaled = float(self.ridge.predict(design)[0])
        estimated = scaled * row.sigma_2d
        score = float(self.logistic.decision_function(design)[0])
        probability = float(self.calibrator.probability(score))
        stressed = reprice_net_return(
            estimated, base=self.base_costs, target=self.stress_costs
        )
        return DecisionEstimate(
            symbol=row.symbol,
            origin_session=row.origin_session,
            estimated_net_return=estimated,
            estimated_net_return_stress=stressed,
            calibrated_probability=probability,
            raw_probability_score=score,
            sigma_2d=row.sigma_2d,
            scaled_return_prediction=scaled,
        )

    def estimate_many(self, rows: Iterable[OriginRow]) -> list[DecisionEstimate]:
        return [self.estimate(row) for row in rows]

    def artifact(self) -> dict[str, Any]:
        """Serialisable description of the fitted artefact for the manifest."""
        return {
            "variant": self.variant,
            "columns": list(self.columns),
            "ridge_coefficients": self.ridge.coef_.tolist(),
            "ridge_intercept": float(self.ridge.intercept_),
            "logistic_coefficients": self.logistic.coef_.tolist(),
            "logistic_intercept": self.logistic.intercept_.tolist(),
            "calibrator": {
                "slope": self.calibrator.slope,
                "intercept": self.calibrator.intercept,
                "n_rows": self.calibrator.n_rows,
                "n_dates": self.calibrator.n_dates,
            },
            "preprocessor": {
                "medians": self.preprocessor.medians.tolist(),
                "centers": self.preprocessor.centers.tolist(),
                "scales": self.preprocessor.scales.tolist(),
                "clip_abs": self.preprocessor.clip_abs,
            },
            "training_base_rate": self.training_base_rate,
            "report": self.report.to_dict(),
        }


def fit_decision_model(
    *,
    variant: str,
    fit_rows: Sequence[OriginRow],
    calibration_rows: Sequence[OriginRow],
    config: DesignConfig,
    fit_deadline: dt.datetime | None = None,
    calibration_deadline: dt.datetime | None = None,
) -> DecisionModel:
    """Fit both heads and the calibrator under the registered settings.

    Args:
        fit_rows: Labelled rows of the fitting block.
        calibration_rows: Labelled rows of the later, disjoint calibration
            block.
        fit_deadline: When the fitting stage runs. Any row whose label was not
            yet observable then is dropped, and the drop is counted. This
            timestamp check overrides the fixed purge length.
        calibration_deadline: The same check for the calibration stage.

    Raises:
        DecisionError: If a block is empty, the blocks overlap in time, or a
            block carries a single class.
    """
    columns = column_set_for_variant(variant)
    clip_abs = config.decision.standardized_feature_clip_abs

    fit_kept, fit_dropped = _usable_rows(fit_rows, fit_deadline)
    calibration_kept, calibration_dropped = _usable_rows(
        calibration_rows, calibration_deadline
    )
    if not fit_kept:
        raise DecisionError(f"{variant}: no usable labelled rows in the fitting block")
    if not calibration_kept:
        raise DecisionError(
            f"{variant}: no usable labelled rows in the calibration block"
        )

    fit_sessions = [row.origin_session for row in fit_kept]
    calibration_sessions = [row.origin_session for row in calibration_kept]
    if max(fit_sessions) >= min(calibration_sessions):
        raise DecisionError(
            f"{variant}: calibration block starts on "
            f"{min(calibration_sessions).isoformat()}, at or before the last fitting "
            f"session {max(fit_sessions).isoformat()}; the blocks must be disjoint and "
            "purged"
        )
    overlap = set(fit_sessions) & set(calibration_sessions)
    if overlap:
        raise DecisionError(f"{variant}: blocks share {len(overlap)} date(s)")

    feature_rows = [dict(row.features) for row in fit_kept]
    preprocessor = Preprocessor.fit(feature_rows, columns, clip_abs=clip_abs)
    design, clip_fraction = preprocessor.transform(feature_rows)
    weights = date_normalised_weights(fit_sessions)

    scaled_targets = np.asarray([row.scaled_target() for row in fit_kept], dtype=float)
    labels = np.asarray(
        [1 if (row.r_net or 0.0) > 0.0 else 0 for row in fit_kept], dtype=int
    )
    if set(np.unique(labels).tolist()) in ({0}, {1}):
        raise DecisionError(
            f"{variant}: fitting block has a single outcome class; extend the block"
        )

    ridge = Ridge(alpha=config.decision.ridge_alpha, fit_intercept=True)
    ridge.fit(design, scaled_targets, sample_weight=weights)

    logistic = _l2_logistic(
        C=config.decision.logistic_C,
        max_iter=config.decision.logistic_max_iter,
    )
    logistic.fit(design, labels, sample_weight=weights)

    calibration_features = [dict(row.features) for row in calibration_kept]
    calibration_design, _fraction = preprocessor.transform(calibration_features)
    calibration_scores = logistic.decision_function(calibration_design)
    calibration_labels = np.asarray(
        [1 if (row.r_net or 0.0) > 0.0 else 0 for row in calibration_kept], dtype=int
    )
    calibrator = PlattCalibrator.fit(
        calibration_scores,
        calibration_labels,
        calibration_sessions,
        date_normalised_weights(calibration_sessions),
    )

    base_costs = HoldingCosts(
        buy_slippage=config.execution.slippage_fraction("base"),
        sell_slippage=config.execution.slippage_fraction("base"),
        explicit_fee_fraction=config.execution.explicit_fee_fraction,
    )
    stress_costs = HoldingCosts(
        buy_slippage=config.execution.slippage_fraction("stress"),
        sell_slippage=config.execution.slippage_fraction("stress"),
        explicit_fee_fraction=config.execution.explicit_fee_fraction,
    )

    notes: list[str] = [
        # Worth recording explicitly: date-normalised weights make the total
        # weight equal the number of dates, not the number of rows, so a given
        # alpha or C shrinks harder than the same value would on an unweighted
        # fit. That is a consequence of the registered design, not a defect,
        # but it should be visible in the manifest rather than inferred.
        f"total sample weight {float(weights.sum()):.1f} over {len(fit_kept)} rows: "
        "regularisation is stronger than an unweighted fit at the same alpha/C",
    ]
    if fit_dropped or calibration_dropped:
        notes.append(
            f"dropped {fit_dropped + calibration_dropped} row(s) whose labels were not "
            "observable at the relevant stage deadline"
        )

    report = FitReport(
        variant=variant,
        columns=columns,
        fit_first_session=min(fit_sessions),
        fit_last_session=max(fit_sessions),
        fit_rows=len(fit_kept),
        fit_dates=len(set(fit_sessions)),
        calibration_first_session=min(calibration_sessions),
        calibration_last_session=max(calibration_sessions),
        calibration_rows=len(calibration_kept),
        calibration_dates=len(set(calibration_sessions)),
        clip_fraction=clip_fraction,
        positive_label_rate=float(np.average(labels, weights=weights)),
        # True only when the check actually ran, not merely when it is required.
        label_timestamps_checked=(
            config.validation.require_label_available_timestamp_check
            and fit_deadline is not None
            and calibration_deadline is not None
        ),
        sample_weight_total=float(weights.sum()),
        dropped_rows=fit_dropped + calibration_dropped,
        notes=tuple(notes),
    )

    return DecisionModel(
        variant=variant,
        columns=columns,
        preprocessor=preprocessor,
        ridge=ridge,
        logistic=logistic,
        calibrator=calibrator,
        base_costs=base_costs,
        stress_costs=stress_costs,
        report=report,
        # The constant a calibrated probability must beat out of sample.
        training_base_rate=float(np.average(labels, weights=weights)),
    )


def _usable_rows(
    rows: Sequence[OriginRow], deadline: dt.datetime | None
) -> tuple[list[OriginRow], int]:
    """Labelled rows whose outcomes were observable by ``deadline``.

    With a deadline, a row that does not say when its label became observable
    is dropped, not trusted: an unknown availability time cannot be shown to
    precede the deadline, and the check must fail closed.
    """
    kept: list[OriginRow] = []
    dropped = 0
    for row in rows:
        if not row.labelled:
            dropped += 1
            continue
        if deadline is not None and (
            row.label_available_at is None or row.label_available_at > deadline
        ):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped
