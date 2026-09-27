"""Chronology: labelled origins, folds, purges, the holdout and refits.

The checkpoint's weights are dated, so the conservative earliest forecast origin
for the main study is the first trading session after that date. Earlier history
is still useful as context and as a separately labelled diagnostic, but it is
not evidence that this model could have been deployed then.

What this module enforces:

* An origin enters outcome analysis only once its D2 exit has been observed.
  With a two-session hold, the last two origins before the data edge have no
  outcome yet -- and that is session arithmetic, not calendar-day subtraction.
* The final 60 fully labelled post-checkpoint origins are reserved and
  evaluated once. Earlier post-checkpoint origins are development.
* Development folds expand chronologically with a purge on both boundaries.
  Every symbol of a date belongs to the same split, so a date is the atomic
  unit throughout.
* A label may only enter a stage whose fit time is at or after the moment the
  label became observable. That timestamp check overrides the fixed gap length
  rather than being replaced by it.
* Deployment is prequential: refit every 20 sessions from matured preceding
  history using the same recipe, without touching features, thresholds or
  hyperparameters.

If development turns out to hold too few sessions, the answer is to stay in
research and collect more dates. Shortening a block after seeing performance is
the one adjustment that cannot be undone.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from .calendar_spec import ExchangeCalendar
from .config import DesignConfig
from .holdout import AccessMode, HoldoutGuard

__all__ = [
    "ProtocolError",
    "InsufficientHistory",
    "OriginLabelTiming",
    "Fold",
    "RefitPoint",
    "ProtocolSchedule",
    "label_available_at",
    "labelled_origins",
    "build_schedule",
    "build_refit_point",
]


class ProtocolError(RuntimeError):
    """Raised on an impossible or unsafe chronology request."""


class InsufficientHistory(ProtocolError):
    """Raised when the clean history cannot support the registered blocks.

    Deliberately an error rather than a silent shrink. The recorded response is
    to remain in research and collect more dates.
    """


@dataclass(frozen=True)
class OriginLabelTiming:
    """When one origin's outcome becomes observable."""

    origin_session: dt.date
    entry_session: dt.date
    exit_session: dt.date
    available_at: pd.Timestamp

    def matured_by(self, deadline: pd.Timestamp | dt.datetime) -> bool:
        return self.available_at <= pd.Timestamp(deadline)


def label_available_at(
    calendar: ExchangeCalendar,
    origin_session: dt.date,
    *,
    horizon_sessions: int = 2,
    delay_minutes: int = 30,
) -> OriginLabelTiming:
    """Entry, exit and label-availability timestamps for one origin.

    The label is observable once the exit session's closing price has been
    collected, which is modelled with the same close-plus-delay convention the
    after-close run uses.
    """
    horizon = calendar.horizon(origin_session, horizon_sessions)
    entry = horizon.session_of(1)
    exit_session = horizon.session_of(horizon_sessions)
    return OriginLabelTiming(
        origin_session=origin_session,
        entry_session=entry,
        exit_session=exit_session,
        available_at=calendar.signal_time(exit_session, delay_minutes=delay_minutes),
    )


def labelled_origins(
    calendar: ExchangeCalendar,
    config: DesignConfig,
    *,
    latest_data_session: dt.date,
    earliest_origin: dt.date | None = None,
) -> list[OriginLabelTiming]:
    """Origins whose two-session outcome has been observed.

    Args:
        latest_data_session: The most recent session with complete data.
        earliest_origin: Defaults to the registered earliest primary origin.

    An origin whose exit session falls after ``latest_data_session`` is excluded,
    which is why the last two origins before the data edge never appear.
    """
    start = earliest_origin or config.validation.earliest_primary_origin
    if latest_data_session < start:
        return []
    horizon_sessions = config.model.horizon_sessions
    timings: list[OriginLabelTiming] = []
    for origin in calendar.sessions(start, latest_data_session):
        try:
            timing = label_available_at(
                calendar, origin, horizon_sessions=horizon_sessions
            )
        except Exception:  # calendar cannot reach the exit yet
            continue
        if timing.exit_session <= latest_data_session:
            timings.append(timing)
    return timings


@dataclass(frozen=True)
class Fold:
    """One expanding development fold.

    Attributes:
        index: Zero-based fold number.
        fit_sessions: Origins used to fit both heads.
        calibration_sessions: Later, disjoint origins used for sigmoid calibration.
        validation_sessions: Origins scored by this fold.
        purge_before_calibration: Sessions skipped between fit and calibration.
        purge_before_validation: Sessions skipped between calibration and validation.
        fit_deadline: Moment the fitting stage runs; a label not observable by
            then may not enter the fit.
        calibration_deadline: The same, for the calibration stage.
    """

    index: int
    fit_sessions: tuple[dt.date, ...]
    calibration_sessions: tuple[dt.date, ...]
    validation_sessions: tuple[dt.date, ...]
    purge_before_calibration: tuple[dt.date, ...]
    purge_before_validation: tuple[dt.date, ...]
    fit_deadline: pd.Timestamp
    calibration_deadline: pd.Timestamp

    def __post_init__(self) -> None:
        blocks = {
            "fit": set(self.fit_sessions),
            "calibration": set(self.calibration_sessions),
            "validation": set(self.validation_sessions),
            "purge": set(self.purge_before_calibration) | set(self.purge_before_validation),
        }
        names = sorted(blocks)
        for position, left in enumerate(names):
            for right in names[position + 1 :]:
                shared = blocks[left] & blocks[right]
                if shared:
                    raise ProtocolError(
                        f"fold {self.index}: {left} and {right} blocks share "
                        f"{len(shared)} date(s); a date belongs to exactly one split"
                    )

    def describe(self) -> dict[str, object]:
        return {
            "index": self.index,
            "fit_first": self.fit_sessions[0].isoformat(),
            "fit_last": self.fit_sessions[-1].isoformat(),
            "fit_sessions": len(self.fit_sessions),
            "calibration_first": self.calibration_sessions[0].isoformat(),
            "calibration_last": self.calibration_sessions[-1].isoformat(),
            "calibration_sessions": len(self.calibration_sessions),
            "validation_first": self.validation_sessions[0].isoformat(),
            "validation_last": self.validation_sessions[-1].isoformat(),
            "validation_sessions": len(self.validation_sessions),
            "fit_deadline": self.fit_deadline.isoformat(),
            "calibration_deadline": self.calibration_deadline.isoformat(),
        }


@dataclass(frozen=True)
class RefitPoint:
    """One scheduled prequential refit.

    Attributes:
        effective_from: First origin scored by this fit.
        fit_sessions: Matured origins used for fitting.
        calibration_sessions: Matured origins used for calibration.
        purged_sessions: Matured origins deliberately skipped as the gap.
        deadline: The moment of the refit. Only labels observable by then enter.
    """

    effective_from: dt.date
    fit_sessions: tuple[dt.date, ...]
    calibration_sessions: tuple[dt.date, ...]
    purged_sessions: tuple[dt.date, ...]
    deadline: pd.Timestamp

    def describe(self) -> dict[str, object]:
        return {
            "effective_from": self.effective_from.isoformat(),
            "fit_sessions": len(self.fit_sessions),
            "calibration_sessions": len(self.calibration_sessions),
            "purged_sessions": len(self.purged_sessions),
            "deadline": self.deadline.isoformat(),
        }


@dataclass(frozen=True)
class ProtocolSchedule:
    """The whole chronology for one study."""

    timings: tuple[OriginLabelTiming, ...]
    development_origins: tuple[dt.date, ...]
    test_origins: tuple[dt.date, ...]
    folds: tuple[Fold, ...]
    refit_points: tuple[RefitPoint, ...]
    notes: tuple[str, ...] = ()

    @property
    def labelled_origins(self) -> tuple[dt.date, ...]:
        return tuple(timing.origin_session for timing in self.timings)

    def timing_for(self, origin: dt.date) -> OriginLabelTiming:
        for timing in self.timings:
            if timing.origin_session == origin:
                return timing
        raise ProtocolError(f"{origin.isoformat()} is not a labelled origin")

    def guard(self, mode: AccessMode = AccessMode.DEVELOPMENT) -> HoldoutGuard:
        """A guard that refuses development access to the reserved origins."""
        return HoldoutGuard.from_sessions(self.test_origins, mode=mode)

    def describe(self) -> dict[str, object]:
        return {
            "labelled_origins": len(self.timings),
            "development_origins": len(self.development_origins),
            "test_origins": len(self.test_origins),
            "folds": [fold.describe() for fold in self.folds],
            "refit_points": [point.describe() for point in self.refit_points],
            "notes": list(self.notes),
        }


def build_schedule(
    calendar: ExchangeCalendar,
    config: DesignConfig,
    *,
    latest_data_session: dt.date,
    earliest_origin: dt.date | None = None,
    require_folds: bool = True,
) -> ProtocolSchedule:
    """Build the development folds, the reserved test block and the refit points.

    Args:
        require_folds: When ``True``, too little history raises
            :class:`InsufficientHistory`. Set ``False`` only to inspect how much
            history a given data edge would provide.

    Raises:
        InsufficientHistory: If the clean history cannot support even one fold
            at the registered block sizes.
    """
    validation = config.validation
    timings = labelled_origins(
        calendar,
        config,
        latest_data_session=latest_data_session,
        earliest_origin=earliest_origin,
    )
    origins = [timing.origin_session for timing in timings]
    notes: list[str] = []

    reserved_count = validation.final_historical_test_origin_sessions
    if len(origins) <= reserved_count:
        message = (
            f"{len(origins)} labelled post-checkpoint origins is not more than the "
            f"{reserved_count} reserved for the final test; there is no development "
            "history yet. Stay in research and collect more dates."
        )
        if require_folds:
            raise InsufficientHistory(message)
        notes.append(message)
        return ProtocolSchedule(
            timings=tuple(timings),
            development_origins=(),
            test_origins=tuple(origins),
            folds=(),
            refit_points=(),
            notes=tuple(notes),
        )

    development = origins[:-reserved_count]
    test = origins[-reserved_count:]

    folds = _build_folds(calendar, config, development)
    if not folds:
        message = (
            f"{len(development)} development origins cannot support one fold of "
            f"{validation.min_fit_origin_sessions} fit + "
            f"{validation.purge_sessions_per_boundary} purge + "
            f"{validation.calibration_origin_sessions} calibration + "
            f"{validation.purge_sessions_per_boundary} purge + "
            f"{validation.validation_origin_sessions} validation sessions. "
            "This clean history supports only a small number of folds, and shortening "
            "a block after seeing performance is not permitted."
        )
        if require_folds:
            raise InsufficientHistory(message)
        notes.append(message)

    refit_points = _build_refit_schedule(calendar, config, origins, test)
    if folds:
        notes.append(
            f"{len(folds)} development fold(s) available; interval width reflects this "
            "short history rather than a five-year validation."
        )
    return ProtocolSchedule(
        timings=tuple(timings),
        development_origins=tuple(development),
        test_origins=tuple(test),
        folds=tuple(folds),
        refit_points=tuple(refit_points),
        notes=tuple(notes),
    )


def _build_folds(
    calendar: ExchangeCalendar,
    config: DesignConfig,
    development: Sequence[dt.date],
) -> list[Fold]:
    """Expanding folds over the development origins."""
    validation = config.validation
    fit_min = validation.min_fit_origin_sessions
    calibration_size = validation.calibration_origin_sessions
    validation_size = validation.validation_origin_sessions
    purge = validation.purge_sessions_per_boundary
    step = validation_size

    folds: list[Fold] = []
    index = 0
    while True:
        fit_end = fit_min + index * step
        calibration_start = fit_end + purge
        calibration_end = calibration_start + calibration_size
        validation_start = calibration_end + purge
        validation_end = validation_start + validation_size
        if validation_end > len(development):
            break
        fit_sessions = tuple(development[:fit_end])
        purge_one = tuple(development[fit_end:calibration_start])
        calibration_sessions = tuple(development[calibration_start:calibration_end])
        purge_two = tuple(development[calibration_end:validation_start])
        validation_sessions = tuple(development[validation_start:validation_end])

        fit_deadline = calendar.signal_time(calibration_sessions[0])
        calibration_deadline = calendar.signal_time(validation_sessions[0])
        fold = Fold(
            index=index,
            fit_sessions=fit_sessions,
            calibration_sessions=calibration_sessions,
            validation_sessions=validation_sessions,
            purge_before_calibration=purge_one,
            purge_before_validation=purge_two,
            fit_deadline=fit_deadline,
            calibration_deadline=calibration_deadline,
        )
        _assert_no_label_leak(calendar, config, fold)
        folds.append(fold)
        index += 1
    return folds


def _assert_no_label_leak(
    calendar: ExchangeCalendar, config: DesignConfig, fold: Fold
) -> None:
    """Verify the purge is actually long enough for this fold's calendar.

    The fixed gap and the timestamp check are both applied: the gap is the
    design, and this is the verification that the design held on these
    particular sessions. Historical feature windows may legitimately share
    older public prices across a boundary; an *outcome* window may not.
    """
    horizon = config.model.horizon_sessions
    for session in fold.fit_sessions:
        timing = label_available_at(calendar, session, horizon_sessions=horizon)
        if not timing.matured_by(fold.fit_deadline):
            raise ProtocolError(
                f"fold {fold.index}: fit origin {session.isoformat()} has an outcome "
                f"observable only at {timing.available_at.isoformat()}, after the "
                f"fitting deadline {fold.fit_deadline.isoformat()}"
            )
    for session in fold.calibration_sessions:
        timing = label_available_at(calendar, session, horizon_sessions=horizon)
        if not timing.matured_by(fold.calibration_deadline):
            raise ProtocolError(
                f"fold {fold.index}: calibration origin {session.isoformat()} has an "
                f"outcome observable only at {timing.available_at.isoformat()}, after "
                f"the calibration deadline {fold.calibration_deadline.isoformat()}"
            )
    first_validation = fold.validation_sessions[0]
    for session in (*fold.fit_sessions, *fold.calibration_sessions):
        timing = label_available_at(calendar, session, horizon_sessions=horizon)
        if timing.exit_session >= first_validation:
            raise ProtocolError(
                f"fold {fold.index}: origin {session.isoformat()} exits on "
                f"{timing.exit_session.isoformat()}, which overlaps the validation "
                f"block starting {first_validation.isoformat()}"
            )


def build_refit_point(
    calendar: ExchangeCalendar,
    config: DesignConfig,
    *,
    available_origins: Iterable[dt.date],
    effective_from: dt.date,
) -> RefitPoint:
    """Build one prequential refit from matured preceding history.

    Only origins whose outcomes were observable at the refit moment are used,
    and the registered purge is then applied on top as an explicit gap. The
    same recipe is replayed identically in the final test; no manual,
    test-driven redesign is permitted.

    Raises:
        InsufficientHistory: If matured history cannot fill the registered
            fitting and calibration blocks.
    """
    validation = config.validation
    horizon = config.model.horizon_sessions
    deadline = calendar.signal_time(effective_from)

    matured = [
        origin
        for origin in sorted(set(available_origins))
        if origin < effective_from
        and label_available_at(
            calendar, origin, horizon_sessions=horizon
        ).matured_by(deadline)
    ]
    purge = min(max(validation.purge_sessions_per_boundary, 0), len(matured))
    purged = tuple(matured[len(matured) - purge :]) if purge else ()
    usable = matured[: len(matured) - purge]

    needed = validation.min_fit_origin_sessions + validation.calibration_origin_sessions
    if len(usable) < needed:
        raise InsufficientHistory(
            f"refit effective {effective_from.isoformat()}: {len(usable)} matured "
            f"origins after the purge, need {needed}"
        )
    calibration = tuple(usable[-validation.calibration_origin_sessions :])
    fit = tuple(usable[: len(usable) - validation.calibration_origin_sessions])
    return RefitPoint(
        effective_from=effective_from,
        fit_sessions=fit,
        calibration_sessions=calibration,
        purged_sessions=purged,
        deadline=deadline,
    )


def _build_refit_schedule(
    calendar: ExchangeCalendar,
    config: DesignConfig,
    all_origins: Sequence[dt.date],
    test_origins: Sequence[dt.date],
) -> list[RefitPoint]:
    """Refit points every ``decision_refit_every_sessions`` across the test block.

    Built from the schedule alone, so the same points are replayed whether the
    final test is being planned or executed.
    """
    if not test_origins:
        return []
    step = config.validation.decision_refit_every_sessions
    points: list[RefitPoint] = []
    for position in range(0, len(test_origins), step):
        effective_from = test_origins[position]
        try:
            points.append(
                build_refit_point(
                    calendar,
                    config,
                    available_origins=all_origins,
                    effective_from=effective_from,
                )
            )
        except InsufficientHistory:
            # Recorded as absent rather than approximated with a shorter block.
            continue
    return points
