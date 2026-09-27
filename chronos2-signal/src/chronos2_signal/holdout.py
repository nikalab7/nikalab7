"""Access control for the reserved final test origins.

The final 60 fully labelled post-checkpoint origin sessions are evaluated once.
"Do not look at the holdout" is not a discipline that survives a long project,
so it is enforced mechanically here: development-mode code that touches a
reserved origin raises instead of quietly returning rows.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

__all__ = ["HoldoutViolation", "AccessMode", "HoldoutGuard"]


class HoldoutViolation(RuntimeError):
    """Raised when reserved holdout data is accessed outside the final test."""


class AccessMode(str, Enum):
    """Which part of the protocol the current process is executing."""

    #: Fold construction, fitting, calibration and candidate selection.
    DEVELOPMENT = "development"
    #: The single registered pass over the reserved origins.
    FINAL_TEST = "final_test"
    #: Timestamped forward paper ledger, after the final test.
    FORWARD = "forward"
    #: Data audits and integrity fixtures that carry no performance meaning.
    INTEGRITY = "integrity"


@dataclass
class HoldoutGuard:
    """Gatekeeper for the reserved origin sessions.

    Attributes:
        reserved: Origin sessions reserved for the single final evaluation.
        mode: Current access mode.
        unlocked: Set once, deliberately, to run the final test.
    """

    reserved: frozenset[dt.date]
    mode: AccessMode = AccessMode.DEVELOPMENT
    unlocked: bool = False

    @classmethod
    def from_sessions(
        cls,
        reserved: Iterable[dt.date],
        *,
        mode: AccessMode = AccessMode.DEVELOPMENT,
    ) -> "HoldoutGuard":
        return cls(reserved=frozenset(reserved), mode=mode)

    @classmethod
    def open(cls) -> "HoldoutGuard":
        """A guard with nothing reserved, for integrity fixtures and audits."""
        return cls(reserved=frozenset(), mode=AccessMode.INTEGRITY, unlocked=True)

    def is_reserved(self, session: dt.date) -> bool:
        return session in self.reserved

    def unlock_final_test(self) -> None:
        """Enter the single registered final-test pass.

        Deliberately one-way within a process. Re-locking and re-running after
        seeing results is exactly the loop the protocol forbids: a failed test
        cannot be relabelled development and reused.
        """
        self.mode = AccessMode.FINAL_TEST
        self.unlocked = True

    def check(self, session: dt.date, *, purpose: str = "read") -> None:
        """Raise if ``session`` may not be touched in the current mode."""
        if not self.is_reserved(session):
            return
        if self.unlocked and self.mode in (AccessMode.FINAL_TEST, AccessMode.INTEGRITY):
            return
        raise HoldoutViolation(
            f"origin {session.isoformat()} is reserved for the final historical test; "
            f"refusing {purpose} in mode {self.mode.value}. Unlock deliberately via "
            "HoldoutGuard.unlock_final_test() only for the single registered pass."
        )

    def check_all(self, sessions: Iterable[dt.date], *, purpose: str = "read") -> None:
        for session in sessions:
            self.check(session, purpose=purpose)

    def filter_visible(self, sessions: Iterable[dt.date]) -> list[dt.date]:
        """Drop reserved sessions when the current mode may not see them.

        Used by report builders, which should show fewer rows rather than fail.
        """
        if self.unlocked and self.mode in (AccessMode.FINAL_TEST, AccessMode.INTEGRITY):
            return list(sessions)
        return [session for session in sessions if not self.is_reserved(session)]

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "unlocked": self.unlocked,
            "reserved_sessions": len(self.reserved),
            "reserved_first": min(self.reserved).isoformat() if self.reserved else None,
            "reserved_last": max(self.reserved).isoformat() if self.reserved else None,
        }
