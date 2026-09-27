"""Rendering of persisted signals.

Two rules define this module:

* It renders **only** what is already committed to the ledger. A signal that
  was not persisted cannot be notified, which is what makes a retry after a
  delivery failure idempotent rather than duplicative.
* No channel is contacted here. A personal Telegram or email adapter can be
  implemented later against :class:`Channel`; nothing in this repository
  configures or reaches one, and any credential would live in an environment
  variable rather than in source or a report.

The rendered text never promises a rise. Before the evaluation gates pass every
output is labelled ``RESEARCH / UNVALIDATED``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .policy import OutputLabel
from .storage import Ledger

__all__ = [
    "NotifierError",
    "Channel",
    "ConsoleChannel",
    "RecordingChannel",
    "render_signal",
    "render_no_alert_notice",
    "Notifier",
]


class NotifierError(RuntimeError):
    """Raised when a notification is attempted without a persisted signal."""


class Channel(Protocol):
    """Somewhere a rendered signal can be delivered."""

    name: str

    def send(self, subject: str, body: str) -> None:  # pragma: no cover
        ...


@dataclass
class ConsoleChannel:
    """Writes to standard output. The only channel shipped."""

    name: str = "console"

    def send(self, subject: str, body: str) -> None:
        print(f"=== {subject} ===\n{body}\n")


@dataclass
class RecordingChannel:
    """Collects messages in memory, for tests."""

    name: str = "recording"
    messages: list[tuple[str, str]] = field(default_factory=list)

    def send(self, subject: str, body: str) -> None:
        self.messages.append((subject, body))


def render_signal(row: Mapping[str, object]) -> tuple[str, str]:
    """Render one persisted signal row as ``(subject, body)``.

    Everything shown comes from the stored payload. No current price is fetched
    and none is invented: a rendered alert that quoted a fresh price would be
    describing a different moment than the decision it reports.
    """
    payload_raw = row.get("payload")
    payload: Mapping[str, object]
    if isinstance(payload_raw, str):
        payload = json.loads(payload_raw)
    elif isinstance(payload_raw, Mapping):
        payload = payload_raw
    else:
        payload = {}

    label = str(row.get("output_label") or payload.get("output_label") or
                OutputLabel.RESEARCH_UNVALIDATED.value)
    symbol = str(row.get("symbol", "?"))
    session = str(row.get("signal_session", "?"))
    subject = f"[{label}] {symbol} - signal session {session}"

    def number(key: str, digits: int = 4) -> str:
        value = row.get(key, payload.get(key))
        try:
            return f"{float(value):.{digits}f}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "unavailable"

    coverage = payload.get("empirical_coverage") or {}
    coverage_text = (
        ", ".join(f"{name}={value}" for name, value in sorted(coverage.items()))  # type: ignore[union-attr]
        if isinstance(coverage, Mapping) and coverage
        else "not yet measured"
    )
    event_flags = payload.get("event_flags") or []
    risk_flags = payload.get("risk_flags") or []

    lines = [
        f"Label:                {label}",
        f"Symbol:               {symbol}  (issuer {row.get('issuer_id', '?')}, "
        f"sector {row.get('sector', '?')})",
        f"Signal session:       {session}",
        f"Data timestamp:       {row.get('data_timestamp', 'unavailable')}",
        f"Entry convention:     {payload.get('entry_convention', 'unavailable')}",
        f"Exit convention:      {payload.get('exit_convention', 'unavailable')}",
        f"Estimated net return: {number('estimated_net_return')} (model estimate, base costs)",
        f"  under stress costs: {number('estimated_net_return_stress')}",
        f"Calibrated P(R>0):    {number('calibrated_probability')} "
        "(a model estimate, not a measured frequency)",
        f"sigma_2d:             {number('sigma_2d')}",
        f"Forecast coverage:    {coverage_text}",
        f"Out-of-sample basis:  {payload.get('out_of_sample_observations', 0)} comparable "
        f"observations across {payload.get('out_of_sample_dates', 0)} dates",
        f"Event flags:          {', '.join(map(str, event_flags)) or 'none recorded'}",
        f"Risk flags:           {', '.join(map(str, risk_flags)) or 'none recorded'}",
        f"Expires:              {row.get('expires_at', 'next opening execution window')}",
    ]
    caveats = payload.get("caveats") or []
    if isinstance(caveats, Sequence) and not isinstance(caveats, str):
        lines.append("Caveats:")
        lines.extend(f"  - {caveat}" for caveat in caveats)
    forecaster = payload.get("forecaster")
    if isinstance(forecaster, Mapping) and forecaster.get("is_frozen_checkpoint") is False:
        lines.append(
            "  - Produced with a non-model forecaster fixture: this carries no "
            "predictive content."
        )
    return subject, "\n".join(lines)


def render_no_alert_notice(
    origin_session: dt.date, notes: Sequence[str], label: str
) -> tuple[str, str]:
    """Render the legitimate zero-alert outcome.

    Reported rather than hidden: an origin with no alerts is a valid result, and
    silence would be indistinguishable from a failed run.
    """
    subject = f"[{label}] no alerts - signal session {origin_session.isoformat()}"
    body = "\n".join(
        [
            f"Signal session: {origin_session.isoformat()}",
            "Alerts:         0",
            "",
            "Notes:",
            *(f"  - {note}" for note in notes or ("no candidate met the registered thresholds",)),
        ]
    )
    return subject, body


@dataclass
class Notifier:
    """Delivers persisted signals exactly once.

    The ledger is the source of truth. ``notify_session`` reads committed rows,
    delivers those not yet marked, and records delivery, so re-running the same
    origin cannot produce a second message.
    """

    ledger: Ledger
    channel: Channel
    label: str = OutputLabel.RESEARCH_UNVALIDATED.value

    def notify_session(
        self,
        origin_session: dt.date,
        *,
        variant: str | None = None,
        now: dt.datetime | None = None,
        notes: Sequence[str] = (),
    ) -> int:
        """Deliver undelivered alerts for one origin. Returns the count sent."""
        now = now or dt.datetime.now(dt.timezone.utc)
        rows = self.ledger.signals_for_session(origin_session, variant)
        alerts = [
            row
            for row in rows
            if row.get("decision") == "alert" and not row.get("notified_at")
        ]
        if not alerts:
            already = any(row.get("decision") == "alert" for row in rows)
            if not already:
                subject, body = render_no_alert_notice(origin_session, notes, self.label)
                self.channel.send(subject, body)
            return 0
        for row in alerts:
            subject, body = render_signal(row)
            self.channel.send(subject, body)
            self.ledger.mark_notified(str(row["signal_id"]), now)
        return len(alerts)

    @staticmethod
    def credential_from_environment(variable: str) -> str:
        """Read a channel credential from the environment.

        Present so that a future adapter has one obvious place to look. A
        secret never belongs in source or in a report.
        """
        value = os.environ.get(variable)
        if not value:
            raise NotifierError(
                f"environment variable {variable} is not set; no channel is configured "
                "by this repository"
            )
        return value
