"""Candidate roster and the frozen watchlist manifest.

The watchlist is frozen *before* any model performance metric exists. That
ordering is the whole point: relative strength is a feature, not a membership
rule, so a symbol may not enter or leave the manifest because of how the
strategy performed on it.

What this module cannot fix, and therefore states plainly: selecting today's
liquid survivors gives a current watchlist, not a historically unbiased
universe. Results on it are conditional watchlist evidence. Delistings and
missing exit prices stay in the forward ledger rather than disappearing.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import DesignConfig
from .provenance import stable_hash

__all__ = [
    "UniverseError",
    "FreezeError",
    "Candidate",
    "CandidateRoster",
    "WatchlistMember",
    "WatchlistManifest",
    "select_watchlist",
    "DISALLOWED_SECURITY_TYPES",
]

#: Security types that may never enter the trade universe. ETFs appear here
#: because they are contextual inputs: SPY and the sector proxies are features,
#: not candidates.
DISALLOWED_SECURITY_TYPES = frozenset(
    {
        "etf",
        "etn",
        "leveraged_etf",
        "inverse_etf",
        "closed_end_fund",
        "adr_preferred",
        "preferred",
        "warrant",
        "unit",
        "right",
        "spac",
    }
)


class UniverseError(RuntimeError):
    """Raised on an invalid roster or manifest."""


class FreezeError(UniverseError):
    """Raised when a manifest freeze would come after results were produced."""


@dataclass(frozen=True)
class Candidate:
    """One issuer on the declared candidate roster."""

    symbol: str
    issuer_id: str
    exchange: str
    sector: str
    sector_etf: str
    share_class: str = "common"
    security_type: str = "common_stock"
    listing_history: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise UniverseError("candidate symbol must not be empty")
        if not self.issuer_id.strip():
            raise UniverseError(f"{self.symbol}: a stable issuer identifier is required")
        if not self.sector.strip() or not self.sector_etf.strip():
            raise UniverseError(f"{self.symbol}: sector and sector ETF proxy are required")
        if self.security_type.lower() in DISALLOWED_SECURITY_TYPES:
            raise UniverseError(
                f"{self.symbol}: security type {self.security_type!r} may not enter the "
                "trade universe"
            )


@dataclass(frozen=True)
class CandidateRoster:
    """A declared roster, chosen independently of this experiment's returns.

    Attributes:
        candidates: Roster members.
        source: Where the roster came from, recorded for the manifest.
        declared_at: When the roster was declared.
    """

    candidates: tuple[Candidate, ...]
    source: str
    declared_at: dt.datetime

    def __post_init__(self) -> None:
        by_symbol: dict[str, Candidate] = {}
        by_issuer: dict[str, Candidate] = {}
        for candidate in self.candidates:
            if candidate.symbol in by_symbol:
                raise UniverseError(f"duplicate symbol {candidate.symbol} on the roster")
            if candidate.issuer_id in by_issuer:
                raise UniverseError(
                    f"issuer {candidate.issuer_id} appears twice "
                    f"({by_issuer[candidate.issuer_id].symbol} and {candidate.symbol}); "
                    "one share class per issuer"
                )
            by_symbol[candidate.symbol] = candidate
            by_issuer[candidate.issuer_id] = candidate

    @classmethod
    def from_csv(cls, path: str | Path, *, source: str | None = None) -> "CandidateRoster":
        """Load a roster from CSV.

        The file is the declaration. Its hash goes into the manifest, so a
        roster edited after the fact is detectable.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise UniverseError(f"candidate roster not found at {file_path}")
        rows: list[Candidate] = []
        with file_path.open(newline="", encoding="utf-8") as handle:
            # Comment lines let a roster carry its own provenance note without
            # a second file.
            lines = [
                line for line in handle if line.strip() and not line.lstrip().startswith("#")
            ]
            reader = csv.DictReader(lines)
            required = {"symbol", "issuer_id", "exchange", "sector", "sector_etf"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise UniverseError(
                    f"{file_path}: roster is missing columns {sorted(missing)}"
                )
            for record in reader:
                rows.append(
                    Candidate(
                        symbol=record["symbol"].strip().upper(),
                        issuer_id=record["issuer_id"].strip(),
                        exchange=record["exchange"].strip(),
                        sector=record["sector"].strip(),
                        sector_etf=record["sector_etf"].strip().upper(),
                        share_class=(record.get("share_class") or "common").strip(),
                        security_type=(
                            record.get("security_type") or "common_stock"
                        ).strip(),
                        listing_history=(record.get("listing_history") or "").strip(),
                        notes=(record.get("notes") or "").strip(),
                    )
                )
        if not rows:
            raise UniverseError(f"{file_path}: roster is empty")
        return cls(
            candidates=tuple(rows),
            source=source or str(file_path),
            declared_at=dt.datetime.now(dt.timezone.utc),
        )

    def hash(self) -> str:
        return stable_hash(
            {
                "source": self.source,
                "candidates": [
                    {
                        "symbol": candidate.symbol,
                        "issuer_id": candidate.issuer_id,
                        "exchange": candidate.exchange,
                        "sector": candidate.sector,
                        "sector_etf": candidate.sector_etf,
                        "share_class": candidate.share_class,
                        "security_type": candidate.security_type,
                    }
                    for candidate in sorted(self.candidates, key=lambda c: c.symbol)
                ],
            }
        )

    def sectors(self) -> set[str]:
        return {candidate.sector for candidate in self.candidates}

    def by_symbol(self) -> dict[str, Candidate]:
        return {candidate.symbol: candidate for candidate in self.candidates}

    def __len__(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True)
class WatchlistMember(Candidate):
    """A roster member that made the frozen watchlist, with its selection metric."""

    selection_dollar_volume: float = float("nan")
    selection_rank: int = -1


@dataclass(frozen=True)
class WatchlistManifest:
    """The frozen trade universe.

    Attributes:
        members: Selected issuers, in selection order.
        selection_session: Session whose trailing window drove the ranking.
        frozen_at: When the manifest was frozen.
        roster_hash: Hash of the declared candidate roster.
        selection_rule: Human-readable statement of the rule applied.
        skipped: Candidates not selected, with the reason.
    """

    members: tuple[WatchlistMember, ...]
    selection_session: dt.date
    frozen_at: dt.datetime
    roster_hash: str
    roster_source: str
    selection_rule: str
    skipped: tuple[tuple[str, str], ...] = ()
    limitations: tuple[str, ...] = (
        "Today's surviving watchlist carries survivorship and selection bias; "
        "historical results on it are conditional watchlist evidence only.",
        "Without a point-in-time universe and delisting outcomes, no whole-market "
        "claim is supported.",
    )

    def hash(self) -> str:
        return stable_hash(
            {
                "roster_hash": self.roster_hash,
                "selection_session": self.selection_session.isoformat(),
                "selection_rule": self.selection_rule,
                "members": [
                    {
                        "symbol": member.symbol,
                        "issuer_id": member.issuer_id,
                        "sector": member.sector,
                        "sector_etf": member.sector_etf,
                        "rank": member.selection_rank,
                    }
                    for member in self.members
                ],
            }
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.members)

    def sector_of(self, symbol: str) -> str:
        for member in self.members:
            if member.symbol == symbol:
                return member.sector
        raise UniverseError(f"{symbol} is not on the frozen watchlist")

    def sector_etfs(self) -> dict[str, str]:
        return {member.sector: member.sector_etf for member in self.members}

    def sector_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for member in self.members:
            counts[member.sector] = counts.get(member.sector, 0) + 1
        return counts

    def peers(self, sector: str) -> tuple[str, ...]:
        return tuple(
            member.symbol for member in self.members if member.sector == sector
        )

    def to_rows(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": member.symbol,
                "issuer_id": member.issuer_id,
                "exchange": member.exchange,
                "sector": member.sector,
                "sector_etf": member.sector_etf,
                "share_class": member.share_class,
                "listing_history": member.listing_history,
                "selection_timestamp": self.frozen_at,
                "candidate_roster": self.roster_hash,
                "status": "active",
                "notes": member.notes,
            }
            for member in self.members
        ]


def select_watchlist(
    roster: CandidateRoster,
    *,
    config: DesignConfig,
    selection_session: dt.date,
    dollar_volume: Mapping[str, float],
    eligible: Iterable[str] | None = None,
    frozen_at: dt.datetime | None = None,
    results_already_exist: bool = False,
) -> WatchlistManifest:
    """Freeze the trade universe from a declared roster.

    The rule is fixed: among roster members that pass the data audit, take the
    highest trailing-window median dollar volume, at most
    ``max_issuers_per_sector`` per sector and at most ``max_issuers`` in total.
    Ties break on symbol so the outcome is reproducible.

    Args:
        results_already_exist: Set by the caller when the ledger already holds
            outcomes. Freezing then would mean selecting a universe with
            knowledge of performance, so it raises.

    Raises:
        FreezeError: If performance already exists, or the roster is required
            but absent.
    """
    if results_already_exist:
        raise FreezeError(
            "refusing to freeze a watchlist after outcomes exist: the manifest must be "
            "fixed before any model performance metric is produced"
        )
    if config.universe.candidate_manifest_required and not len(roster):
        raise FreezeError("a declared candidate roster is required before selection")

    allowed = set(eligible) if eligible is not None else None
    skipped: list[tuple[str, str]] = []
    scored: list[tuple[float, str, Candidate]] = []

    for candidate in roster.candidates:
        if allowed is not None and candidate.symbol not in allowed:
            skipped.append((candidate.symbol, "failed the data audit or eligibility gates"))
            continue
        volume = dollar_volume.get(candidate.symbol)
        if volume is None:
            skipped.append((candidate.symbol, "no trailing dollar-volume measurement"))
            continue
        if math.isnan(volume) or math.isinf(volume):
            skipped.append((candidate.symbol, "non-finite trailing dollar volume"))
            continue
        if volume < config.universe.min_median_daily_dollar_volume_usd:
            skipped.append(
                (
                    candidate.symbol,
                    f"trailing median dollar volume {volume:,.0f} below the "
                    f"{config.universe.min_median_daily_dollar_volume_usd:,.0f} floor",
                )
            )
            continue
        scored.append((volume, candidate.symbol, candidate))

    # Descending dollar volume, ascending symbol on ties.
    scored.sort(key=lambda item: (-item[0], item[1]))

    members: list[WatchlistMember] = []
    per_sector: dict[str, int] = {}
    for volume, symbol, candidate in scored:
        if len(members) >= config.universe.max_issuers:
            skipped.append((symbol, "watchlist already at its issuer cap"))
            continue
        taken = per_sector.get(candidate.sector, 0)
        if taken >= config.universe.max_issuers_per_sector:
            skipped.append(
                (symbol, f"sector {candidate.sector} already at its per-sector cap")
            )
            continue
        per_sector[candidate.sector] = taken + 1
        members.append(
            WatchlistMember(
                symbol=candidate.symbol,
                issuer_id=candidate.issuer_id,
                exchange=candidate.exchange,
                sector=candidate.sector,
                sector_etf=candidate.sector_etf,
                share_class=candidate.share_class,
                security_type=candidate.security_type,
                listing_history=candidate.listing_history,
                notes=candidate.notes,
                selection_dollar_volume=volume,
                selection_rank=len(members) + 1,
            )
        )

    market_proxy = config.runtime.market_proxy
    for member in members:
        if member.symbol in {market_proxy, member.sector_etf}:
            raise UniverseError(
                f"{member.symbol} is a contextual input, not a trade candidate"
            )

    rule = (
        f"top {config.universe.max_issuers} declared-roster issuers by trailing-"
        f"{config.universe.liquidity_window_sessions}-session median dollar volume, "
        f"at most {config.universe.max_issuers_per_sector} per sector, ties on symbol; "
        "underfilled sectors accepted"
    )
    return WatchlistManifest(
        members=tuple(members),
        selection_session=selection_session,
        frozen_at=frozen_at or dt.datetime.now(dt.timezone.utc),
        roster_hash=roster.hash(),
        roster_source=roster.source,
        selection_rule=rule,
        skipped=tuple(skipped),
    )


@dataclass
class SectorContext:
    """Mapping from sector to its frozen ETF proxy, plus the market proxy."""

    market_proxy: str
    sector_etfs: Mapping[str, str]
    extras: Sequence[str] = field(default_factory=tuple)

    def context_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted({self.market_proxy, *self.sector_etfs.values(), *self.extras})
        )

    def etf_for(self, sector: str) -> str:
        if sector not in self.sector_etfs:
            raise UniverseError(f"no frozen sector ETF proxy for sector {sector!r}")
        return self.sector_etfs[sector]
