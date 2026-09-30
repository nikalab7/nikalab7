"""Deterministic synthetic market fixtures.

These series exist to test *plumbing*, not to estimate anything. They are a
seeded random walk with a market factor, a sector factor and an overnight gap
term. Nothing about them supports a claim regarding real returns, and no
performance number produced from them means anything.

Two properties make them useful for the integrity invariants:

* Hourly and daily bars are mutually consistent by construction -- the daily
  open is the first hourly open of the session, the daily close is the last
  hourly close, the high and low are the session extremes, and the volume is
  the session sum. A leakage test can therefore compare the two resolutions.
* The generator is a pure function of ``(symbol, bar index)``, so appending
  later sessions cannot change an earlier bar. That is what lets the tests
  distinguish a genuine leak from a fixture artefact.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .actions import ActionLedger, CorporateAction
from .calendar_spec import BarSpec, ExchangeCalendar
from .market import BarPanel, DailyPanel
from .provenance import stable_hash

__all__ = ["SyntheticSpec", "SyntheticMarket", "make_default_market"]


@dataclass(frozen=True)
class SyntheticSpec:
    """Configuration of a synthetic market.

    Attributes:
        symbols: Stock symbols to generate.
        sectors: Sector label per symbol.
        market_proxy: Market factor symbol.
        sector_etfs: Sector label to ETF symbol.
        start: First session generated.
        end: Last session generated.
        seed: Master seed; every series derives from it deterministically.
        base_price: Starting price level per symbol, defaulting to 100.
        drift_bps_per_bar: Per-bar drift, per symbol. Zero by default.
        overnight_gap_bps: Standard deviation of the overnight gap.
        actions: Corporate actions to embed, per symbol.
        emitted_price_jumps: Price discontinuities to write into the raw
            series, as ``(ex_date, factor)`` per symbol: every price before the
            ex-date is multiplied by ``factor`` and every volume divided by it.
            This emulates a provider that did *not* restate a split. With the
            factor equal to a declared split ratio the split audit should find
            ``NOT_APPLIED`` and restate it; with a different factor it should
            find the episode ambiguous and quarantine it.
    """

    symbols: tuple[str, ...]
    sectors: Mapping[str, str]
    market_proxy: str
    sector_etfs: Mapping[str, str]
    start: dt.date
    end: dt.date
    seed: int = 20260927
    base_price: Mapping[str, float] = field(default_factory=dict)
    drift_bps_per_bar: Mapping[str, float] = field(default_factory=dict)
    bar_vol_bps: float = 35.0
    market_vol_bps: float = 20.0
    sector_vol_bps: float = 15.0
    overnight_gap_bps: float = 45.0
    base_volume: float = 1_500_000.0
    actions: Mapping[str, tuple[CorporateAction, ...]] = field(default_factory=dict)
    emitted_price_jumps: Mapping[str, tuple[tuple[dt.date, float], ...]] = field(
        default_factory=dict
    )

    def spec_hash(self) -> str:
        """Identity of everything that determines the generated numbers.

        Forecast cache keys include it, so a fixture with a different seed or
        drift can never be served another fixture's cached forecasts.
        """
        return stable_hash(
            {
                "symbols": list(self.symbols),
                "sectors": dict(self.sectors),
                "market_proxy": self.market_proxy,
                "sector_etfs": dict(self.sector_etfs),
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "seed": self.seed,
                "base_price": dict(self.base_price),
                "drift_bps_per_bar": dict(self.drift_bps_per_bar),
                "vols": [
                    self.bar_vol_bps,
                    self.market_vol_bps,
                    self.sector_vol_bps,
                    self.overnight_gap_bps,
                    self.base_volume,
                ],
                "actions": {
                    symbol: [
                        [action.ex_date.isoformat(), action.action_type, action.value]
                        for action in actions
                    ]
                    for symbol, actions in self.actions.items()
                },
                "emitted_price_jumps": {
                    symbol: [[ex_date.isoformat(), factor] for ex_date, factor in jumps]
                    for symbol, jumps in self.emitted_price_jumps.items()
                },
            }
        )

    def all_symbols(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [*self.symbols, self.market_proxy, *sorted(set(self.sector_etfs.values()))]
            )
        )


class SyntheticMarket:
    """Generates aligned hourly and daily panels for a synthetic universe."""

    def __init__(self, calendar: ExchangeCalendar, spec: SyntheticSpec) -> None:
        self.calendar = calendar
        self.spec = spec
        self.sessions: list[dt.date] = calendar.sessions(spec.start, spec.end)
        if not self.sessions:
            raise ValueError("synthetic spec covers no trading sessions")
        # Seed the first session's overnight gap from a synthetic predecessor so
        # that no bar in the panel carries an undefined calendar channel.
        first_open = calendar.session_open(self.sessions[0])
        self.bars: list[BarSpec] = calendar.bar_index(
            self.sessions, previous_end=first_open - pd.Timedelta(hours=17, minutes=30)
        )
        self._factors = self._build_factors()
        self._panels: dict[str, BarPanel] = {}
        self._daily: dict[str, DailyPanel] = {}
        self.spec_hash = spec.spec_hash()

    # -- factor structure -------------------------------------------------- #

    def _rng(self, *tokens: object) -> np.random.Generator:
        """A stream keyed by ``tokens``, stable across processes.

        Python's ``hash`` of a string is salted per process, so it cannot be
        used here: a fixture that changes between runs would make every
        reproducibility test meaningless.
        """
        key = "|".join(str(token) for token in tokens)
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:8]
        return np.random.default_rng(
            np.random.SeedSequence([self.spec.seed, int.from_bytes(digest, "big")])
        )

    def _build_factors(self) -> dict[str, np.ndarray]:
        n = len(self.bars)
        factors: dict[str, np.ndarray] = {}
        rng = self._rng("market")
        factors["__market__"] = rng.normal(0.0, self.spec.market_vol_bps / 10_000.0, n)
        for sector in sorted(set(self.spec.sectors.values())):
            sector_rng = self._rng("sector", sector)
            factors[sector] = sector_rng.normal(
                0.0, self.spec.sector_vol_bps / 10_000.0, n
            )
        return factors

    def _overnight(self, symbol: str) -> np.ndarray:
        """Session-boundary gap applied to the first bar of each session."""
        rng = self._rng("gap", symbol)
        gaps = rng.normal(0.0, self.spec.overnight_gap_bps / 10_000.0, len(self.sessions))
        gaps[0] = 0.0
        return gaps

    def _log_path(self, symbol: str) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative log path and the per-bar intrabar step.

        The two are returned separately because the overnight gap belongs
        *between* bars: it moves the next session's open away from the previous
        close without being part of any bar's own open-to-close move. Keeping
        them apart is what makes the fixture's gap a real discontinuity that
        the reference strategy provably does not capture.
        """
        n = len(self.bars)
        idiosyncratic = self._rng("idio", symbol).normal(
            0.0, self.spec.bar_vol_bps / 10_000.0, n
        )
        drift = self.spec.drift_bps_per_bar.get(symbol, 0.0) / 10_000.0
        steps = idiosyncratic + drift

        if symbol != self.spec.market_proxy:
            steps = steps + self._factors["__market__"]
        sector = self.spec.sectors.get(symbol)
        if sector is None:
            # An ETF loads on the sector it proxies.
            for label, etf in self.spec.sector_etfs.items():
                if etf == symbol:
                    sector = label
                    break
        if sector is not None and sector in self._factors:
            steps = steps + self._factors[sector]

        gaps = self._overnight(symbol)
        session_index = {session: i for i, session in enumerate(self.sessions)}
        between = np.zeros(n)
        for position, bar in enumerate(self.bars):
            if bar.slot == 0:
                between[position] = gaps[session_index[bar.session]]
        return np.cumsum(steps + between), steps

    # -- panels ------------------------------------------------------------ #

    def hourly(self, symbol: str) -> BarPanel:
        """Full hourly panel for ``symbol`` across the whole generated range."""
        if symbol in self._panels:
            return self._panels[symbol]

        path, steps = self._log_path(symbol)
        base = self.spec.base_price.get(symbol, 100.0)
        closes = base * np.exp(path)

        # Each bar's open is its close discounted by that bar's own move. Inside
        # a session this makes the open equal the previous close exactly; across
        # a session boundary the overnight gap remains, unowned by any bar.
        opens = closes / np.exp(steps)

        spread = self._rng("spread", symbol).uniform(
            0.0005, 0.004, len(self.bars)
        )
        highs = np.maximum(opens, closes) * (1.0 + spread)
        lows = np.minimum(opens, closes) * (1.0 - spread)

        volume_rng = self._rng("volume", symbol)
        # A U-shaped intraday volume profile keeps the matching-slot median
        # meaningful instead of comparing a quiet slot against a busy one.
        slot_profile = np.array([1.6, 1.0, 0.8, 0.75, 0.8, 1.1, 1.5])
        volumes = np.empty(len(self.bars))
        for position, bar in enumerate(self.bars):
            profile = slot_profile[min(bar.slot, len(slot_profile) - 1)]
            duration_scale = bar.duration_minutes / 60.0
            volumes[position] = (
                self.spec.base_volume
                * profile
                * duration_scale
                * volume_rng.lognormal(0.0, 0.18)
            )

        volumes = np.round(volumes)

        # Emulate a provider that did not restate a split: before each emitted
        # ex-date, prices carry the pre-split level and volumes the pre-split
        # share count. The economic path underneath is unchanged.
        for ex_date, factor in self.spec.emitted_price_jumps.get(symbol, ()):
            before = np.asarray([bar.session < ex_date for bar in self.bars], dtype=bool)
            for series in (opens, highs, lows, closes):
                series[before] = series[before] * factor
            volumes[before] = volumes[before] / factor

        observations = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
            index=pd.DatetimeIndex([bar.start for bar in self.bars]),
        )
        panel = BarPanel.aligned(
            symbol,
            "1h",
            self.bars,
            observations,
            snapshot_hash=f"fixture-{self.spec_hash[:16]}-{symbol}-1h",
        )
        self._panels[symbol] = panel
        return panel

    def daily(self, symbol: str) -> DailyPanel:
        """Daily panel aggregated from the hourly bars of the same symbol."""
        if symbol in self._daily:
            return self._daily[symbol]
        panel = self.hourly(symbol)
        frame = panel.frame
        rows: dict[dt.date, dict[str, float]] = {}
        for session, group in frame.groupby("session", sort=True):
            observed = group[group["observed"].astype(bool)]
            if not len(observed):
                continue
            rows[session] = {  # type: ignore[index]
                "open": float(observed["open"].iloc[0]),
                "high": float(observed["high"].max()),
                "low": float(observed["low"].min()),
                "close": float(observed["close"].iloc[-1]),
                "volume": float(observed["volume"].sum()),
            }
        daily = DailyPanel.from_records(
            symbol, rows, snapshot_hash=f"fixture-{self.spec_hash[:16]}-{symbol}-1d"
        )
        self._daily[symbol] = daily
        return daily

    def panel_ending_at(self, symbol: str, origin_session: dt.date, count: int) -> BarPanel:
        """Trailing ``count`` bars ending at ``origin_session``.

        Built by slicing the full generated panel on the calendar's expected
        schedule, so the result is identical whether or not later sessions
        exist in the fixture.
        """
        wanted = self.calendar.bars_ending_at(origin_session, count)
        index = pd.DatetimeIndex([bar.start for bar in wanted])
        full = self.hourly(symbol)
        missing = index.difference(full.frame.index)
        if len(missing):
            raise ValueError(
                f"fixture for {symbol} does not cover {len(missing)} requested bars; "
                f"first missing {missing[0].isoformat()}"
            )
        observations = full.frame.loc[index, ["open", "high", "low", "close", "volume"]]
        observed_mask = full.frame.loc[index, "observed"].astype(bool)
        observations = observations.loc[observed_mask.to_numpy()]
        return BarPanel.aligned(
            symbol, "1h", wanted, observations, snapshot_hash=full.snapshot_hash
        )

    def drop_bars(self, symbol: str, timestamps: Iterable[pd.Timestamp]) -> None:
        """Mark specific bars unobserved, to exercise the masking paths."""
        panel = self.hourly(symbol)
        frame = panel.frame.copy()
        for timestamp in timestamps:
            key = pd.Timestamp(timestamp)
            if key not in frame.index:
                raise KeyError(f"{symbol}: {key.isoformat()} is not in the fixture panel")
            frame.loc[key, ["open", "high", "low", "close", "volume"]] = np.nan
            frame.loc[key, "observed"] = False
        self._panels[symbol] = BarPanel(
            symbol=panel.symbol,
            interval=panel.interval,
            bars=panel.bars,
            frame=frame,
            snapshot_hash=panel.snapshot_hash + "-masked",
        )
        self._daily.pop(symbol, None)

    def action_ledger(self, symbol: str) -> ActionLedger:
        return ActionLedger(symbol=symbol, actions=self.spec.actions.get(symbol, ()))

    def truncated(self, last_session: dt.date) -> "SyntheticMarket":
        """A market generated only up to ``last_session``.

        Used by the leakage tests: a truncated fixture must reproduce the
        earlier bars of the full fixture exactly, so any difference in a feature
        is caused by the pipeline rather than by the data generator.
        """
        # replace() rather than a hand-written field list: a field added to the
        # spec later cannot be silently dropped from the truncated copy.
        return SyntheticMarket(
            self.calendar, dataclasses.replace(self.spec, end=last_session)
        )


def make_default_market(
    calendar: ExchangeCalendar,
    *,
    start: dt.date,
    end: dt.date,
    symbols: Sequence[str] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"),
    seed: int = 20260927,
) -> SyntheticMarket:
    """A small three-sector synthetic universe for tests and demos."""
    sector_labels = ("alpha", "beta", "gamma")
    sectors = {
        symbol: sector_labels[index % len(sector_labels)]
        for index, symbol in enumerate(symbols)
    }
    sector_etfs = {label: f"XL{label[0].upper()}" for label in sector_labels}
    spec = SyntheticSpec(
        symbols=tuple(symbols),
        sectors=sectors,
        market_proxy="SPY",
        sector_etfs=sector_etfs,
        start=start,
        end=end,
        seed=seed,
        base_price={symbol: 50.0 + 15.0 * index for index, symbol in enumerate(symbols)},
    )
    return SyntheticMarket(calendar, spec)
