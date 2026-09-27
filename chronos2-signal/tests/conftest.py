"""Shared fixtures: a synthetic universe, a frozen watchlist and a pipeline.

Everything here is offline and deterministic. No network, no checkpoint, no
performance meaning.
"""

from __future__ import annotations

import datetime as dt

import pytest

from chronos2_signal.actions import CorporateAction
from chronos2_signal.calendar_spec import ExchangeCalendar
from chronos2_signal.config import load_design
from chronos2_signal.fixtures import SyntheticMarket, SyntheticSpec
from chronos2_signal.forecaster import DeterministicStubForecaster
from chronos2_signal.pipeline import FixtureMarketSource, ResearchPipeline
from chronos2_signal.universe import Candidate, CandidateRoster, select_watchlist

# A window long enough to satisfy the registered eligibility mask (512
# scheduled hourly bars and 200 valid daily observations) with room for
# development and scoring blocks on top.
FIXTURE_START = dt.date(2023, 6, 1)
FIXTURE_END = dt.date(2025, 12, 19)

SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
SECTOR_LABELS = ("alpha", "beta", "gamma")


@pytest.fixture(scope="session")
def config():
    return load_design()


@pytest.fixture(scope="session")
def calendar(config):
    return ExchangeCalendar(config.runtime.exchange_calendar)


@pytest.fixture(scope="session")
def synthetic_spec():
    sectors = {
        symbol: SECTOR_LABELS[index % len(SECTOR_LABELS)]
        for index, symbol in enumerate(SYMBOLS)
    }
    sector_etfs = {label: f"XL{label[0].upper()}" for label in SECTOR_LABELS}
    return SyntheticSpec(
        symbols=SYMBOLS,
        sectors=sectors,
        market_proxy="SPY",
        sector_etfs=sector_etfs,
        start=FIXTURE_START,
        end=FIXTURE_END,
        seed=424242,
        base_price={symbol: 40.0 + 11.0 * index for index, symbol in enumerate(SYMBOLS)},
        # A mild drift spread so the decision heads see a non-degenerate
        # cross-section. It is a fixture property, not a market claim.
        drift_bps_per_bar={
            symbol: (0.6 if index % 3 == 0 else -0.3) for index, symbol in enumerate(SYMBOLS)
        },
    )


@pytest.fixture(scope="session")
def market(calendar, synthetic_spec):
    return SyntheticMarket(calendar, synthetic_spec)


@pytest.fixture(scope="session")
def roster(synthetic_spec):
    candidates = tuple(
        Candidate(
            symbol=symbol,
            issuer_id=f"ISSUER-{symbol}",
            exchange="XNYS",
            sector=synthetic_spec.sectors[symbol],
            sector_etf=synthetic_spec.sector_etfs[synthetic_spec.sectors[symbol]],
            listing_history=f"{symbol} listed before the fixture window",
        )
        for symbol in synthetic_spec.symbols
    )
    return CandidateRoster(
        candidates=candidates,
        source="tests/conftest.py synthetic roster",
        declared_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )


@pytest.fixture(scope="session")
def watchlist(roster, config, market):
    selection_session = market.sessions[-1]
    dollar_volume = {}
    for symbol in roster.by_symbol():
        daily = market.daily(symbol).up_to(selection_session)
        closes = daily.column("close")[-20:]
        volumes = daily.column("volume")[-20:]
        dollar_volume[symbol] = float((closes * volumes).mean())
    return select_watchlist(
        roster,
        config=config,
        selection_session=selection_session,
        dollar_volume=dollar_volume,
        frozen_at=dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc),
    )


@pytest.fixture(scope="session")
def stub_forecaster(config):
    return DeterministicStubForecaster(
        quantile_levels=config.model.quantile_levels,
        batch_size_channels=config.model.batch_size_channels,
    )


@pytest.fixture
def pipeline(config, calendar, market, watchlist, stub_forecaster):
    return ResearchPipeline(
        config=config,
        calendar=calendar,
        source=FixtureMarketSource(market),
        watchlist=watchlist,
        forecaster=stub_forecaster,
    )


@pytest.fixture(scope="session")
def split_dividend_market(calendar, synthetic_spec):
    """A market with a known split and dividend inside the fixture window."""
    actions = {
        "AAA": (
            CorporateAction(
                symbol="AAA",
                ex_date=dt.date(2025, 6, 12),
                action_type="split",
                value=2.0,
                available_at=dt.datetime(2025, 6, 12, tzinfo=dt.timezone.utc),
            ),
            CorporateAction(
                symbol="AAA",
                ex_date=dt.date(2025, 9, 11),
                action_type="dividend",
                value=0.35,
                available_at=dt.datetime(2025, 9, 11, tzinfo=dt.timezone.utc),
            ),
        )
    }
    spec = SyntheticSpec(
        symbols=synthetic_spec.symbols,
        sectors=synthetic_spec.sectors,
        market_proxy=synthetic_spec.market_proxy,
        sector_etfs=synthetic_spec.sector_etfs,
        start=synthetic_spec.start,
        end=synthetic_spec.end,
        seed=synthetic_spec.seed,
        base_price=synthetic_spec.base_price,
        drift_bps_per_bar=synthetic_spec.drift_bps_per_bar,
        actions=actions,
    )
    return SyntheticMarket(calendar, spec)
