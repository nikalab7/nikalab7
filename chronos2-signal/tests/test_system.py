"""System-level tests: every guarantee checked through the pipeline.

The invariant suite once proved that the *functions* were correct while the
orchestration never called several of them -- the split audit, the holdout
guard on the research path, the forecast ledger, the ledger-backed data source.
Every test here drives :meth:`ResearchPipeline.build_batch`,
:func:`~chronos2_signal.simulation.simulate` or an operations entry point, so a
guarantee that is not wired in fails here even if its component works.

Nothing in this file is a performance result: the data is a seeded random walk
and the forecaster is an arithmetic fixture.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from chronos2_signal.actions import CorporateAction
from chronos2_signal.collector import FetchRequest, FixtureCollector, Ingestor
from chronos2_signal.decision import OriginRow, column_set_for_variant, fit_decision_model
from chronos2_signal.features import build_chronos_task, recommended_panel_bars
from chronos2_signal.fixtures import SyntheticMarket
from chronos2_signal.forecast_store import ForecastCache
from chronos2_signal.forecaster import Chronos2Forecaster, DeterministicStubForecaster
from chronos2_signal.holdout import AccessMode, HoldoutViolation
from chronos2_signal.market import DailyPanel, aligned_log_returns
from chronos2_signal.notifier import Notifier, RecordingChannel
from chronos2_signal.operations import (
    OperationsError,
    _verify_holdout,
    render_comparison_markdown,
    run_after_close,
    run_development_comparison,
    run_final_test,
    run_study,
)
from chronos2_signal.pipeline import ResearchPipeline
from chronos2_signal.portfolio import EntryRequest, ReferencePortfolio
from chronos2_signal.protocol import build_schedule, label_available_at
from chronos2_signal.simulation import (
    ModelSchedule,
    Selection,
    SelectionOutcome,
    WalkForwardRunner,
    _mark_prices,
    simulate,
)
from chronos2_signal.sources import FixtureMarketSource, LedgerMarketSource, Vintage
from chronos2_signal.storage import Ledger, SnapshotStore

ORIGIN = dt.date(2025, 11, 25)
DATA_EDGE = dt.date(2025, 12, 19)
DIAGNOSTIC_EARLIEST = dt.date(2024, 7, 1)
#: An ex-date inside ORIGIN's feature window (the 512-bar panel reaches back to
#: August 2025).
SPLIT_IN_WINDOW = dt.date(2025, 11, 12)
#: An ex-date long before that window.
SPLIT_OUTSIDE_WINDOW = dt.date(2025, 3, 12)


def _pipeline(config, calendar, market, watchlist, forecaster, **overrides):
    return ResearchPipeline(
        config=config,
        calendar=calendar,
        source=FixtureMarketSource(market),
        watchlist=watchlist,
        forecaster=forecaster,
        **overrides,
    )


def _split(ex_date: dt.date, ratio: float = 2.0) -> CorporateAction:
    return CorporateAction(symbol="AAA", ex_date=ex_date, action_type="split", value=ratio)


def _market_with(calendar, spec, **changes) -> SyntheticMarket:
    return SyntheticMarket(calendar, dataclasses.replace(spec, **changes))


def _permissive(config):
    """Relaxed alert thresholds, to reach the alert path. Plumbing only."""
    return dataclasses.replace(
        config,
        status="plumbing_test_only_not_registered",
        decision=dataclasses.replace(
            config.decision,
            min_probability=0.01,
            min_estimated_net_return=1e-9,
            require_positive_stress_estimate=False,
        ),
    )


def _small_blocks(config, **validation):
    """Registered structure at a scale a test can run. Plumbing only."""
    settings = {
        "min_fit_origin_sessions": 20,
        "calibration_origin_sessions": 10,
        "validation_origin_sessions": 10,
        "final_historical_test_origin_sessions": 10,
        "decision_refit_every_sessions": 5,
        "development_min_executed_trades": 1,
        "development_min_signal_sessions": 1,
    }
    settings.update(validation)
    return dataclasses.replace(
        _permissive(config), validation=dataclasses.replace(config.validation, **settings)
    )


def _quick_model(config, variant="C256"):
    """A fitted model from synthetic rows -- enough to drive the scoring path."""
    rng = np.random.default_rng(11)
    columns = column_set_for_variant(variant)
    sessions = [dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(200)]
    sessions = [session for session in sessions if session.weekday() < 5]

    def rows(block):
        return [
            OriginRow(
                origin_session=session,
                symbol=symbol,
                features={name: float(rng.normal()) for name in columns},
                sigma_2d=0.02,
                r_net=float(rng.normal(scale=0.01)),
                eligible_count=4,
            )
            for session in block
            for symbol in ("A", "B", "C", "D")
        ]

    return fit_decision_model(
        variant=variant,
        fit_rows=rows(sessions[:80]),
        calibration_rows=rows(sessions[82:112]),
        config=config,
    )


# --------------------------------------------------------------------------- #
# 1. The split contract, through build_batch and label
# --------------------------------------------------------------------------- #


def test_unrestated_split_is_restated_before_any_feature_sees_it(
    calendar, config, synthetic_spec, watchlist, stub_forecaster
):
    """A provider that did not restate a split yields the same features as one that did."""
    restated = _market_with(calendar, synthetic_spec, actions={"AAA": (_split(SPLIT_IN_WINDOW),)})
    raw = _market_with(
        calendar,
        synthetic_spec,
        actions={"AAA": (_split(SPLIT_IN_WINDOW),)},
        emitted_price_jumps={"AAA": ((SPLIT_IN_WINDOW, 2.0),)},
    )
    # The raw series really does carry the pre-split level.
    raw_closes = raw.daily("AAA").frame["close"]
    before = raw_closes.loc[[s for s in raw_closes.index if s < SPLIT_IN_WINDOW][-1]]
    after = raw_closes.loc[SPLIT_IN_WINDOW]
    assert before / after == pytest.approx(2.0, rel=0.05)

    raw_source = FixtureMarketSource(raw)
    audits = raw_source.view(ORIGIN).split_audits("AAA")
    assert [audit.verdict.value for audit in audits] == ["not_applied"]
    assert [audit.verdict.value for audit in FixtureMarketSource(restated).view(ORIGIN).split_audits("AAA")] == [
        "already_applied"
    ]

    def features(market):
        batch = _pipeline(config, calendar, market, watchlist, stub_forecaster).build_batch(
            ORIGIN, "C256"
        )
        return {score.symbol: score for score in batch.scores}

    raw_scores, restated_scores = features(raw), features(restated)
    assert raw_scores["AAA"].eligibility.eligible, raw_scores["AAA"].eligibility.failures
    for name, value in restated_scores["AAA"].features.items():
        other = raw_scores["AAA"].features[name]
        assert (math.isnan(value) and math.isnan(other)) or value == pytest.approx(
            other, rel=1e-9, abs=1e-12
        ), name


def test_unresolvable_split_quarantines_the_origin_and_the_label(
    calendar, config, synthetic_spec, watchlist, stub_forecaster
):
    """An ambiguous episode makes the symbol ineligible and its label unavailable."""
    ambiguous = _market_with(
        calendar,
        synthetic_spec,
        actions={"AAA": (_split(SPLIT_IN_WINDOW),)},
        emitted_price_jumps={"AAA": ((SPLIT_IN_WINDOW, 1.5),)},
    )
    pipeline = _pipeline(config, calendar, ambiguous, watchlist, stub_forecaster)
    batch = pipeline.build_batch(ORIGIN, "C256")
    aaa = next(score for score in batch.scores if score.symbol == "AAA")
    assert not aaa.eligibility.eligible
    assert any("corporate_action" in failure for failure in aaa.eligibility.failures)
    # Other symbols are unaffected.
    assert next(score for score in batch.scores if score.symbol == "BBB").eligibility.eligible

    # A holding window containing the ex-date cannot be labelled honestly.
    spanning = calendar.previous_sessions(SPLIT_IN_WINDOW, 2)[0]
    timing = label_available_at(calendar, spanning, horizon_sessions=2)
    assert timing.entry_session < SPLIT_IN_WINDOW <= timing.exit_session
    assert pipeline.label(spanning, "AAA") is None
    assert pipeline.label(spanning, "BBB") is not None
    # A window that does not contain it is labelled normally.
    assert pipeline.label(calendar.next_session(SPLIT_IN_WINDOW, 2), "AAA") is not None


def test_split_outside_the_feature_window_does_not_quarantine(
    calendar, config, synthetic_spec, watchlist, stub_forecaster
):
    old = _market_with(
        calendar,
        synthetic_spec,
        actions={"AAA": (_split(SPLIT_OUTSIDE_WINDOW),)},
        emitted_price_jumps={"AAA": ((SPLIT_OUTSIDE_WINDOW, 1.5),)},
    )
    pipeline = _pipeline(config, calendar, old, watchlist, stub_forecaster)
    audits = pipeline.source.view(ORIGIN).split_audits("AAA")
    assert audits and not audits[0].usable  # unresolved, but irrelevant here
    aaa = next(score for score in pipeline.build_batch(ORIGIN, "C256").scores if score.symbol == "AAA")
    assert aaa.eligibility.eligible, aaa.eligibility.failures


# --------------------------------------------------------------------------- #
# 2. The ledger-backed source: the real data path reaches the pipeline
# --------------------------------------------------------------------------- #


def _ingest_fixture(tmp_path, calendar, market, symbols, *, hourly_from):
    """Write a fixture's bars to a ledger through the real collector path."""
    frames = {}
    for symbol in symbols:
        hourly = market.hourly(symbol).frame
        hourly = hourly[hourly["observed"].astype(bool)]
        frames[(symbol, "1h")] = hourly[["open", "high", "low", "close", "volume"]]
        daily = market.daily(symbol).frame[["open", "high", "low", "close", "volume"]].copy()
        daily.index = pd.DatetimeIndex([calendar.session_open(session) for session in daily.index])
        frames[(symbol, "1d")] = daily
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ingestor = Ingestor(
        calendar=calendar,
        ledger=ledger,
        snapshots=SnapshotStore(tmp_path / "snapshots"),
        collector=FixtureCollector(frames=frames),
    )
    for symbol in symbols:
        assert ingestor.ingest(FetchRequest(symbol=symbol, interval="1h", start=hourly_from)).ok
        assert ingestor.ingest(FetchRequest(symbol=symbol, interval="1d")).ok
    return ledger


def test_ledger_source_reproduces_the_fixture_through_the_pipeline(
    tmp_path, calendar, config, market, watchlist, stub_forecaster
):
    """Collector -> ledger -> LedgerMarketSource -> build_batch equals the fixture."""
    symbols = [*watchlist.symbols, "SPY", *sorted(set(watchlist.sector_etfs().values()))]
    ledger = _ingest_fixture(
        tmp_path, calendar, market, symbols, hourly_from=calendar.previous_sessions(ORIGIN, 120)[0]
    )
    ledger_pipeline = dataclasses.replace(
        _pipeline(config, calendar, market, watchlist, stub_forecaster),
        source=LedgerMarketSource(
            ledger=ledger,
            calendar=calendar,
            vintage=Vintage.LATEST,
            max_data_delay_minutes=config.execution.max_data_delay_after_close_minutes,
        ),
    )
    fixture_batch = _pipeline(config, calendar, market, watchlist, stub_forecaster).build_batch(
        ORIGIN, "C256"
    )
    ledger_batch = ledger_pipeline.build_batch(ORIGIN, "C256")

    assert ledger_batch.eligible_count == fixture_batch.eligible_count > 0
    for fixture_score, ledger_score in zip(fixture_batch.scores, ledger_batch.scores, strict=True):
        assert fixture_score.symbol == ledger_score.symbol
        assert ledger_score.eligibility.eligible == fixture_score.eligibility.eligible
        for name, value in (fixture_score.features or {}).items():
            other = ledger_score.features[name]
            assert (math.isnan(value) and math.isnan(other)) or value == pytest.approx(
                other, rel=1e-12, abs=1e-12
            ), f"{fixture_score.symbol}.{name}"
    # The report says which vintage it was read at, and what that implies.
    assert "limitation" in ledger_pipeline.source.describe()
    ledger.close()


def test_point_in_time_views_exclude_later_revisions(tmp_path, calendar, config):
    """A revision retrieved after an origin's deadline is invisible to that origin."""
    ledger = Ledger(tmp_path / "vintage.sqlite")
    session = ORIGIN
    open_ts, close_ts = calendar.session_window(session)
    for snapshot_id, close, retrieved in (
        ("original", 50.0, close_ts + pd.Timedelta(minutes=30)),
        ("revised", 55.0, close_ts + pd.Timedelta(days=5)),
    ):
        ledger.record_snapshot(
            snapshot_id=snapshot_id,
            provider="fixture",
            provider_version="0",
            symbol="AAA",
            interval="1d",
            requested_start=None,
            requested_end=None,
            retrieved_at=retrieved.to_pydatetime(),
            declared_adjustment_mode="auto_adjust=False",
            row_count=1,
            payload_path=None,
            payload_sha256=snapshot_id,
            request_params={},
            status="ok",
        )
        ledger.record_bars(
            [
                {
                    "symbol": "AAA",
                    "interval": "1d",
                    "bar_start": open_ts,
                    "bar_end": close_ts,
                    "session": session,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1e6,
                    "adj_close": None,
                    "observed": True,
                    "snapshot_id": snapshot_id,
                    "retrieved_at": retrieved.to_pydatetime(),
                }
            ]
        )

    def close_seen(vintage):
        source = LedgerMarketSource(
            ledger=ledger,
            calendar=calendar,
            vintage=vintage,
            max_data_delay_minutes=config.execution.max_data_delay_after_close_minutes,
        )
        return float(source.view(session).daily_panel("AAA").row(session)["close"])

    assert close_seen(Vintage.POINT_IN_TIME) == 50.0
    assert close_seen(Vintage.LATEST) == 55.0
    ledger.close()


# --------------------------------------------------------------------------- #
# 3. The holdout guard on the research path
# --------------------------------------------------------------------------- #


def test_the_research_path_cannot_touch_reserved_origins(calendar, config, pipeline):
    schedule = build_schedule(
        calendar, config, latest_data_session=DATA_EDGE, earliest_origin=DIAGNOSTIC_EARLIEST
    )
    guarded = dataclasses.replace(pipeline, guard=schedule.guard(AccessMode.DEVELOPMENT))
    reserved = schedule.test_origins[0]

    with pytest.raises(HoldoutViolation):
        guarded.build_batch(reserved, "C256")
    with pytest.raises(HoldoutViolation):
        guarded.label(reserved, "AAA")
    with pytest.raises(HoldoutViolation):
        guarded.realised_log_return(reserved, "AAA")

    cache: dict = {}
    runner = WalkForwardRunner(pipeline=guarded)
    with pytest.raises(HoldoutViolation):
        runner.run(
            "C256",
            model=_quick_model(config),
            score_sessions=[schedule.development_origins[-1], reserved],
            batch_cache=cache,
        )
    # It failed before doing any work: nothing was built, not even the legal origin.
    assert cache == {}


def test_studies_verify_the_holdout_instead_of_asserting_it(
    calendar, config, market, watchlist, stub_forecaster
):
    small = _small_blocks(config)
    schedule = build_schedule(
        calendar, small, latest_data_session=DATA_EDGE, earliest_origin=DIAGNOSTIC_EARLIEST
    )
    study = run_study(
        pipeline=_pipeline(small, calendar, market, watchlist, stub_forecaster),
        schedule=schedule,
        variant="C256",
        checkpoint="holdout-check",
        folds=(0,),
        bootstrap_samples=50,
    )
    assert any(
        note.startswith("holdout verified: 0 of the 10 reserved") for note in study.notes
    ), study.notes

    # The note is the result of a check that can fail: a development run that
    # had touched a reserved origin is rejected, not reported as clean.
    touched = {("C256", schedule.test_origins[0]): None}
    with pytest.raises(OperationsError, match="reserved origin"):
        _verify_holdout(schedule, touched, [], mode=AccessMode.DEVELOPMENT)


# --------------------------------------------------------------------------- #
# 4. One scoring path for the backtest and the live run
# --------------------------------------------------------------------------- #


def test_live_and_backtest_scoring_agree(
    tmp_path, calendar, config, market, watchlist, stub_forecaster
):
    # Every threshold open, so that both paths reach ranking and capacity: the
    # comparison must cover alerted decisions, not only suppressed ones.
    permissive = dataclasses.replace(
        config,
        status="plumbing_test_only_not_registered",
        decision=dataclasses.replace(
            config.decision,
            min_probability=0.0,
            min_estimated_net_return=-1.0,
            require_positive_stress_estimate=False,
        ),
    )
    pipeline = _pipeline(permissive, calendar, market, watchlist, stub_forecaster)
    model = _quick_model(permissive)

    backtest = WalkForwardRunner(pipeline=pipeline).run(
        "C256", model=model, score_sessions=[ORIGIN]
    )
    ledger = Ledger(tmp_path / "live.sqlite")
    live = run_after_close(
        pipeline=pipeline, model=model, ledger=ledger, origin_session=ORIGIN, variant="C256"
    )

    def fingerprint(outcome):
        return [
            (
                decision.symbol,
                decision.alerted,
                decision.rank_position,
                decision.reasons,
                round(decision.candidate.estimate.estimated_net_return, 12),
                round(decision.candidate.estimate.calibrated_probability, 12),
            )
            for decision in outcome.decisions
        ]

    assert fingerprint(backtest.policy_outcomes[0]) == fingerprint(live.outcome)
    assert live.outcome.alert_count > 0  # the comparison covered real alerts
    ledger.close()


# --------------------------------------------------------------------------- #
# 5. The session engine: ordering and marks
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _ScriptedSelector:
    """Always tries the same three symbols; records what it saw."""

    name: str = "scripted"
    seen: dict = dataclasses.field(default_factory=dict)

    def select(self, session, book):
        self.seen[session] = len(book.positions)
        return SelectionOutcome(
            selections=(
                Selection("AAA", "ISSUER-AAA", "alpha"),
                Selection("BBB", "ISSUER-BBB", "beta"),
                Selection("CCC", "ISSUER-CCC", "gamma"),
            )
        )

    def provenance(self):
        return {"forecaster": "none (scripted test selector)"}


@dataclasses.dataclass
class _OneShotSelector:
    """Buys one symbol after one session, and nothing else."""

    session: dt.date
    selection: Selection
    name: str = "one-shot"

    def select(self, session, book):
        return SelectionOutcome(selections=(self.selection,) if session == self.session else ())

    def provenance(self):
        return {"forecaster": "none (scripted test selector)"}


def test_scoring_happens_after_the_close_against_post_exit_state(calendar, pipeline):
    """Positions exiting at a close no longer occupy capacity when that session is scored."""
    sessions = calendar.sessions(dt.date(2025, 11, 3), dt.date(2025, 11, 7))
    selector = _ScriptedSelector()
    result = simulate(pipeline=pipeline, selector=selector, score_sessions=sessions)

    first, second, third = sessions[:3]
    assert selector.seen[first] == 0
    assert selector.seen[second] == 3  # the first cohort is still open
    # The first cohort exits at the third session's close. Scoring after that
    # close must see an empty book; the old order saw three positions here and
    # blocked every new alert until the next day.
    assert selector.seen[third] == 0
    assert result.diagnostics["entry_price_unavailable"] == 0


def test_a_missing_close_is_marked_at_the_last_observed_close(config):
    """Never at the entry fill, which would hide the drawdown since entry."""
    sessions = [dt.date(2025, 6, 9), dt.date(2025, 6, 10), dt.date(2025, 6, 11)]
    panel = DailyPanel.from_records(
        "AAA",
        {
            sessions[0]: {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6},
            sessions[1]: {"open": 90.0, "high": 91.0, "low": 80.0, "close": 82.0, "volume": 1e6},
            sessions[2]: {"open": 81.0, "high": 83.0, "low": 79.0, "close": float("nan"), "volume": 1e6},
        },
    )

    class _View:
        def daily_panel(self, symbol):
            return panel

    book = ReferencePortfolio(config=config, variant="mark-check")
    book.allocate(
        EntryRequest(
            signal_id="s",
            symbol="AAA",
            issuer_id="I-AAA",
            sector="alpha",
            signal_session=sessions[0],
            entry_session=sessions[1],
            planned_exit_session=dt.date(2025, 6, 13),
            reference_open_price=90.0,
        ),
        equity_at_open=100_000.0,
        gross_value_at_open=0.0,
    )
    marks, stale = _mark_prices(_View(), book, sessions[2])
    assert stale == 1
    assert marks["AAA"] == pytest.approx(82.0)
    assert marks["AAA"] != pytest.approx(book.positions[0].entry_fill_price)


def test_a_split_inside_a_hold_changes_units_not_value(
    calendar, config, synthetic_spec, watchlist, stub_forecaster
):
    """A position held across an ex-date is valued in one set of units.

    Every view is split-consistent within itself, but two views need not share
    units: a view before the ex-date of a split the provider left raw quotes
    pre-split prices, and one after it post-split prices. Entering from one and
    exiting from the other must not turn a 2-for-1 split into a 50% loss.
    """
    signal = calendar.previous_sessions(SPLIT_IN_WINDOW, 2)[0]
    timing = label_available_at(calendar, signal, horizon_sessions=2)
    assert timing.entry_session < SPLIT_IN_WINDOW <= timing.exit_session

    runs = {}
    for convention, jumps in (
        ("applied", {}),
        ("not_applied", {"AAA": ((SPLIT_IN_WINDOW, 2.0),)}),
    ):
        market = _market_with(
            calendar,
            synthetic_spec,
            actions={"AAA": (_split(SPLIT_IN_WINDOW),)},
            emitted_price_jumps=jumps,
        )
        pipeline = _pipeline(config, calendar, market, watchlist, stub_forecaster)
        result = simulate(
            pipeline=pipeline,
            selector=_OneShotSelector(signal, Selection("AAA", "ISSUER-AAA", "alpha")),
            score_sessions=[signal],
        )
        runs[convention] = (result, pipeline.label(signal, "AAA"))

    for convention, (result, label) in runs.items():
        (trade,) = result.trades
        assert trade.resolved, convention
        # The realised trade is exactly the registered label, which reads both
        # prices from one view ...
        assert trade.r_net_base == pytest.approx(label.net_return, rel=1e-9, abs=1e-12), convention
        # ... and the split manufactured no excursion.
        assert trade.max_adverse_excursion > -0.2, convention
        (position,) = result.portfolio.closed
        assert position.share_adjustment == pytest.approx(2.0), convention
        assert result.diagnostics["unit_change_unresolved"] == 0, convention

    applied, raw = runs["applied"][0], runs["not_applied"][0]
    assert raw.trades[0].r_net_base == pytest.approx(applied.trades[0].r_net_base, rel=1e-9)
    assert raw.trades[0].r_net_stress == pytest.approx(applied.trades[0].r_net_stress, rel=1e-9)
    np.testing.assert_allclose(raw.daily.returns, applied.daily.returns, rtol=1e-9, atol=1e-12)


@dataclasses.dataclass
class _RequotedEntrySource:
    """Serves ``inner``, except that views after ``after`` re-quote one open."""

    inner: FixtureMarketSource
    symbol: str
    session: dt.date
    after: dt.date
    factor: float

    def view(self, session):
        view = self.inner.view(session)
        return view if session <= self.after else _RequotedView(view, self)

    def describe(self):
        return self.inner.describe()


class _RequotedView:
    def __init__(self, view, spec):
        self._view, self._spec = view, spec
        self.session = view.session

    def __getattr__(self, name):
        return getattr(self._view, name)

    def daily_panel(self, symbol):
        panel = self._view.daily_panel(symbol)
        if symbol != self._spec.symbol:
            return panel
        frame = panel.frame.copy()
        frame.loc[self._spec.session, "open"] *= self._spec.factor
        return DailyPanel(symbol=panel.symbol, frame=frame, snapshot_hash=panel.snapshot_hash)


@pytest.mark.parametrize("factor", [0.97, 0.7])
def test_a_requoted_entry_is_neither_a_split_nor_a_return(calendar, pipeline, factor):
    """A later view that re-quotes the entry bar, with no split recorded."""
    signal = dt.date(2025, 11, 3)
    timing = label_available_at(calendar, signal, horizon_sessions=2)
    selector = _OneShotSelector(signal, Selection("AAA", "ISSUER-AAA", "alpha"))
    plain = simulate(pipeline=pipeline, selector=selector, score_sessions=[signal])
    requoted = simulate(
        pipeline=dataclasses.replace(
            pipeline,
            source=_RequotedEntrySource(
                pipeline.source, "AAA", timing.entry_session, after=timing.entry_session, factor=factor
            ),
        ),
        selector=selector,
        score_sessions=[signal],
    )
    (trade,) = requoted.trades
    assert requoted.diagnostics["unit_restatements"] == 0
    if factor == 0.97:
        # An ordinary revision: the units did not change, and the trade is the
        # one the unrevised data gives.
        assert trade.resolved
        assert trade.r_net_base == pytest.approx(plain.trades[0].r_net_base, rel=1e-12)
        assert requoted.diagnostics["unit_change_unresolved"] == 0
    else:
        # Accepting the measured change would book a 43% gain the data never
        # showed; ignoring it is what turned a split into a 50% loss. With no
        # recorded split to explain it, the exit stays unresolved.
        assert not trade.resolved
        assert requoted.diagnostics["unit_change_unresolved"] == 1
        assert any("unpriced" in note for note in requoted.notes)


# --------------------------------------------------------------------------- #
# 6. Correlation on shared sessions
# --------------------------------------------------------------------------- #


def test_correlation_is_computed_on_shared_sessions(
    calendar, config, synthetic_spec, watchlist, stub_forecaster
):
    gappy = SyntheticMarket(calendar, synthetic_spec)
    missing_session = calendar.previous_sessions(ORIGIN, 10)[0]
    gappy.drop_bars(
        "BBB", [bar.start for bar in gappy.hourly("BBB").bars if bar.session == missing_session]
    )
    pipeline = _pipeline(config, calendar, gappy, watchlist, stub_forecaster)
    view = pipeline.source.view(ORIGIN)
    assert missing_session not in view.daily_panel("BBB").sessions

    window = config.portfolio.correlation_window_sessions
    left, right = aligned_log_returns(view.daily_panel("AAA"), view.daily_panel("BBB"))
    expected = float(np.corrcoef(left[-window:], right[-window:])[0, 1])
    naive = float(
        np.corrcoef(
            view.daily_panel("AAA").log_returns()[-window:],
            view.daily_panel("BBB").log_returns()[-window:],
        )[0, 1]
    )
    lookup = pipeline.correlation_lookup(ORIGIN)
    assert lookup("AAA", "BBB") == pytest.approx(expected)
    assert lookup("AAA", "BBB") != pytest.approx(naive)
    assert lookup("BBB", "AAA") == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 7. Forecasts are recorded before outcomes, and served from the ledger
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _CountingForecaster:
    inner: DeterministicStubForecaster
    calls: int = 0

    def predict(self, tasks):
        self.calls += len(tasks)
        return self.inner.predict(tasks)

    def provenance(self):
        return self.inner.provenance()

    def cache_identity(self):
        return self.inner.cache_identity()


def test_forecasts_are_written_to_the_ledger_at_batch_time(
    tmp_path, calendar, config, market, watchlist
):
    ledger = Ledger(tmp_path / "forecasts.sqlite")
    counting = _CountingForecaster(
        DeterministicStubForecaster(quantile_levels=config.model.quantile_levels)
    )

    def batch(forecaster):
        return _pipeline(
            config, calendar, market, watchlist, forecaster, forecast_cache=ForecastCache(ledger)
        ).build_batch(ORIGIN, "C256")

    first = batch(counting)
    tasks = len([score for score in first.scores if score.forecast is not None])
    assert tasks > 0 and counting.calls == tasks
    # Recorded at batch time -- before any label for this origin was computed.
    assert len(ledger.query("SELECT cache_key FROM forecasts")) == tasks
    assert all(score.forecast_cache_key for score in first.scores if score.forecast is not None)

    second = batch(counting)
    assert counting.calls == tasks  # nothing recomputed
    for a, b in zip(first.scores, second.scores, strict=True):
        if a.forecast is not None:
            np.testing.assert_array_equal(a.forecast.paths, b.forecast.paths)
            assert a.forecast_cache_key == b.forecast_cache_key

    # A different forecaster identity is a different forecast.
    changed = _CountingForecaster(
        DeterministicStubForecaster(quantile_levels=config.model.quantile_levels, band_scale=2.0)
    )
    batch(changed)
    assert changed.calls == tasks
    assert len(ledger.query("SELECT cache_key FROM forecasts")) == 2 * tasks
    ledger.close()


# --------------------------------------------------------------------------- #
# 8. The development comparison and the final test
# --------------------------------------------------------------------------- #


def test_development_comparison_runs_every_system_on_one_date_index(
    calendar, config, market, watchlist, stub_forecaster
):
    small = _small_blocks(config)
    schedule = build_schedule(
        calendar, small, latest_data_session=DATA_EDGE, earliest_origin=DIAGNOSTIC_EARLIEST
    )
    comparison = run_development_comparison(
        pipeline=_pipeline(small, calendar, market, watchlist, stub_forecaster),
        schedule=schedule,
        variants=("C256",),
        folds=(0, 1),
        bootstrap_samples=100,
        notes=("plumbing check on synthetic data",),
    )

    # B0 is always present: it is what the central question is asked against.
    assert set(comparison.systems) == {"B0", "C256", "momentum"}
    sessions = {result.daily.sessions for result in comparison.systems.values()}
    sessions |= {series.sessions for series in comparison.controls.values()}
    assert len(sessions) == 1

    for key in ("C256_minus_B0", "C256_minus_momentum", "C256_minus_C256_spy_matched", "B0_minus_momentum"):
        assert key in comparison.bootstrap, key

    # One portfolio per system across both folds, and one model per fold.
    scored = sorted({s for i in (0, 1) for s in schedule.folds[i].validation_sessions})
    c256 = comparison.systems["C256"]
    assert c256.origins_scanned == len(scored)
    assert len(c256.model_schedule) == 2
    assert len({record.model_label for record in c256.predictions}) == 2

    # Gate-7 inputs now exist, and B0 correctly has no forecast coverage.
    report = comparison.predictive["C256"]
    assert report.rows > 0
    assert math.isfinite(report.brier) and math.isfinite(report.base_rate_brier)
    assert math.isfinite(report.coverage_p10_p90)
    assert math.isnan(comparison.predictive["B0"].coverage_p10_p90)

    selection = comparison.selection
    assert "C256" in selection.eligibility
    assert selection.retained_for_research == (selection.selected or "C256")
    assert any(note.startswith("holdout verified: 0 of the 10 reserved") for note in comparison.notes)
    assert "Selection:" in render_comparison_markdown(comparison)


def test_final_test_replays_the_registered_refit_schedule(
    calendar, config, market, watchlist, stub_forecaster
):
    small = _small_blocks(config)
    schedule = build_schedule(
        calendar, small, latest_data_session=DATA_EDGE, earliest_origin=dt.date(2025, 8, 1)
    )
    assert len(schedule.refit_points) == 2
    result = run_final_test(
        pipeline=_pipeline(small, calendar, market, watchlist, stub_forecaster),
        schedule=schedule,
        variant="C256",
        bootstrap_samples=50,
    )
    assert result.candidate.origins_scanned == len(schedule.test_origins)
    assert len(result.candidate.model_schedule) == 2
    labels = {record.model_label for record in result.candidate.predictions}
    assert labels == {f"from {point.effective_from.isoformat()}" for point in schedule.refit_points}
    assert result.baseline is not None and result.baseline.variant == "B0"
    assert len(result.gates.results) == 8
    assert any(note.startswith("final-test pass: 10 of the 10 reserved") for note in result.notes)


def test_the_dev_test_boundary_is_purged(calendar, config):
    schedule = build_schedule(
        calendar, config, latest_data_session=DATA_EDGE, earliest_origin=DIAGNOSTIC_EARLIEST
    )
    purge = config.validation.purge_sessions_per_boundary
    assert len(schedule.purged_before_test) == purge
    last_dev = label_available_at(calendar, schedule.development_origins[-1], horizon_sessions=2)
    assert last_dev.exit_session < schedule.test_origins[0]
    assert set(schedule.purged_before_test).isdisjoint(schedule.development_origins)
    assert set(schedule.purged_before_test).isdisjoint(schedule.test_origins)
    assert schedule.describe()["purged_before_test"] == purge


# --------------------------------------------------------------------------- #
# 9. The forecaster adapter's input and output mapping
# --------------------------------------------------------------------------- #


def _task(calendar, market, symbol, *, include_covariates=True, context=128):
    bars = max(recommended_panel_bars(context), 512)
    sector = {"AAA": "XLA", "BBB": "XLB"}[symbol]
    return build_chronos_task(
        symbol=symbol,
        origin_session=ORIGIN,
        calendar=calendar,
        stock=market.panel_ending_at(symbol, ORIGIN, bars),
        market=market.panel_ending_at("SPY", ORIGIN, bars),
        sector=market.panel_ending_at(sector, ORIGIN, bars),
        context_length=context,
        include_covariates=include_covariates,
    )


class _FakePipeline:
    """Returns ``(quantiles, mean)`` with one ``(1, horizon, levels)`` array per task.

    The output of each task depends on its target and on the channels listed in
    ``reads``, so the adapter's mapping can be checked without the checkpoint.
    """

    def __init__(self, reads):
        self.reads = set(reads)

    def predict_quantiles(self, inputs, quantile_levels, prediction_length, cross_learning):
        quantiles, means = [], []
        for item in inputs:
            base = float(np.nanmean(item["target"][-8:]))
            if "past_calendar" in self.reads:
                base += float(item["past_covariates"]["cal_weekday_sin"][-1])
            if "future_calendar" in self.reads:
                base += float(item["future_covariates"]["cal_weekday_cos"][0])
            if "past_covariates" in self.reads:
                base += float(np.nansum(item["past_covariates"]["relative_volume_log"]))
            ladder = np.linspace(-1.0, 1.0, len(quantile_levels))
            steps = np.arange(1, prediction_length + 1, dtype=float)[:, None]
            quantiles.append((base + ladder[None, :] * steps)[None, :, :])
            means.append(np.full((1, prediction_length), -999.0))
        return quantiles, means


def _adapter(config, fake):
    forecaster = Chronos2Forecaster(
        model_id=config.model.id,
        revision=config.model.revision,
        quantile_levels=config.model.quantile_levels,
    )
    forecaster._pipeline = fake
    forecaster._api = {
        "prediction_length_parameter": "prediction_length",
        "supports_cross_learning": True,
    }
    return forecaster


def test_known_future_covariates_share_names_across_past_and_future(calendar, config, market):
    forecaster = _adapter(config, _FakePipeline(()))
    payload = forecaster._build_pipeline_input(_task(calendar, market, "AAA"))
    assert set(payload) == {"target", "past_covariates", "future_covariates"}
    assert set(payload["future_covariates"]) <= set(payload["past_covariates"])
    assert len(payload["past_covariates"]) == 6 + 5
    target_only = forecaster._build_pipeline_input(
        _task(calendar, market, "AAA", include_covariates=False)
    )
    assert len(target_only["past_covariates"]) == 5  # calendar history only


def test_two_task_batch_is_not_read_as_quantiles_and_mean(calendar, config, market):
    """The ``(quantiles, mean)`` pair must not be mistaken for two tasks."""
    forecaster = _adapter(config, _FakePipeline(()))
    tasks = [_task(calendar, market, "AAA"), _task(calendar, market, "BBB")]
    results = forecaster.predict(tasks)
    assert [result.symbol for result in results] == ["AAA", "BBB"]
    for task, result in zip(tasks, results, strict=True):
        assert result.usable, result.detail
        assert result.paths.shape == (len(config.model.quantile_levels), task.prediction_length)
        expected = float(np.nanmean(task.target[-8:]))
        assert result.forecast_median[0] == pytest.approx(expected)
        assert not np.any(result.paths == -999.0)  # the mean never leaks in


def test_channel_usage_check_detects_a_dropped_channel(calendar, config, market):
    task = _task(calendar, market, "AAA")
    everything = _adapter(config, _FakePipeline(("past_calendar", "future_calendar", "past_covariates")))
    assert everything.channel_usage_check(task) == {
        "baseline_status": "ok",
        "past_calendar_used": True,
        "future_calendar_used": True,
        "past_covariates_used": True,
    }
    missing = _adapter(config, _FakePipeline(("past_calendar", "future_calendar")))
    assert missing.channel_usage_check(task)["past_covariates_used"] is False


# --------------------------------------------------------------------------- #
# 10. The after-close run: transactions and the zero-alert notice
# --------------------------------------------------------------------------- #


def test_a_failure_mid_write_leaves_no_partial_origin(
    tmp_path, calendar, config, market, watchlist, stub_forecaster, monkeypatch
):
    ledger = Ledger(tmp_path / "atomic.sqlite")
    pipeline = _pipeline(config, calendar, market, watchlist, stub_forecaster)
    real = ledger.record_signal
    calls = {"count": 0}

    def flaky(**fields):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("disk full")
        return real(**fields)

    monkeypatch.setattr(ledger, "record_signal", flaky)
    with pytest.raises(RuntimeError):
        run_after_close(
            pipeline=pipeline,
            model=_quick_model(config),
            ledger=ledger,
            origin_session=ORIGIN,
            variant="C256",
        )
    assert calls["count"] == 2
    assert ledger.query("SELECT * FROM signals") == []
    assert [row["status"] for row in ledger.query("SELECT status FROM run_log")] == ["failed"]
    ledger.close()


def test_a_retried_quiet_origin_sends_one_notice(
    tmp_path, calendar, config, market, watchlist, stub_forecaster
):
    """Zero alerts is a result, and it is reported exactly once."""
    ledger = Ledger(tmp_path / "quiet.sqlite")
    channel = RecordingChannel()
    pipeline = _pipeline(config, calendar, market, watchlist, stub_forecaster)
    model = _quick_model(config)
    for _attempt in range(2):
        result = run_after_close(
            pipeline=pipeline,
            model=model,
            ledger=ledger,
            origin_session=ORIGIN,
            variant="C256",
            notifier=Notifier(ledger=ledger, channel=channel),
        )
        assert result.outcome.alert_count == 0
    assert len(channel.messages) == 1
    assert "no alerts" in channel.messages[0][0]
    ledger.close()


def test_a_model_schedule_rejects_an_origin_before_its_first_fit(config):
    schedule = ModelSchedule(entries=((dt.date(2025, 6, 2), _quick_model(config)),))
    with pytest.raises(Exception) as excinfo:
        schedule.model_for(dt.date(2025, 5, 30))
    assert "no fitted model is in force" in str(excinfo.value)
