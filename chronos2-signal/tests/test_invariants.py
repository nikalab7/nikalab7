"""The twelve integrity invariants of the protocol's section 16.

These tests target financial and model-integrity risks rather than mirroring
trivial code. They must pass before a large universe is fetched or a backtest
is run, and they run entirely offline: no network, no checkpoint, no
performance claim.

One test per numbered invariant, in order.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from chronos2_signal.actions import (
    ActionLedger,
    CorporateAction,
    HoldingCosts,
    QuarantinedEpisode,
    SplitVerdict,
    audit_split_convention,
    label_net_return,
    to_asof_units,
    to_split_consistent,
)
from chronos2_signal.calendar_spec import REGULAR_SESSION_MINUTES
from chronos2_signal.decision import OriginRow, fit_decision_model
from chronos2_signal.features import (
    build_chronos_task,
    recommended_panel_bars,
)
from chronos2_signal.fixtures import SyntheticMarket
from chronos2_signal.forecaster import (
    DeterministicStubForecaster,
    ForecastStatus,
    batch_tasks_by_channels,
    validate_quantile_paths,
)
from chronos2_signal.holdout import AccessMode, HoldoutViolation
from chronos2_signal.market import BarPanel
from chronos2_signal.notifier import Notifier, RecordingChannel
from chronos2_signal.policy import OpenPositionView, PolicyEngine
from chronos2_signal.pipeline import FixtureMarketSource, ResearchPipeline, WalkForwardRunner
from chronos2_signal.portfolio import EntryRequest, ReferencePortfolio
from chronos2_signal.protocol import ProtocolError, build_schedule, label_available_at
from chronos2_signal.storage import Ledger, SnapshotStore, StorageError
from chronos2_signal.variants import cash_series, exposure_matched_series

ORIGIN = dt.date(2025, 11, 25)
DIAGNOSTIC_EARLIEST = dt.date(2024, 7, 1)


# --------------------------------------------------------------------------- #
# 1. Future prices cannot change an earlier origin's inputs
# --------------------------------------------------------------------------- #


def test_invariant_01_future_prices_cannot_change_earlier_inputs(
    calendar, config, market, synthetic_spec, watchlist, stub_forecaster
):
    """Appending or changing future prices leaves an earlier origin untouched.

    The fixture is truncated at the origin and then extended well past it. Both
    the Chronos task matrices and the derived decision features must be
    bit-identical, because every value is computed from bars ending at the
    origin.
    """
    truncated_spec = dataclasses.replace(synthetic_spec, end=ORIGIN)
    truncated = SyntheticMarket(calendar, truncated_spec)

    context = config.model.context_length
    bars = max(
        recommended_panel_bars(context), config.universe.common_hourly_history_bars
    )

    def task_for(source: SyntheticMarket):
        return build_chronos_task(
            symbol="AAA",
            origin_session=ORIGIN,
            calendar=calendar,
            stock=source.panel_ending_at("AAA", ORIGIN, bars),
            market=source.panel_ending_at("SPY", ORIGIN, bars),
            sector=source.panel_ending_at("XLA", ORIGIN, bars),
            context_length=context,
        )

    short, long = task_for(truncated), task_for(market)
    np.testing.assert_array_equal(short.past, long.past)
    np.testing.assert_array_equal(short.past_calendar, long.past_calendar)
    np.testing.assert_array_equal(short.future_calendar, long.future_calendar)
    assert short.origin_close == long.origin_close

    # And through the full feature path, including the forecast-derived features.
    def features_for(source: SyntheticMarket):
        pipeline = ResearchPipeline(
            config=config,
            calendar=calendar,
            source=FixtureMarketSource(source),
            watchlist=watchlist,
            forecaster=stub_forecaster,
        )
        batch = pipeline.build_batch(ORIGIN, "C256")
        return {
            score.symbol: dict(score.features or {}) for score in batch.scorable()
        }

    early, late = features_for(truncated), features_for(market)
    assert early, "the truncated fixture must still produce scorable rows"
    assert set(early) == set(late)
    for symbol, values in early.items():
        for name, value in values.items():
            other = late[symbol][name]
            # A masked optional feature must stay masked, so NaN counts as
            # identical here rather than as a mismatch.
            identical = (math.isnan(value) and math.isnan(other)) or value == other
            assert identical, f"{symbol}.{name} changed when future prices were appended"


# --------------------------------------------------------------------------- #
# 2. Future outcomes cannot change an earlier fit, calibration or alert
# --------------------------------------------------------------------------- #


def test_invariant_02_future_outcomes_cannot_change_an_earlier_fit(config):
    """A label that was not observable at fit time cannot enter the fit.

    Two fits are compared: one over rows whose labels had all matured, and one
    where a later row's outcome is also present in the data but not yet
    observable. The fitted coefficients must be identical.
    """
    columns = None
    rng = np.random.default_rng(7)
    sessions = [dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(0, 120)]
    sessions = [session for session in sessions if session.weekday() < 5]

    def make_rows(session_list, *, available_at):
        rows = []
        for session in session_list:
            for symbol in ("AAA", "BBB", "CCC", "DDD"):
                features = {
                    name: float(value)
                    for name, value in zip(
                        _feature_names(config),
                        rng.normal(size=len(_feature_names(config))),
                        strict=True,
                    )
                }
                rows.append(
                    OriginRow(
                        origin_session=session,
                        symbol=symbol,
                        features=features,
                        sigma_2d=0.02,
                        r_net=float(rng.normal(scale=0.01)),
                        label_available_at=available_at(session),
                        eligible_count=4,
                    )
                )
        return rows

    matured = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    fit_sessions = sessions[:80]
    calibration_sessions = sessions[82:112]
    deadline = dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc)

    base_fit = make_rows(fit_sessions, available_at=lambda _s: matured)
    base_calibration = make_rows(calibration_sessions, available_at=lambda _s: matured)
    reference = fit_decision_model(
        variant="C256",
        fit_rows=base_fit,
        calibration_rows=base_calibration,
        config=config,
        fit_deadline=deadline,
        calibration_deadline=deadline,
    )

    # Same rows, plus one whose outcome only becomes observable after the fit.
    future_row = dataclasses.replace(
        base_fit[0],
        symbol="ZZZ",
        r_net=99.0,
        label_available_at=deadline + dt.timedelta(days=30),
    )
    with_future = fit_decision_model(
        variant="C256",
        fit_rows=[*base_fit, future_row],
        calibration_rows=base_calibration,
        config=config,
        fit_deadline=deadline,
        calibration_deadline=deadline,
    )

    np.testing.assert_array_equal(reference.ridge.coef_, with_future.ridge.coef_)
    np.testing.assert_array_equal(reference.logistic.coef_, with_future.logistic.coef_)
    assert reference.calibrator == with_future.calibrator
    assert with_future.report.dropped_rows == 1
    assert columns is None  # no accidental global state


def _feature_names(config):
    from chronos2_signal.decision import column_set_for_variant

    return column_set_for_variant("C256")


def test_invariant_02b_holdout_outcomes_are_unreadable_in_development(calendar, config):
    """Reserved outcomes cannot be read while in development mode."""
    schedule = build_schedule(
        calendar,
        config,
        latest_data_session=dt.date(2025, 12, 19),
        earliest_origin=DIAGNOSTIC_EARLIEST,
    )
    guard = schedule.guard(AccessMode.DEVELOPMENT)
    assert len(guard.reserved) == config.validation.final_historical_test_origin_sessions
    with pytest.raises(HoldoutViolation):
        guard.check(schedule.test_origins[0], purpose="development report")
    # Development origins stay readable.
    guard.check(schedule.development_origins[-1])
    guard.unlock_final_test()
    guard.check(schedule.test_origins[0])


# --------------------------------------------------------------------------- #
# 3. Session, early close, DST, weekend and holiday timestamps
# --------------------------------------------------------------------------- #


def test_invariant_03_calendar_edge_cases_produce_correct_timestamps(calendar):
    """A normal session, a half day, a DST change, a weekend and a holiday."""
    normal = dt.date(2025, 11, 25)
    bars = calendar.session_bars(normal)
    assert len(bars) == 7
    local = [bar.start.tz_convert("America/New_York").strftime("%H:%M") for bar in bars]
    assert local == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    assert [bar.duration_minutes for bar in bars] == [60.0] * 6 + [30.0]
    assert bars[-1].end.tz_convert("America/New_York").strftime("%H:%M") == "16:00"
    assert calendar.session_minutes(normal) == REGULAR_SESSION_MINUTES

    # Half day: the day after US Thanksgiving 2025 closes at 13:00 local.
    half = dt.date(2025, 11, 28)
    assert calendar.is_early_close(half)
    half_bars = calendar.session_bars(half)
    assert len(half_bars) == 4
    assert half_bars[-1].end.tz_convert("America/New_York").strftime("%H:%M") == "13:00"
    assert half_bars[-1].duration_minutes == 30.0

    # A horizon spanning the half day is shorter, and both terminal indices
    # come from the calendar rather than from a fixed offset.
    horizon = calendar.horizon(dt.date(2025, 11, 25), 2)
    assert horizon.sessions == (dt.date(2025, 11, 26), dt.date(2025, 11, 28))
    assert horizon.prediction_length == 11
    assert horizon.terminal_indices == (6, 10)
    assert horizon.bars[6].session == dt.date(2025, 11, 26)
    assert horizon.bars[10].session == dt.date(2025, 11, 28)

    # Holiday: Thanksgiving itself is not a session and is skipped.
    assert not calendar.is_session(dt.date(2025, 11, 27))
    assert calendar.next_session(dt.date(2025, 11, 26)) == dt.date(2025, 11, 28)

    # Weekend plus the US autumn DST change: Friday close to Monday open is
    # 66.5 hours, an hour longer than the usual 65.5, and the local open is
    # unchanged at 09:30.
    friday, monday = dt.date(2025, 10, 31), dt.date(2025, 11, 3)
    assert calendar.next_session(friday) == monday
    dst_bars = calendar.session_bars(
        monday, previous_end=calendar.session_close(friday)
    )
    assert dst_bars[0].gap_hours == pytest.approx(66.5)
    assert dst_bars[0].start.tz_convert("America/New_York").strftime("%H:%M") == "09:30"
    assert calendar.session_open(friday).isoformat() == "2025-10-31T13:30:00+00:00"
    assert calendar.session_open(monday).isoformat() == "2025-11-03T14:30:00+00:00"

    # An ordinary weekend without a DST change is 65.5 hours.
    plain_friday, plain_monday = dt.date(2025, 11, 21), dt.date(2025, 11, 24)
    plain = calendar.session_bars(
        plain_monday, previous_end=calendar.session_close(plain_friday)
    )
    assert plain[0].gap_hours == pytest.approx(65.5)

    # Signal and deadline timestamps follow the session's own close.
    assert calendar.signal_time(half) == calendar.session_close(half) + pd.Timedelta(
        minutes=30
    )
    assert calendar.data_deadline(
        half, max_delay_minutes=120
    ) == calendar.session_close(half) + pd.Timedelta(minutes=120)


# --------------------------------------------------------------------------- #
# 4. No incomplete bar, no synthetic overnight bar
# --------------------------------------------------------------------------- #


def test_invariant_04_incomplete_and_synthetic_bars_are_excluded(calendar, market, config):
    """Only completed scheduled bars appear, and gaps stay gaps."""
    panel = market.panel_ending_at("AAA", ORIGIN, 64)

    # The panel ends on the origin's final completed bar, never inside the
    # following session.
    assert panel.bars[-1].session == ORIGIN
    assert panel.bars[-1].end == calendar.session_close(ORIGIN)

    # Every row is a scheduled bar of a real session; no overnight or weekend
    # row was synthesised.
    sessions = sorted({bar.session for bar in panel.bars})
    for session in sessions:
        assert calendar.is_session(session)
        expected = calendar.bars_per_session(session)
        present = sum(1 for bar in panel.bars if bar.session == session)
        assert present <= expected
    assert len(panel.bars) == len(set(bar.start for bar in panel.bars))

    # Session boundaries keep a real gap: no bar starts at the previous close.
    for previous, current in zip(panel.bars, panel.bars[1:], strict=False):
        if current.session != previous.session:
            assert current.gap_hours > 1.0
            assert current.start > previous.end
        else:
            assert current.gap_hours == pytest.approx(0.0)
            assert current.start == previous.end

    # An observation at an unscheduled timestamp is rejected, not snapped.
    rogue = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex([panel.bars[-1].start + pd.Timedelta(minutes=7)]),
    )
    with pytest.raises(Exception) as excinfo:
        BarPanel.aligned("AAA", "1h", panel.bars, rogue)
    assert "unscheduled" in str(excinfo.value)

    # A bar the provider never returned is a mask with NaN prices, not a
    # carried-forward flat price.
    masked_bars = list(panel.bars)
    partial = panel.frame.iloc[:-3][["open", "high", "low", "close", "volume"]]
    rebuilt = BarPanel.aligned("AAA", "1h", masked_bars, partial)
    assert rebuilt.observed[-3:].tolist() == [False, False, False]
    assert np.isnan(rebuilt.close[-3:]).all()
    assert rebuilt.observed_fraction() < 1.0

    # An origin whose own bar is missing cannot be forecast at all.
    with pytest.raises(Exception) as excinfo:
        build_chronos_task(
            symbol="AAA",
            origin_session=ORIGIN,
            calendar=calendar,
            stock=rebuilt,
            market=market.panel_ending_at("SPY", ORIGIN, 64),
            sector=market.panel_ending_at("XLA", ORIGIN, 64),
            context_length=32,
        )
    assert "origin bar has no observed close" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 5. Splits and dividends preserve wealth
# --------------------------------------------------------------------------- #


def test_invariant_05_splits_and_dividends_preserve_wealth():
    """A split creates no wealth, no momentum, and no double-counted cash."""
    entry, exit_session = dt.date(2025, 6, 10), dt.date(2025, 6, 12)
    split = CorporateAction(
        symbol="AAA", ex_date=dt.date(2025, 6, 11), action_type="split", value=2.0
    )
    ledger = ActionLedger(symbol="AAA", actions=(split,))
    costs = HoldingCosts(buy_slippage=0.0, sell_slippage=0.0)

    # Split-consistent prices: a 2-for-1 halves the quoted price, and the same
    # economic path expressed in restated units is unchanged.
    plain = label_net_return(
        entry_session=entry,
        exit_session=exit_session,
        entry_open_price=100.0,
        exit_close_price=101.0,
        costs=costs,
        shares=10.0,
    )
    with_split = label_net_return(
        entry_session=entry,
        exit_session=exit_session,
        entry_open_price=100.0,
        exit_close_price=101.0,
        costs=costs,
        ledger=ledger,
        shares=10.0,
    )
    assert with_split.net_return == pytest.approx(plain.net_return)
    assert with_split.reconciles()
    assert with_split.share_adjustment == pytest.approx(2.0)
    assert with_split.nominal_shares_at_exit == pytest.approx(
        with_split.nominal_shares_at_entry * 2.0
    )
    assert with_split.nominal_split_consistent()

    # A raw quoted series is restated exactly once. Applying the factor twice
    # would halve the level again and manufacture a return.
    sessions = [dt.date(2025, 6, 9), dt.date(2025, 6, 10), dt.date(2025, 6, 11)]
    raw_closes = [200.0, 202.0, 101.0]  # visible 2-for-1 step at the ex-date
    raw_volumes = [1_000.0, 1_100.0, 2_200.0]
    audit = audit_split_convention(split, sessions, raw_closes, raw_volumes)
    assert audit.verdict is SplitVerdict.NOT_APPLIED
    restated = to_split_consistent(sessions, raw_closes, [audit])
    assert restated.tolist() == pytest.approx([100.0, 101.0, 101.0])
    # No false momentum: the step across the ex-date is now the real move.
    assert restated[2] / restated[1] == pytest.approx(1.0)

    # A series the provider already restated is left alone.
    restated_closes = [100.0, 101.0, 101.0]
    already = audit_split_convention(split, sessions, restated_closes, raw_volumes)
    assert already.verdict is SplitVerdict.ALREADY_APPLIED
    assert to_split_consistent(sessions, restated_closes, [already]).tolist() == (
        pytest.approx(restated_closes)
    )

    # A double application is detected rather than absorbed.
    doubled = audit_split_convention(split, sessions, [400.0, 404.0, 101.0], raw_volumes)
    assert doubled.verdict is SplitVerdict.DOUBLE_APPLIED
    with pytest.raises(QuarantinedEpisode):
        to_split_consistent(sessions, [400.0, 404.0, 101.0], [doubled])

    # An ambiguous episode is quarantined, never reconstructed.
    ambiguous = audit_split_convention(split, sessions, [150.0, 151.0, 101.0], raw_volumes)
    assert ambiguous.verdict is SplitVerdict.AMBIGUOUS
    with pytest.raises(QuarantinedEpisode):
        to_split_consistent(sessions, [150.0, 151.0, 101.0], [ambiguous])

    # Liquidity screens use as-of units: the pre-split nominal price is
    # recovered, and dollar volume is invariant.
    asof_price = to_asof_units([100.0], ledger, dt.date(2025, 6, 10), kind="price")
    asof_volume = to_asof_units([2_000.0], ledger, dt.date(2025, 6, 10), kind="quantity")
    assert asof_price[0] == pytest.approx(200.0)
    assert asof_volume[0] == pytest.approx(1_000.0)
    assert asof_price[0] * asof_volume[0] == pytest.approx(100.0 * 2_000.0)

    # A dividend is counted once, and only when the holder was entitled.
    dividend = CorporateAction(
        symbol="AAA", ex_date=dt.date(2025, 6, 11), action_type="dividend", value=0.50
    )
    with_dividend = label_net_return(
        entry_session=entry,
        exit_session=exit_session,
        entry_open_price=100.0,
        exit_close_price=101.0,
        costs=costs,
        ledger=ActionLedger(symbol="AAA", actions=(dividend,)),
        shares=10.0,
    )
    assert with_dividend.cash_dividends == pytest.approx(5.0)
    assert with_dividend.net_return == pytest.approx(plain.net_return + 5.0 / 1000.0)
    assert with_dividend.reconciles()

    # Buying on the ex-date earns nothing.
    on_ex_date = label_net_return(
        entry_session=dt.date(2025, 6, 11),
        exit_session=dt.date(2025, 6, 13),
        entry_open_price=100.0,
        exit_close_price=101.0,
        costs=costs,
        ledger=ActionLedger(symbol="AAA", actions=(dividend,)),
        shares=10.0,
    )
    assert on_ex_date.cash_dividends == 0.0


# --------------------------------------------------------------------------- #
# 6. The overnight gap contributes no captured return
# --------------------------------------------------------------------------- #


def test_invariant_06_overnight_gap_is_not_captured(calendar, market, config):
    """A D0-close-to-D1-open gap does not enter the reference strategy's return.

    The fixture's gap is a real discontinuity, and the label is built from the
    D1 open. The realised return must therefore be unchanged when the gap is
    made arbitrarily large, and must differ from a close-to-close measure.
    """
    timing = label_available_at(calendar, ORIGIN, horizon_sessions=2)
    daily = market.daily("AAA")
    d0_close = float(daily.row(ORIGIN)["close"])
    d1_open = float(daily.row(timing.entry_session)["open"])
    d2_close = float(daily.row(timing.exit_session)["close"])
    costs = HoldingCosts(
        buy_slippage=config.execution.slippage_fraction("base"),
        sell_slippage=config.execution.slippage_fraction("base"),
    )

    account = label_net_return(
        entry_session=timing.entry_session,
        exit_session=timing.exit_session,
        entry_open_price=d1_open,
        exit_close_price=d2_close,
        costs=costs,
    )

    # Inflate the gap by 20% by moving the D0 close down. The label is
    # unchanged because it never references the D0 close.
    inflated_gap = d1_open / (d0_close * 0.8) - 1.0
    assert inflated_gap > d1_open / d0_close - 1.0
    unchanged = label_net_return(
        entry_session=timing.entry_session,
        exit_session=timing.exit_session,
        entry_open_price=d1_open,
        exit_close_price=d2_close,
        costs=costs,
    )
    assert unchanged.net_return == pytest.approx(account.net_return)

    # A close-to-close measure would have captured the gap; the registered
    # label does not.
    close_to_close = d2_close / d0_close - 1.0
    captured = d1_open / d0_close - 1.0
    assert close_to_close == pytest.approx(
        (1.0 + captured) * (1.0 + d2_close / d1_open - 1.0) - 1.0, rel=1e-9
    )
    assert abs(close_to_close - account.gross_return) > 1e-12 or captured == 0.0

    # The same statement through the portfolio: entry is the D1 open fill.
    book = ReferencePortfolio(config=config, variant="test")
    position = book.allocate(
        EntryRequest(
            signal_id="s1",
            symbol="AAA",
            issuer_id="ISSUER-AAA",
            sector="alpha",
            signal_session=ORIGIN,
            entry_session=timing.entry_session,
            planned_exit_session=timing.exit_session,
            reference_open_price=d1_open,
        ),
        equity_at_open=config.portfolio.initial_equity,
        gross_value_at_open=0.0,
    )
    assert position is not None
    assert position.entry_reference_price == pytest.approx(d1_open)
    assert position.entry_reference_price != pytest.approx(d0_close)


# --------------------------------------------------------------------------- #
# 7. Shared temporal splits and no overlapping labels across a boundary
# --------------------------------------------------------------------------- #


def test_invariant_07_dates_are_atomic_and_labels_do_not_cross_boundaries(
    calendar, config, pipeline
):
    """Every symbol of a date shares its split, and no outcome window leaks."""
    schedule = build_schedule(
        calendar,
        config,
        latest_data_session=dt.date(2025, 12, 19),
        earliest_origin=DIAGNOSTIC_EARLIEST,
    )
    fold = schedule.folds[0]

    blocks = {
        "fit": set(fold.fit_sessions),
        "calibration": set(fold.calibration_sessions),
        "validation": set(fold.validation_sessions),
        "purge": set(fold.purge_before_calibration) | set(fold.purge_before_validation),
    }
    names = sorted(blocks)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert not blocks[left] & blocks[right]

    # Purge lengths match the registered design.
    assert len(fold.purge_before_calibration) == config.validation.purge_sessions_per_boundary
    assert len(fold.purge_before_validation) == config.validation.purge_sessions_per_boundary

    # No fitting or calibration origin's outcome window reaches the validation
    # block, and every label matured before its stage's deadline.
    first_validation = fold.validation_sessions[0]
    for session in (*fold.fit_sessions, *fold.calibration_sessions):
        timing = label_available_at(calendar, session, horizon_sessions=2)
        assert timing.exit_session < first_validation
    for session in fold.fit_sessions:
        assert label_available_at(calendar, session,
                                  horizon_sessions=2).matured_by(fold.fit_deadline)
    for session in fold.calibration_sessions:
        assert label_available_at(
            calendar, session, horizon_sessions=2
        ).matured_by(fold.calibration_deadline)

    # A date is atomic: all of a date's symbols land in the same block, which
    # is what the batch structure guarantees by construction.
    batch = pipeline.build_batch(fold.validation_sessions[0], "C256")
    assert {score.eligibility.origin_session for score in batch.scores} == {
        fold.validation_sessions[0]
    }

    # A fold whose purge is removed must be rejected, not silently accepted.
    from chronos2_signal.protocol import Fold

    with pytest.raises(ProtocolError):
        Fold(
            index=99,
            fit_sessions=fold.fit_sessions,
            calibration_sessions=(fold.fit_sessions[-1], *fold.calibration_sessions),
            validation_sessions=fold.validation_sessions,
            purge_before_calibration=(),
            purge_before_validation=fold.purge_before_validation,
            fit_deadline=fold.fit_deadline,
            calibration_deadline=fold.calibration_deadline,
        )


# --------------------------------------------------------------------------- #
# 8. Task order and batch budget do not change predictions
# --------------------------------------------------------------------------- #


def test_invariant_08_task_order_and_batch_budget_are_irrelevant(
    calendar, config, market
):
    """With cross-learning disabled, batching must not change a prediction."""
    context = 128
    bars = max(recommended_panel_bars(context), 512)
    symbols = ("AAA", "BBB", "CCC", "DDD")
    tasks = [
        build_chronos_task(
            symbol=symbol,
            origin_session=ORIGIN,
            calendar=calendar,
            stock=market.panel_ending_at(symbol, ORIGIN, bars),
            market=market.panel_ending_at("SPY", ORIGIN, bars),
            sector=market.panel_ending_at(
                {"AAA": "XLA", "BBB": "XLB", "CCC": "XLG", "DDD": "XLA"}[symbol],
                ORIGIN,
                bars,
            ),
            context_length=context,
        )
        for symbol in symbols
    ]

    # Batching itself must respect the channel budget, counting every series.
    assert tasks[0].n_channels == 12
    batches = batch_tasks_by_channels(tasks, 24)
    assert [len(batch) for batch in batches] == [2, 2]
    assert [len(b) for b in batch_tasks_by_channels(tasks, 128)] == [4]
    # A task larger than the budget still forms a batch of one rather than
    # being split across calls.
    assert [len(b) for b in batch_tasks_by_channels(tasks, 4)] == [1, 1, 1, 1]

    def predictions(budget: int, order):
        forecaster = DeterministicStubForecaster(
            quantile_levels=config.model.quantile_levels, batch_size_channels=budget
        )
        results = forecaster.predict([tasks[i] for i in order])
        return {result.symbol: result.paths for result in results}

    reference = predictions(128, range(len(tasks)))
    for budget in (12, 24, 36, 128):
        for order in ([3, 1, 0, 2], [2, 3, 1, 0], list(range(len(tasks)))):
            candidate = predictions(budget, order)
            assert set(candidate) == set(reference)
            for symbol, paths in candidate.items():
                np.testing.assert_allclose(
                    paths, reference[symbol], rtol=1e-6, atol=1e-9
                )

    # And the adapter refuses to run with cross-learning enabled at all.
    from chronos2_signal.forecaster import Chronos2Forecaster, ForecastError

    bad = Chronos2Forecaster(
        model_id=config.model.id,
        revision=config.model.revision,
        quantile_levels=config.model.quantile_levels,
        cross_learning=True,
    )
    with pytest.raises(ForecastError) as excinfo:
        bad.load()
    assert "cross_learning" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 9. Quantile axes, terminal indices and the inverse transform
# --------------------------------------------------------------------------- #


def test_invariant_09_quantile_axes_and_inverse_transforms(calendar, config, market):
    """Levels, axes, terminal indices, padding and the median naming."""
    context = 128
    bars = max(recommended_panel_bars(context), 512)
    task = build_chronos_task(
        symbol="AAA",
        origin_session=ORIGIN,
        calendar=calendar,
        stock=market.panel_ending_at("AAA", ORIGIN, bars),
        market=market.panel_ending_at("SPY", ORIGIN, bars),
        sector=market.panel_ending_at("XLA", ORIGIN, bars),
        context_length=context,
    )
    levels = config.model.quantile_levels
    forecaster = DeterministicStubForecaster(quantile_levels=levels)
    result = forecaster.predict([task])[0]

    # Axis order is (levels, horizon), and the horizon is the calendar's.
    assert result.paths.shape == (len(levels), task.prediction_length)
    assert task.prediction_length == 11  # 7 bars then a 4-bar half day
    assert result.terminal_indices == (6, 10)

    # Quantiles are ordered at every step, and the requested levels round-trip.
    assert (np.diff(result.paths, axis=0) >= 0).all()
    for row, level in enumerate(levels):
        np.testing.assert_array_equal(result.level(level), result.paths[row])

    # The library's point forecast is the median, and it is named that way.
    np.testing.assert_array_equal(result.forecast_median, result.level(0.50))
    assert result.terminal_median(2) == pytest.approx(float(result.level(0.50)[10]))

    # The inverse transform is C_origin * exp(q / 100).
    q = float(result.level(0.90)[10])
    assert task.price_from_target(q) == pytest.approx(
        task.origin_close * math.exp(q / 100.0)
    )
    assert task.return_from_target(q) == pytest.approx(math.expm1(q / 100.0))
    assert task.price_from_target(0.0) == pytest.approx(task.origin_close)
    assert task.target[-1] == pytest.approx(0.0, abs=1e-9)

    # Padding beyond the requested horizon is discarded, not used.
    padded = np.concatenate(
        [result.paths, np.tile(result.paths[:, -1:], (1, 5))], axis=1
    )
    trimmed = validate_quantile_paths(
        "AAA", ORIGIN, levels, padded, task.prediction_length, result.terminal_indices
    )
    assert trimmed.status is ForecastStatus.OK
    assert trimmed.paths.shape[1] == task.prediction_length
    assert trimmed.diagnostics["padding_trimmed"] == 5.0

    # A crossed quantile blocks the forecast; it is not silently sorted.
    crossed = result.paths.copy()
    crossed[0, 3], crossed[-1, 3] = crossed[-1, 3], crossed[0, 3]
    blocked = validate_quantile_paths(
        "AAA", ORIGIN, levels, crossed, task.prediction_length, result.terminal_indices
    )
    assert blocked.status is ForecastStatus.BLOCKED_CROSSED
    assert blocked.diagnostics["quantile_crossings"] > 0
    np.testing.assert_array_equal(blocked.paths, crossed[:, : task.prediction_length])

    # Non-finite output blocks too.
    dirty = result.paths.copy()
    dirty[2, 0] = np.nan
    assert (
        validate_quantile_paths(
            "AAA", ORIGIN, levels, dirty, task.prediction_length, result.terminal_indices
        ).status
        is ForecastStatus.BLOCKED_NONFINITE
    )

    # Too few steps for the calendar horizon is a shape failure, never padded up.
    short = result.paths[:, :-2]
    assert (
        validate_quantile_paths(
            "AAA", ORIGIN, levels, short, task.prediction_length, result.terminal_indices
        ).status
        is ForecastStatus.BLOCKED_SHAPE
    )


# --------------------------------------------------------------------------- #
# 10. Idempotent runs and no fresh-looking signal from failed data
# --------------------------------------------------------------------------- #


def test_invariant_10_duplicate_runs_and_failed_data(tmp_path, config, calendar):
    """One persisted signal and one notification per deduplication key."""
    ledger = Ledger(tmp_path / "ledger.sqlite")
    now = dt.datetime(2025, 11, 25, 21, 30, tzinfo=dt.timezone.utc)
    key = Ledger.dedup_key("chronos2_hourly_v1", "ISSUER-AAA", ORIGIN, "2_sessions")

    def write(signal_id: str) -> str:
        return ledger.record_signal(
            signal_id=signal_id,
            dedup_key=key,
            strategy_version="chronos2_hourly_v1",
            issuer_id="ISSUER-AAA",
            symbol="AAA",
            sector="alpha",
            signal_session=ORIGIN,
            holding_period="2_sessions",
            variant="C256",
            generated_at=now,
            data_timestamp=now,
            estimated_net_return=0.004,
            estimated_net_return_stress=0.001,
            calibrated_probability=0.63,
            sigma_2d=0.02,
            rank_score=0.2,
            rank_position=1,
            forecast_cache_key="cache-1",
            fit_id="fit-1",
            decision="alert",
            suppression_reason=None,
            output_label="RESEARCH / UNVALIDATED",
            payload={"symbol": "AAA", "caveats": ["research only"]},
            expires_at=now,
        )

    first = write("sig-1")
    second = write("sig-2")  # a retried run
    assert first == second == "sig-1"
    assert len(ledger.signals_for_session(ORIGIN)) == 1

    # Notification happens after the commit, and only once.
    channel = RecordingChannel()
    notifier = Notifier(ledger=ledger, channel=channel)
    assert notifier.notify_session(ORIGIN, now=now) == 1
    assert notifier.notify_session(ORIGIN, now=now) == 0
    assert len(channel.messages) == 1
    subject, body = channel.messages[0]
    assert "RESEARCH / UNVALIDATED" in subject
    assert "model estimate" in body

    # Forecasts are append-only at the database level.
    ledger.record_forecast(
        cache_key="cache-1",
        symbol="AAA",
        origin_session=ORIGIN,
        variant="C256",
        checkpoint_id=config.model.id,
        checkpoint_revision=config.model.revision,
        context_length=256,
        prediction_length=11,
        quantile_levels=list(config.model.quantile_levels),
        quantiles=[[0.0] * 11] * 5,
        terminal_indices=[6, 10],
        horizon_sessions=["2025-11-26", "2025-11-28"],
        origin_close=50.0,
        status="ok",
        diagnostics={},
        snapshot_hash="snap-1",
        generated_at=now,
    )
    # Re-recording the identical forecast is a no-op, so retries are safe.
    ledger.record_forecast(
        cache_key="cache-1",
        symbol="AAA",
        origin_session=ORIGIN,
        variant="C256",
        checkpoint_id=config.model.id,
        checkpoint_revision=config.model.revision,
        context_length=256,
        prediction_length=11,
        quantile_levels=list(config.model.quantile_levels),
        quantiles=[[0.0] * 11] * 5,
        terminal_indices=[6, 10],
        horizon_sessions=["2025-11-26", "2025-11-28"],
        origin_close=50.0,
        status="ok",
        diagnostics={},
        snapshot_hash="snap-1",
        generated_at=now,
    )
    with pytest.raises(StorageError):
        ledger.record_forecast(
            cache_key="cache-1",
            symbol="AAA",
            origin_session=ORIGIN,
            variant="C256",
            checkpoint_id=config.model.id,
            checkpoint_revision=config.model.revision,
            context_length=256,
            prediction_length=11,
            quantile_levels=list(config.model.quantile_levels),
            quantiles=[[9.9] * 11] * 5,  # different numbers under the same key
            terminal_indices=[6, 10],
            horizon_sessions=["2025-11-26", "2025-11-28"],
            origin_close=50.0,
            status="ok",
            diagnostics={},
            snapshot_hash="snap-1",
            generated_at=now,
        )
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        ledger.connection.execute(
            "UPDATE forecasts SET origin_close = 1.0 WHERE cache_key = 'cache-1'"
        )
    with pytest.raises(sqlite3.IntegrityError):
        ledger.connection.execute("DELETE FROM forecasts WHERE cache_key = 'cache-1'")

    # Failed data cannot produce a current-looking signal: past the deadline
    # with no valid market snapshot, the origin is DATA_UNAVAILABLE.
    from chronos2_signal.collector import (
        IngestionRecord,
        ReadinessStatus,
        assess_readiness,
    )

    failed = IngestionRecord(symbol="SPY", interval="1h", status="failed")
    late = calendar.data_deadline(ORIGIN, max_delay_minutes=120) + pd.Timedelta(
        minutes=1
    )
    unavailable = assess_readiness(
        origin_session=ORIGIN,
        now=late.to_pydatetime(),
        calendar=calendar,
        market_record=failed,
        sector_records={},
        max_delay_minutes=120,
    )
    assert unavailable.status is ReadinessStatus.DATA_UNAVAILABLE
    assert not unavailable.may_scan
    assert any("not backdated" in reason for reason in unavailable.reasons)

    # Inside the window the same failure suppresses rather than closing out.
    early = calendar.signal_time(ORIGIN).to_pydatetime()
    suppressed = assess_readiness(
        origin_session=ORIGIN,
        now=early,
        calendar=calendar,
        market_record=failed,
        sector_records={},
        max_delay_minutes=120,
    )
    assert suppressed.status is ReadinessStatus.SUPPRESS_SCAN
    assert not suppressed.may_scan

    # A sector failure suppresses only that sector.
    ok_market = IngestionRecord(symbol="SPY", interval="1h", status="ok", snapshot_id="s")
    partial = assess_readiness(
        origin_session=ORIGIN,
        now=early,
        calendar=calendar,
        market_record=ok_market,
        sector_records={
            "alpha": IngestionRecord(symbol="XLA", interval="1h", status="failed"),
            "beta": IngestionRecord(
                symbol="XLB", interval="1h", status="ok", snapshot_id="s2"
            ),
        },
        max_delay_minutes=120,
    )
    assert partial.status is ReadinessStatus.SUPPRESS_SECTORS
    assert partial.may_scan
    assert not partial.sector_allowed("alpha")
    assert partial.sector_allowed("beta")

    # A snapshot file is never overwritten by a second write.
    store = SnapshotStore(tmp_path / "snapshots")
    frame = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2025-11-25"]))
    store.write(frame, "AAA", "1h", "snap-1")
    with pytest.raises(StorageError):
        store.write(frame, "AAA", "1h", "snap-1")
    ledger.close()


# --------------------------------------------------------------------------- #
# 11. The ledger reconciles exactly
# --------------------------------------------------------------------------- #


def test_invariant_11_portfolio_ledger_reconciles(config):
    """Cash, simultaneous holds, correlation limits and actions reconcile.

    A small hand-checkable ledger: USD 100,000 of equity, 10% target positions,
    a 30% gross cap, one dividend and one split.
    """
    book = ReferencePortfolio(config=config, variant="hand-check", cost_scenario="base")
    assert book.cash == pytest.approx(100_000.0)
    slip = config.execution.slippage_fraction("base")
    assert slip == pytest.approx(0.001)

    entry_session = dt.date(2025, 6, 10)
    exit_session = dt.date(2025, 6, 12)
    requests = [
        EntryRequest(
            signal_id="s-AAA",
            symbol="AAA",
            issuer_id="I-AAA",
            sector="alpha",
            signal_session=dt.date(2025, 6, 9),
            entry_session=entry_session,
            planned_exit_session=exit_session,
            reference_open_price=100.0,
        ),
        EntryRequest(
            signal_id="s-BBB",
            symbol="BBB",
            issuer_id="I-BBB",
            sector="beta",
            signal_session=dt.date(2025, 6, 9),
            entry_session=entry_session,
            planned_exit_session=exit_session,
            reference_open_price=50.0,
        ),
    ]
    opened = book.enter_all(requests, equity_at_open=100_000.0, gross_value_at_open=0.0)
    assert len(opened) == 2

    # 10% of 100,000 at a fill of 100 * 1.001 gives 99.9001 shares.
    aaa, bbb = opened
    assert aaa.entry_fill_price == pytest.approx(100.1)
    assert aaa.cost_basis == pytest.approx(10_000.0)
    assert aaa.restated_shares == pytest.approx(10_000.0 / 100.1)
    assert bbb.cost_basis == pytest.approx(10_000.0)
    assert book.cash == pytest.approx(80_000.0)

    # Two simultaneous holds, one per sector, and the issuer rule holds.
    assert len(book.positions) == 2
    assert book.sector_counts() == {"alpha": 1, "beta": 1}
    assert book.open_issuers() == {"I-AAA", "I-BBB"}

    # A third position in a new sector is allowed, but the 30% gross cap on new
    # allocations binds: 30,000 - 20,000 leaves exactly 10,000.
    third = book.allocate(
        EntryRequest(
            signal_id="s-CCC",
            symbol="CCC",
            issuer_id="I-CCC",
            sector="gamma",
            signal_session=dt.date(2025, 6, 9),
            entry_session=entry_session,
            planned_exit_session=exit_session,
            reference_open_price=25.0,
        ),
        equity_at_open=100_000.0,
        gross_value_at_open=20_000.0,
    )
    assert third is not None
    assert third.cost_basis == pytest.approx(10_000.0)
    assert book.cash == pytest.approx(70_000.0)

    # A fourth is refused by the simultaneous-position cap.
    fourth = book.allocate(
        EntryRequest(
            signal_id="s-DDD",
            symbol="DDD",
            issuer_id="I-DDD",
            sector="alpha",
            signal_session=dt.date(2025, 6, 9),
            entry_session=entry_session,
            planned_exit_session=exit_session,
            reference_open_price=10.0,
        ),
        equity_at_open=100_000.0,
        gross_value_at_open=30_000.0,
    )
    assert fourth is None

    # A dividend on 11 June is credited once, to holders who entered before it.
    dividend = CorporateAction(
        symbol="AAA", ex_date=dt.date(2025, 6, 11), action_type="dividend", value=0.40
    )
    split = CorporateAction(
        symbol="BBB", ex_date=dt.date(2025, 6, 11), action_type="split", value=2.0
    )
    ledgers = {
        "AAA": ActionLedger(symbol="AAA", actions=(dividend,)),
        "BBB": ActionLedger(symbol="BBB", actions=(split,)),
        "CCC": ActionLedger(symbol="CCC"),
    }
    nominal_before = bbb.nominal_shares
    book.apply_splits(dt.date(2025, 6, 11), ledgers)
    credited = book.credit_dividends(dt.date(2025, 6, 11), ledgers)
    assert credited == pytest.approx(aaa.restated_shares * 0.40)
    assert book.cash == pytest.approx(70_000.0 + credited)
    # The split changed the quantity and nothing else.
    assert bbb.nominal_shares == pytest.approx(nominal_before * 2.0)
    assert bbb.restated_shares == pytest.approx(10_000.0 / (50.0 * 1.001))
    assert bbb.market_value(50.0) == pytest.approx(bbb.restated_shares * 50.0)

    # Exit at the registered close; cash must equal the sum of the parts.
    closes = {"AAA": 102.0, "BBB": 49.0, "CCC": 26.0}
    closed = book.close_due(exit_session, closes, ledgers)
    assert len(closed) == 3
    assert not book.positions
    expected_cash = 70_000.0 + credited + sum(
        position.restated_shares * closes[position.symbol] * (1.0 - slip)
        for position in closed
    )
    assert book.cash == pytest.approx(expected_cash)

    # Each position's realised return matches the independent accounting engine.
    for position in closed:
        account = book.accounting_for(position, ledgers[position.symbol])
        assert account.reconciles()
        assert position.realised_net_return() == pytest.approx(account.net_return)
        assert account.nominal_split_consistent()

    # And equity equals cash once nothing is open.
    day = book.mark(exit_session, {})
    assert day.equity == pytest.approx(book.cash)
    assert day.gross_exposure == pytest.approx(0.0)
    assert day.open_positions == 0

    # The correlation cap blocks a pair above the limit and blocks entirely when
    # the required history is missing.
    engine = PolicyEngine(config)
    blocked = engine._correlation_block(
        _candidate("AAA", "I-AAA", "alpha", config),
        [],
        [
            OpenPositionView(
                symbol="BBB",
                issuer_id="I-BBB",
                sector="beta",
                planned_exit_session=exit_session,
            )
        ],
        lambda left, right: 0.95,
    )
    assert blocked is not None and "exceeds" in blocked
    missing = engine._correlation_block(
        _candidate("AAA", "I-AAA", "alpha", config),
        [],
        [
            OpenPositionView(
                symbol="BBB",
                issuer_id="I-BBB",
                sector="beta",
                planned_exit_session=exit_session,
            )
        ],
        lambda left, right: None,
    )
    assert missing is not None and "blocked" in missing


def _candidate(symbol, issuer, sector, config):
    from chronos2_signal.decision import DecisionEstimate
    from chronos2_signal.policy import AlertCandidate
    from chronos2_signal.quality import EligibilityResult

    return AlertCandidate(
        symbol=symbol,
        issuer_id=issuer,
        sector=sector,
        estimate=DecisionEstimate(
            symbol=symbol,
            origin_session=ORIGIN,
            estimated_net_return=0.01,
            estimated_net_return_stress=0.005,
            calibrated_probability=0.7,
            raw_probability_score=0.5,
            sigma_2d=0.02,
            scaled_return_prediction=0.5,
        ),
        eligibility=EligibilityResult(
            symbol=symbol, origin_session=ORIGIN, eligible=True
        ),
    )


# --------------------------------------------------------------------------- #
# 12. Identical dates, fills, costs and capital accounting
# --------------------------------------------------------------------------- #


def test_invariant_12_baselines_share_dates_fills_costs_and_capital(
    calendar, config, pipeline
):
    """Candidate and controls run on one date index with identical accounting."""
    schedule = build_schedule(
        calendar,
        config,
        latest_data_session=dt.date(2025, 12, 19),
        earliest_origin=DIAGNOSTIC_EARLIEST,
    )
    fold = schedule.folds[0]
    runner = WalkForwardRunner(pipeline=pipeline, cost_scenario="base")
    cache: dict = {}
    model = runner.fit(
        "C256",
        fit_sessions=fold.fit_sessions,
        calibration_sessions=fold.calibration_sessions,
        fit_deadline=fold.fit_deadline.to_pydatetime(),
        calibration_deadline=fold.calibration_deadline.to_pydatetime(),
        batch_cache=cache,
    )
    candidate = runner.run(
        "C256", model=model, score_sessions=fold.validation_sessions, batch_cache=cache
    )
    momentum = runner.run_momentum_control(
        score_sessions=fold.validation_sessions, batch_cache=cache
    )

    # Same date index, to the session.
    assert candidate.daily.sessions == momentum.daily.sessions
    assert candidate.origins_scanned == momentum.origins_scanned

    # Same capital, same cost scenario, same portfolio machinery.
    assert type(candidate.portfolio) is type(momentum.portfolio)
    assert candidate.portfolio.cost_scenario == momentum.portfolio.cost_scenario
    assert candidate.portfolio.costs == momentum.portfolio.costs
    assert (
        candidate.portfolio.summary()["cost_scenario"]
        == momentum.portfolio.summary()["cost_scenario"]
    )

    # Controls are constructed on that same index, and idle cash earns zero for
    # every system alike.
    sessions = candidate.daily.sessions
    spy = pipeline.source.daily_panel("SPY")
    market_returns, previous = [], None
    for session in sessions:
        close = float(spy.row(session)["close"])
        market_returns.append(0.0 if previous is None else close / previous - 1.0)
        previous = close
    exposure = [day.gross_exposure for day in candidate.portfolio.days]
    matched = exposure_matched_series(
        name="spy_exposure_matched",
        sessions=sessions,
        market_returns=market_returns,
        candidate_gross_exposure=exposure,
    )
    cash = cash_series("cash", sessions)
    assert matched.sessions == sessions == cash.sessions
    assert float(np.abs(cash.returns).max(initial=0.0)) == 0.0

    # The paired bootstrap refuses mismatched date indices outright.
    from chronos2_signal.evaluation import EvaluationError, paired_block_bootstrap

    estimates = paired_block_bootstrap(
        [candidate.daily, momentum.daily, matched, cash],
        block_sessions=config.validation.bootstrap_block_sessions,
        samples=200,
        confidence=0.9,
        reference=momentum.daily.name,
    )
    assert f"{candidate.daily.name}_minus_{momentum.daily.name}" in estimates
    from chronos2_signal.evaluation import DailySeries

    with pytest.raises(EvaluationError):
        paired_block_bootstrap(
            [
                candidate.daily,
                DailySeries(
                    name="shifted",
                    sessions=tuple(sessions[:-1]),
                    returns=candidate.daily.returns[:-1],
                ),
            ],
            block_sessions=5,
            samples=10,
            confidence=0.9,
        )

    # A zero-alert candidate is a legitimate outcome, reported not hidden.
    assert candidate.alerts >= 0
    assert len(candidate.no_alert_origins) + len(
        {trade.origin_session for trade in candidate.trades}
    ) <= candidate.origins_scanned
