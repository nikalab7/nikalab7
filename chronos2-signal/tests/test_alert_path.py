"""End-to-end exercise of the alert path.

At the registered thresholds a random-walk fixture produces no alerts, which is
the correct outcome and is covered in ``test_invariants.py``. To exercise the
code that runs *when* an alert fires -- persistence, deduplication,
notification, entry at the next open, exit at the registered close, the ledger
and the reconciliation against the accounting engine -- these tests lower the
thresholds on a copy of the configuration.

Nothing here is a performance result. The forecaster is an arithmetic fixture
with no predictive content, the prices are a seeded random walk, and the
thresholds are not the registered ones. These tests check plumbing.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pytest

from chronos2_signal.decision import column_set_for_variant
from chronos2_signal.evaluation import trade_metrics
from chronos2_signal.notifier import Notifier, RecordingChannel
from chronos2_signal.operations import (
    OperationsError,
    render_markdown,
    run_after_close,
    run_study,
)
from chronos2_signal.pipeline import FixtureMarketSource, ResearchPipeline, WalkForwardRunner
from chronos2_signal.protocol import build_schedule
from chronos2_signal.storage import Ledger

DIAGNOSTIC_EARLIEST = dt.date(2024, 7, 1)


@pytest.fixture(scope="module")
def permissive_config(request):
    """A copy of the registered configuration with relaxed alert thresholds.

    Used only to reach the alert path. Every threshold that changed is listed
    here so the deviation is explicit rather than buried.
    """
    from chronos2_signal.config import load_design

    base = load_design()
    return dataclasses.replace(
        base,
        status="plumbing_test_only_not_registered",
        decision=dataclasses.replace(
            base.decision,
            min_probability=0.01,
            min_estimated_net_return=1e-9,
            require_positive_stress_estimate=False,
        ),
    )


@pytest.fixture(scope="module")
def permissive_pipeline(permissive_config, calendar, market, watchlist, stub_forecaster):
    return ResearchPipeline(
        config=permissive_config,
        calendar=calendar,
        source=FixtureMarketSource(market),
        watchlist=watchlist,
        forecaster=stub_forecaster,
    )


@pytest.fixture(scope="module")
def fitted(permissive_pipeline, calendar, permissive_config):
    schedule = build_schedule(
        calendar,
        permissive_config,
        latest_data_session=dt.date(2025, 12, 19),
        earliest_origin=DIAGNOSTIC_EARLIEST,
    )
    fold = schedule.folds[0]
    runner = WalkForwardRunner(pipeline=permissive_pipeline, cost_scenario="base")
    cache: dict = {}
    model = runner.fit(
        "C256",
        fit_sessions=fold.fit_sessions,
        calibration_sessions=fold.calibration_sessions,
        fit_deadline=fold.fit_deadline.to_pydatetime(),
        calibration_deadline=fold.calibration_deadline.to_pydatetime(),
        batch_cache=cache,
    )
    return schedule, fold, runner, model, cache


def test_alerts_fire_and_produce_reconciled_trades(fitted, permissive_config):
    """With relaxed thresholds the full path runs: alert, entry, exit, ledger."""
    _schedule, fold, runner, model, cache = fitted
    result = runner.run(
        "C256", model=model, score_sessions=fold.validation_sessions, batch_cache=cache
    )
    assert result.alerts > 0, "relaxed thresholds should reach the alert path"
    assert result.trades, "an alert should become a reference trade"

    for trade in result.trades:
        assert trade.origin_session in set(fold.validation_sessions)
        if trade.resolved:
            assert np.isfinite(trade.r_net_base)
            # Costs are monotone: a wider assumed spread never helps.
            assert trade.r_net_severe <= trade.r_net_stress <= trade.r_net_base

    # Capacity rules held throughout the walk.
    for day in result.portfolio.days:
        assert day.open_positions <= permissive_config.portfolio.max_open_positions
    for outcome in result.policy_outcomes:
        assert outcome.alert_count <= permissive_config.portfolio.max_new_alerts_per_origin

    # Each closed position reconciles against the independent accounting engine.
    book = result.portfolio
    for position in book.closed:
        if position.status != "closed":
            continue
        ledger = runner.pipeline.source.action_ledger(position.symbol)
        account = book.accounting_for(position, ledger)
        assert account.reconciles()
        assert position.realised_net_return() == pytest.approx(account.net_return)

    # Entry is always the session after the signal, exit two sessions after it.
    calendar = runner.pipeline.calendar
    for position in book.closed:
        assert position.entry_session == calendar.next_session(position.signal_session)
        assert position.planned_exit_session == calendar.next_session(
            position.signal_session, 2
        )

    metrics = trade_metrics(result.trades, origins_scanned=result.origins_scanned)
    assert metrics.trades == len(result.trades)
    assert metrics.origins_scanned == len(fold.validation_sessions)
    assert 0.0 <= metrics.alert_coverage <= 1.0


def test_after_close_commits_before_notifying(tmp_path, fitted, permissive_pipeline):
    """A retried after-close run persists once and notifies once."""
    _schedule, fold, _runner, model, _cache = fitted
    origin = fold.validation_sessions[0]
    ledger = Ledger(tmp_path / "ops.sqlite")
    channel = RecordingChannel()
    notifier = Notifier(ledger=ledger, channel=channel)
    now = dt.datetime(2025, 12, 20, 21, 30, tzinfo=dt.timezone.utc)

    first = run_after_close(
        pipeline=permissive_pipeline,
        model=model,
        ledger=ledger,
        origin_session=origin,
        variant="C256",
        notifier=notifier,
        now=now,
    )
    assert first.summary()["output_label"] == "RESEARCH / UNVALIDATED"
    # Every decision is persisted, alerted or not, so a quiet origin is
    # explainable after the fact.
    persisted = ledger.signals_for_session(origin, "C256")
    assert len(persisted) == len(first.outcome.decisions)
    assert first.notified == first.outcome.alert_count

    messages_after_first = len(channel.messages)
    second = run_after_close(
        pipeline=permissive_pipeline,
        model=model,
        ledger=ledger,
        origin_session=origin,
        variant="C256",
        notifier=notifier,
        now=now + dt.timedelta(minutes=5),
    )
    # The retry adds no rows and sends no duplicate alert.
    assert len(ledger.signals_for_session(origin, "C256")) == len(persisted)
    assert second.notified == 0
    if first.outcome.alert_count:
        assert len(channel.messages) == messages_after_first

    runs = ledger.query("SELECT * FROM run_log ORDER BY started_at")
    assert len(runs) == 2
    assert {row["status"] for row in runs} == {"ok"}
    assert all(row["design_fingerprint"] for row in runs)
    assert all(row["environment_digest"] for row in runs)
    ledger.close()


def test_study_runs_controls_and_gates(fitted, permissive_pipeline, tmp_path):
    """A registered study produces controls, a paired bootstrap and gates."""
    schedule, _fold, _runner, _model, _cache = fitted
    study = run_study(
        pipeline=permissive_pipeline,
        schedule=schedule,
        variant="C256",
        checkpoint="plumbing-check",
        fold_index=0,
        bootstrap_samples=200,
        code_revision="test",
        notes=("plumbing check on synthetic data",),
    )

    assert set(study.controls) == {"momentum", "spy_exposure_matched", "cash"}
    for series in study.controls.values():
        assert series.sessions == study.candidate.daily.sessions
    assert "C256_minus_momentum" in study.bootstrap

    # Gates cannot pass on a plumbing run, and the label reflects that.
    assert not study.gates.passed
    assert study.gates.status_label == "RESEARCH / UNVALIDATED"
    assert len(study.gates.results) == 8

    # The manifest records that a non-model forecaster produced the numbers.
    manifest = study.manifest.to_dict()
    assert manifest["design_version"] == "chronos2_hourly_v1"
    assert manifest["checkpoint_revision"] == (
        "95a9710e2596287d08352589f42634fa5abdf0a7"
    )
    assert any("no predictive content" in note for note in manifest["notes"])
    assert manifest["environment"]["packages"]["chronos-forecasting"] == "absent"

    report = render_markdown(study)
    assert "RESEARCH / UNVALIDATED" in report
    assert "Origins with no alert" in report
    assert "Promotion gates" in report

    written = study.write(tmp_path / "study.json")
    assert written.is_file()
    import json

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["gates"]["passed"] is False
    assert payload["trade_metrics"]["origins_scanned"] > 0


def test_study_refuses_a_schedule_without_folds(permissive_pipeline, calendar, config):
    """No folds means no study: collect more dates instead."""
    empty = build_schedule(
        calendar,
        config,
        latest_data_session=dt.date(2025, 12, 19),
        require_folds=False,
    )
    assert empty.folds == ()
    with pytest.raises(OperationsError) as excinfo:
        run_study(
            pipeline=permissive_pipeline,
            schedule=empty,
            variant="C256",
            checkpoint="should-not-run",
        )
    assert "collect more dates" in str(excinfo.value)


def test_b0_uses_fewer_columns_and_needs_no_forecaster(
    permissive_config, calendar, market, watchlist
):
    """B0 is a different feature set, not the same set with blanks."""
    pipeline = ResearchPipeline(
        config=permissive_config,
        calendar=calendar,
        source=FixtureMarketSource(market),
        watchlist=watchlist,
        forecaster=None,  # B0 needs no forecasts at all
    )
    batch = pipeline.build_batch(dt.date(2025, 11, 25), "B0")
    assert batch.scorable()
    for score in batch.scorable():
        assert set(score.features or {}) == set(column_set_for_variant("B0"))
        assert not any(name.startswith("f01") for name in (score.features or {}))
        assert score.forecast is None

    # A Chronos variant without a forecaster refuses rather than improvising.
    from chronos2_signal.pipeline import PipelineError

    with pytest.raises(PipelineError) as excinfo:
        pipeline.build_batch(dt.date(2025, 11, 25), "C256")
    assert "does not fabricate" in str(excinfo.value)
