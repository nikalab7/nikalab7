"""Unit tests for the individual layers.

These complement the twelve integrity invariants: where those check the
system-level guarantees, these check that each component enforces the
registered rule it is responsible for.
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
    HoldingCosts,
    label_net_return,
    max_adverse_excursion,
    reprice_net_return,
)
from chronos2_signal.collector import (
    FetchRequest,
    FixtureCollector,
    Ingestor,
    ProviderError,
    RateLimited,
    RetryPolicy,
    fetch_with_retry,
)
from chronos2_signal.config import ConfigError, load_design
from chronos2_signal.decision import (
    DecisionError,
    OriginRow,
    Preprocessor,
    column_set_for_variant,
    date_normalised_weights,
    fit_decision_model,
)
from chronos2_signal.evaluation import (
    DailySeries,
    TradeRecord,
    block_profitability,
    brier_score,
    contributor_concentration,
    evaluate_promotion_gates,
    interval_coverage,
    moving_block_bootstrap,
    pinball_loss,
    reliability_table,
    spearman_correlation,
    trade_metrics,
)
from chronos2_signal.features import (
    BREADTH_FEATURE_NAME,
    BREADTH_MISSING_FLAG,
    CHRONOS_CHANNEL_NAMES,
    FEATURE_NAMES,
    PREPROCESSING_COLUMNS,
    sigma_2d,
)
from chronos2_signal.forecaster import DeterministicStubForecaster
from chronos2_signal.policy import OutputLabel, PolicyEngine, dedup_key
from chronos2_signal.portfolio import EntryRequest, ReferencePortfolio
from chronos2_signal.protocol import (
    InsufficientHistory,
    build_refit_point,
    build_schedule,
)
from chronos2_signal.quality import evaluate_eligibility
from chronos2_signal.storage import Ledger, SnapshotStore
from chronos2_signal.universe import (
    Candidate,
    CandidateRoster,
    FreezeError,
    UniverseError,
    select_watchlist,
)
from chronos2_signal.variants import (
    REGISTERED_VARIANTS,
    MomentumCandidate,
    VariantError,
    rank_momentum_candidates,
    variant_spec,
)

ORIGIN = dt.date(2025, 11, 25)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_config_matches_the_registered_document(config):
    assert config.design_version == "chronos2_hourly_v1"
    assert config.status == "design_only_unvalidated"
    assert not config.is_validated
    assert config.model.id == "amazon/chronos-2"
    assert config.model.context_length == 256
    assert config.model.context_candidates == (128, 256, 512)
    assert config.model.quantile_levels == (0.10, 0.25, 0.50, 0.75, 0.90)
    assert config.model.cross_learning is False
    assert config.model.finetune is False
    assert config.decision.min_probability == 0.60
    assert config.decision.min_estimated_net_return == 0.003
    assert config.execution.base_slippage_bps_per_side == 10
    assert config.portfolio.max_open_positions == 3
    assert config.validation.earliest_primary_origin == dt.date(2025, 10, 31)
    assert config.validation.final_historical_test_origin_sessions == 60
    assert config.events.version1_mode == "annotations_only"
    assert len(config.fingerprint()) == 64


def test_config_rejects_protocol_drift(tmp_path):
    """The loader refuses a configuration that has drifted from the protocol."""
    source = load_design().source_path
    assert source is not None
    text = source.read_text(encoding="utf-8")

    def write(replacement: tuple[str, str]) -> str:
        target = tmp_path / "drifted.yaml"
        target.write_text(text.replace(*replacement), encoding="utf-8")
        return str(target)

    for replacement, needle in (
        (("cross_learning: false", "cross_learning: true"), "cross_learning"),
        (("auto_adjust: false", "auto_adjust: true"), "auto_adjust"),
        (("finetune: false", "finetune: true"), "finetune"),
        (("gap_filter: disabled", "gap_filter: enabled"), "gap_filter"),
        (("purge_sessions_per_boundary: 2", "purge_sessions_per_boundary: 1"), "purge"),
        (("max_gross_exposure: 0.30", "max_gross_exposure: 1.5"), "max_gross_exposure"),
        (("hourly_request_lookback_days: 729", "hourly_request_lookback_days: 900"), "730"),
        (("archive_initial_lookback_days: 59", "archive_initial_lookback_days: 120"), "60"),
        (("version1_mode: annotations_only", "version1_mode: veto"), "annotations_only"),
        (
            ("calibration: sigmoid_on_later_disjoint_dates", "calibration: random_split"),
            "random-split",
        ),
        (
            ("initial_learned_variants: [B0, C128, C256, C512, U256]",
             "initial_learned_variants: [B0, C256, LGBM]"),
            "registered variants",
        ),
    ):
        with pytest.raises(ConfigError) as excinfo:
            load_design(write(replacement))
        assert needle in str(excinfo.value)

    # An unknown key is a drift too, not a harmless extra.
    extra = tmp_path / "extra.yaml"
    extra.write_text(text + "\nsurprise: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_design(extra)
    assert "unknown keys" in str(excinfo.value)


def test_output_label_fails_closed_on_an_unknown_status(config):
    """Only an explicitly validated status unlocks the qualified label.

    An unrecognised status -- a typo, an experimental branch, a local override --
    must read as unvalidated. Anything else would let a project promote its own
    output by accident.
    """
    from chronos2_signal.config import KNOWN_STATUSES, VALIDATED_STATUSES

    assert not config.is_validated
    for status in ("plumbing_test_only", "", "nearly_validated", "VALIDATED"):
        assert not dataclasses.replace(config, status=status).is_validated
        engine = PolicyEngine(dataclasses.replace(config, status=status))
        assert engine.output_label is OutputLabel.RESEARCH_UNVALIDATED
    for status in sorted(VALIDATED_STATUSES):
        assert dataclasses.replace(config, status=status).is_validated
        assert (
            PolicyEngine(dataclasses.replace(config, status=status)).output_label
            is OutputLabel.QUALIFIED_SIGNAL
        )
    assert VALIDATED_STATUSES <= KNOWN_STATUSES


def test_loader_rejects_an_unknown_status(tmp_path):
    source = load_design().source_path
    assert source is not None
    target = tmp_path / "status.yaml"
    target.write_text(
        source.read_text(encoding="utf-8").replace(
            "status: design_only_unvalidated", "status: looks_good"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_design(target)
    assert "not recognised" in str(excinfo.value)


def test_cost_scenarios_are_ordered(config):
    base = config.execution.slippage_fraction("base")
    stress = config.execution.slippage_fraction("stress")
    severe = config.execution.slippage_fraction("severe")
    assert base < stress < severe
    assert base == pytest.approx(0.0010)
    assert stress == pytest.approx(0.0025)
    assert severe == pytest.approx(0.0050)
    with pytest.raises(ConfigError):
        config.execution.slippage_fraction("optimistic")


# --------------------------------------------------------------------------- #
# Channels and features
# --------------------------------------------------------------------------- #


def test_channel_and_feature_counts_match_the_protocol():
    """Twelve channels, twenty features, twenty-one preprocessing columns."""
    assert len(CHRONOS_CHANNEL_NAMES) == 12
    assert len(FEATURE_NAMES) == 20
    assert len(PREPROCESSING_COLUMNS) == 21
    assert BREADTH_MISSING_FLAG in PREPROCESSING_COLUMNS
    assert BREADTH_FEATURE_NAME in FEATURE_NAMES
    assert len(column_set_for_variant("B0")) == 15
    assert len(column_set_for_variant("C256")) == 21
    with pytest.raises(DecisionError):
        column_set_for_variant("LGBM")


def test_sigma_2d_uses_the_registered_floor(market):
    value = sigma_2d(market.daily("AAA"), ORIGIN, window=20, floor=0.005)
    assert value >= 0.005
    returns = market.daily("AAA").up_to(ORIGIN).log_returns()[-20:]
    expected = max(math.sqrt(2.0) * float(np.std(returns, ddof=1)), 0.005)
    assert value == pytest.approx(expected)
    # A flat series hits the floor rather than collapsing to zero.
    from chronos2_signal.market import DailyPanel

    flat = DailyPanel.from_records(
        "FLAT",
        {
            dt.date(2025, 1, 1) + dt.timedelta(days=i): {"close": 10.0}
            for i in range(40)
        },
    )
    assert sigma_2d(flat, dt.date(2025, 2, 5), floor=0.005) == pytest.approx(0.005)


def test_relative_volume_masks_incompatible_half_day_slots(calendar, market):
    """A half day's shortened bar has no full-length counterpart to compare to."""
    from chronos2_signal.features import _relative_volume

    panel = market.panel_ending_at("AAA", dt.date(2025, 11, 28), 400)
    values, masked = _relative_volume(panel)
    half_day_tail = [
        index
        for index, bar in enumerate(panel.bars)
        if bar.session == dt.date(2025, 11, 28) and bar.slot == 3
    ]
    assert half_day_tail
    # Slot 3 on the half day lasts 30 minutes; slot 3 on a normal session lasts
    # 60, so the comparison set is empty and the value is masked.
    assert panel.bars[half_day_tail[0]].duration_minutes == 30.0
    assert math.isnan(values[half_day_tail[0]])
    assert masked > 0
    # A normal session's final 30-minute bar does have matching history.
    normal_tail = [
        index
        for index, bar in enumerate(panel.bars)
        if bar.session == dt.date(2025, 11, 25) and bar.slot == 6
    ]
    assert math.isfinite(values[normal_tail[0]])


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #


def test_watchlist_selection_rule_and_caps(config, roster, market):
    dollar_volume = {symbol: 1e9 - index for index, symbol in enumerate(roster.by_symbol())}
    manifest = select_watchlist(
        roster,
        config=config,
        selection_session=ORIGIN,
        dollar_volume=dollar_volume,
        frozen_at=dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc),
    )
    # Ranked by dollar volume descending, ties on symbol.
    assert manifest.members[0].selection_dollar_volume >= manifest.members[-1].selection_dollar_volume
    assert all(count <= config.universe.max_issuers_per_sector for count in manifest.sector_counts().values())
    assert len(manifest.members) <= config.universe.max_issuers
    assert len(manifest.hash()) == 64
    assert "survivorship" in " ".join(manifest.limitations)

    # Below the liquidity floor, a candidate is skipped with a reason.
    thin = dict(dollar_volume)
    first = next(iter(thin))
    thin[first] = 1_000.0
    sparse = select_watchlist(
        roster, config=config, selection_session=ORIGIN, dollar_volume=thin
    )
    assert first not in sparse.symbols
    assert any(symbol == first for symbol, _reason in sparse.skipped)


def test_watchlist_cannot_be_frozen_after_results_exist(config, roster):
    with pytest.raises(FreezeError):
        select_watchlist(
            roster,
            config=config,
            selection_session=ORIGIN,
            dollar_volume={symbol: 1e9 for symbol in roster.by_symbol()},
            results_already_exist=True,
        )


def test_roster_rejects_etfs_and_duplicate_issuers():
    with pytest.raises(UniverseError):
        Candidate(
            symbol="XLK",
            issuer_id="ETF-XLK",
            exchange="ARCX",
            sector="tech",
            sector_etf="XLK",
            security_type="etf",
        )
    duplicate = (
        Candidate(symbol="AAA", issuer_id="I1", exchange="XNYS", sector="s", sector_etf="XLS"),
        Candidate(symbol="AAA2", issuer_id="I1", exchange="XNYS", sector="s", sector_etf="XLS"),
    )
    with pytest.raises(UniverseError) as excinfo:
        CandidateRoster(
            candidates=duplicate, source="t", declared_at=dt.datetime.now(dt.timezone.utc)
        )
    assert "one share class per issuer" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Quality and eligibility
# --------------------------------------------------------------------------- #


def test_eligibility_reports_every_failure(config, calendar, market):
    """All gates are evaluated, so the failure list is complete."""
    panel = market.panel_ending_at("AAA", ORIGIN, 200)  # deliberately short
    result = evaluate_eligibility(
        symbol="AAA",
        origin_session=ORIGIN,
        config=config,
        hourly=panel,
        daily=market.daily("AAA").up_to(dt.date(2024, 1, 5)),
        market_hourly=market.panel_ending_at("SPY", ORIGIN, 200),
        sector_hourly=market.panel_ending_at("XLA", ORIGIN, 200),
        action_ledger=ActionLedger(symbol="AAA"),
        issuer_id=None,
    )
    assert not result.eligible
    reasons = " ".join(result.failures)
    assert "issuer_mapping" in reasons
    assert "hourly_schedule" in reasons
    assert "daily_history" in reasons

    good = evaluate_eligibility(
        symbol="AAA",
        origin_session=ORIGIN,
        config=config,
        hourly=market.panel_ending_at("AAA", ORIGIN, 652),
        daily=market.daily("AAA"),
        market_hourly=market.panel_ending_at("SPY", ORIGIN, 652),
        sector_hourly=market.panel_ending_at("XLA", ORIGIN, 652),
        action_ledger=ActionLedger(symbol="AAA"),
        issuer_id="ISSUER-AAA",
    )
    assert good.eligible, good.failures
    assert good.metrics["observed_fraction"] == pytest.approx(1.0)

    suppressed = evaluate_eligibility(
        symbol="AAA",
        origin_session=ORIGIN,
        config=config,
        hourly=market.panel_ending_at("AAA", ORIGIN, 652),
        daily=market.daily("AAA"),
        market_hourly=market.panel_ending_at("SPY", ORIGIN, 652),
        sector_hourly=market.panel_ending_at("XLA", ORIGIN, 652),
        action_ledger=ActionLedger(symbol="AAA"),
        issuer_id="ISSUER-AAA",
        sector_suppressed=True,
    )
    assert not suppressed.eligible
    assert any("sector_suppressed" in failure for failure in suppressed.failures)


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #


def test_retry_backoff_and_rate_limit_behaviour():
    """Transient failures retry with backoff; rate limiting does not retry."""
    delays: list[float] = []
    policy = RetryPolicy(max_retries=3, sleep=delays.append)

    attempts = {"count": 0}

    class Flaky:
        name, version = "flaky", "0"

        def fetch(self, request):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ProviderError("transient")
            return FixtureCollector(frames={(request.symbol, request.interval): pd.DataFrame(
                {"close": [1.0]}, index=pd.DatetimeIndex(["2025-11-25 14:30:00+00:00"])
            )}).fetch(request)

    request = FetchRequest(symbol="AAA", interval="1h")
    response = fetch_with_retry(Flaky(), request, policy)
    assert response.status == "ok"
    assert attempts["count"] == 3
    assert len(delays) == 2
    assert delays[0] < delays[1]  # exponential

    class Limited:
        name, version = "limited", "0"

        def fetch(self, request):
            raise RateLimited("429")

    hits: list[float] = []
    with pytest.raises(RateLimited):
        fetch_with_retry(Limited(), request, RetryPolicy(sleep=hits.append))
    assert hits == []  # no aggressive retrying


def test_ingestor_records_provenance_and_bar_anchoring(tmp_path, calendar, config):
    """Bars land on the expected schedule and unexpected anchors are reported."""
    starts = [bar.start for bar in calendar.session_bars(ORIGIN)]
    rogue = starts[0] + pd.Timedelta(minutes=13)
    frame = pd.DataFrame(
        {
            "open": [10.0] * len(starts) + [10.0],
            "high": [10.5] * len(starts) + [10.5],
            "low": [9.5] * len(starts) + [9.5],
            "close": [10.2] * len(starts) + [10.2],
            "volume": [1000.0] * len(starts) + [1000.0],
        },
        index=pd.DatetimeIndex([*starts, rogue]),
    )
    collector = FixtureCollector(frames={("AAA", "1h"): frame})
    ledger = Ledger(tmp_path / "led.sqlite")
    ingestor = Ingestor(
        calendar=calendar,
        ledger=ledger,
        snapshots=SnapshotStore(tmp_path / "snap"),
        collector=collector,
    )
    record = ingestor.ingest(FetchRequest(symbol="AAA", interval="1h"))
    assert record.ok
    assert record.anchoring is not None
    assert len(record.anchoring["unexpected"]) == 1
    assert record.anchoring["missing"] == []

    snapshots = ledger.query("SELECT * FROM source_snapshots")
    assert len(snapshots) == 1
    assert snapshots[0]["declared_adjustment_mode"] == "auto_adjust=False"
    assert snapshots[0]["payload_sha256"]

    bars = ledger.read_bars("AAA", "1h", as_of=collector.retrieved_at)
    assert len(bars) == len(starts)  # the rogue bar was not stored as a price
    assert all(row["observed"] == 1 for row in bars)
    ledger.close()


def test_point_in_time_bar_read_prefers_the_old_vintage(tmp_path, calendar):
    """A provider revision does not overwrite the vintage a decision used."""
    ledger = Ledger(tmp_path / "led.sqlite")
    start = calendar.session_bars(ORIGIN)[0]
    for snapshot_id, close, retrieved in (
        ("snap-old", 10.0, dt.datetime(2025, 11, 25, 21, 30, tzinfo=dt.timezone.utc)),
        ("snap-new", 11.0, dt.datetime(2025, 12, 1, 21, 30, tzinfo=dt.timezone.utc)),
    ):
        ledger.record_snapshot(
            snapshot_id=snapshot_id,
            provider="fixture",
            provider_version="0",
            symbol="AAA",
            interval="1h",
            requested_start=None,
            requested_end=None,
            retrieved_at=retrieved,
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
                    "interval": "1h",
                    "bar_start": start.start,
                    "bar_end": start.end,
                    "session": ORIGIN,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1.0,
                    "adj_close": close,
                    "observed": True,
                    "snapshot_id": snapshot_id,
                    "retrieved_at": retrieved,
                }
            ]
        )
    at_decision = ledger.read_bars(
        "AAA", "1h", as_of=dt.datetime(2025, 11, 26, tzinfo=dt.timezone.utc)
    )
    assert [row["close"] for row in at_decision] == [10.0]
    latest = ledger.read_bars("AAA", "1h")
    assert [row["close"] for row in latest] == [11.0]
    ledger.close()


# --------------------------------------------------------------------------- #
# Decision model
# --------------------------------------------------------------------------- #


def test_preprocessor_uses_training_statistics_only():
    columns = ("a", "b", BREADTH_FEATURE_NAME)
    train = [
        {"a": 1.0, "b": 10.0, BREADTH_FEATURE_NAME: 0.5},
        {"a": 3.0, "b": 12.0, BREADTH_FEATURE_NAME: 0.7},
        {"a": 5.0, "b": 14.0, BREADTH_FEATURE_NAME: float("nan")},
    ]
    pre = Preprocessor.fit(train, columns, clip_abs=8.0)
    assert pre.centers[0] == pytest.approx(3.0)

    # A later block with a wildly different scale does not move the statistics.
    later = [{"a": 1000.0, "b": 2000.0, BREADTH_FEATURE_NAME: 0.9}]
    matrix, clipped = pre.transform(later)
    assert pre.centers[0] == pytest.approx(3.0)
    assert clipped > 0.0  # far-out values are clipped, and the rate is reported
    assert float(np.abs(matrix).max()) <= 8.0

    # The optional feature is median-imputed; a required one is not.
    imputed, _ = pre.transform([{"a": 1.0, "b": 10.0, BREADTH_FEATURE_NAME: float("nan")}])
    assert np.isfinite(imputed).all()
    with pytest.raises(DecisionError) as excinfo:
        pre.transform([{"a": float("nan"), "b": 10.0, BREADTH_FEATURE_NAME: 0.5}])
    assert "skipped rather than silently imputed" in str(excinfo.value)


def test_date_normalised_weights_sum_to_one_per_date():
    sessions = [dt.date(2025, 1, 6)] * 3 + [dt.date(2025, 1, 7)] * 5
    weights = date_normalised_weights(sessions)
    assert weights[:3].sum() == pytest.approx(1.0)
    assert weights[3:].sum() == pytest.approx(1.0)
    assert weights.sum() == pytest.approx(2.0)


def test_calibration_block_must_be_later_and_disjoint(config):
    columns = column_set_for_variant("C256")
    rng = np.random.default_rng(3)

    def rows(sessions):
        out = []
        for session in sessions:
            for symbol in ("A", "B", "C", "D"):
                out.append(
                    OriginRow(
                        origin_session=session,
                        symbol=symbol,
                        features={name: float(rng.normal()) for name in columns},
                        sigma_2d=0.02,
                        r_net=float(rng.normal(scale=0.01)),
                        eligible_count=4,
                    )
                )
        return out

    base = [dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(200)]
    base = [session for session in base if session.weekday() < 5]
    fit_block, calibration_block = base[:80], base[82:112]

    model = fit_decision_model(
        variant="C256",
        fit_rows=rows(fit_block),
        calibration_rows=rows(calibration_block),
        config=config,
    )
    assert model.report.fit_dates == 80
    assert model.report.calibration_dates == 30
    assert model.report.sample_weight_total == pytest.approx(80.0)
    assert 0.0 <= model.training_base_rate <= 1.0

    # Overlapping blocks are refused outright.
    with pytest.raises(DecisionError) as excinfo:
        fit_decision_model(
            variant="C256",
            fit_rows=rows(base[:80]),
            calibration_rows=rows(base[70:100]),
            config=config,
        )
    assert "disjoint" in str(excinfo.value)


def test_estimate_rescales_by_sigma_and_reprices_costs(config):
    columns = column_set_for_variant("B0")
    rng = np.random.default_rng(5)

    def rows(sessions):
        out = []
        for session in sessions:
            for symbol in ("A", "B", "C", "D"):
                features = {name: float(rng.normal()) for name in columns}
                out.append(
                    OriginRow(
                        origin_session=session,
                        symbol=symbol,
                        features=features,
                        sigma_2d=0.02,
                        r_net=0.01 * features[columns[0]] + float(rng.normal(scale=0.002)),
                        eligible_count=4,
                    )
                )
        return out

    base = [dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(200)]
    base = [session for session in base if session.weekday() < 5]
    model = fit_decision_model(
        variant="B0",
        fit_rows=rows(base[:80]),
        calibration_rows=rows(base[82:112]),
        config=config,
    )
    row = rows([base[130]])[0]
    estimate = model.estimate(row)
    assert estimate.estimated_net_return == pytest.approx(
        estimate.scaled_return_prediction * row.sigma_2d
    )
    # Stress costs are strictly worse than base costs for the same estimate.
    assert estimate.estimated_net_return_stress < estimate.estimated_net_return
    assert 0.0 < estimate.calibrated_probability < 1.0

    # The same feature row at twice the volatility scale yields exactly twice
    # the estimated net return: the head predicts a scaled quantity.
    doubled = dataclasses.replace(row, sigma_2d=2.0 * row.sigma_2d)
    doubled_estimate = model.estimate(doubled)
    assert doubled_estimate.scaled_return_prediction == pytest.approx(
        estimate.scaled_return_prediction
    )
    assert doubled_estimate.estimated_net_return == pytest.approx(
        2.0 * estimate.estimated_net_return
    )
    with pytest.raises(DecisionError):
        model.estimate(dataclasses.replace(row, sigma_2d=0.0))


def test_reprice_net_return_is_exact_for_a_realised_trade():
    base = HoldingCosts(buy_slippage=0.001, sell_slippage=0.001)
    stress = HoldingCosts(buy_slippage=0.0025, sell_slippage=0.0025)
    account_base = label_net_return(
        entry_session=dt.date(2025, 6, 10),
        exit_session=dt.date(2025, 6, 12),
        entry_open_price=100.0,
        exit_close_price=103.0,
        costs=base,
    )
    account_stress = label_net_return(
        entry_session=dt.date(2025, 6, 10),
        exit_session=dt.date(2025, 6, 12),
        entry_open_price=100.0,
        exit_close_price=103.0,
        costs=stress,
    )
    assert reprice_net_return(
        account_base.net_return, base=base, target=stress
    ) == pytest.approx(account_stress.net_return)


def test_max_adverse_excursion_is_diagnostic_only():
    assert max_adverse_excursion(100.0, [99.0, 95.0, 101.0]) == pytest.approx(-0.05)
    assert max_adverse_excursion(100.0, [101.0, 102.0]) == pytest.approx(0.0)
    assert math.isnan(max_adverse_excursion(100.0, []))


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def _candidate(config, symbol, *, probability, estimate, sector="alpha", sigma=0.02):
    from chronos2_signal.decision import DecisionEstimate
    from chronos2_signal.policy import AlertCandidate
    from chronos2_signal.quality import EligibilityResult

    stressed = reprice_net_return(
        estimate,
        base=HoldingCosts(
            buy_slippage=config.execution.slippage_fraction("base"),
            sell_slippage=config.execution.slippage_fraction("base"),
        ),
        target=HoldingCosts(
            buy_slippage=config.execution.slippage_fraction("stress"),
            sell_slippage=config.execution.slippage_fraction("stress"),
        ),
    )
    return AlertCandidate(
        symbol=symbol,
        issuer_id=f"I-{symbol}",
        sector=sector,
        estimate=DecisionEstimate(
            symbol=symbol,
            origin_session=ORIGIN,
            estimated_net_return=estimate,
            estimated_net_return_stress=stressed,
            calibrated_probability=probability,
            raw_probability_score=0.0,
            sigma_2d=sigma,
            scaled_return_prediction=estimate / sigma,
        ),
        eligibility=EligibilityResult(symbol=symbol, origin_session=ORIGIN, eligible=True),
    )


def test_policy_thresholds_ranking_and_capacity(config):
    engine = PolicyEngine(config)
    assert engine.output_label is OutputLabel.RESEARCH_UNVALIDATED

    candidates = [
        _candidate(config, "AAA", probability=0.70, estimate=0.010, sector="alpha"),
        _candidate(config, "BBB", probability=0.65, estimate=0.020, sector="beta"),
        _candidate(config, "CCC", probability=0.59, estimate=0.050, sector="gamma"),
        _candidate(config, "DDD", probability=0.80, estimate=0.001, sector="delta"),
        _candidate(config, "EEE", probability=0.66, estimate=0.015, sector="beta"),
    ]
    outcome = engine.evaluate(
        origin_session=ORIGIN,
        candidates=candidates,
        correlation=lambda left, right: 0.1,
    )
    alerted = [decision.symbol for decision in outcome.alerts]
    # BBB has the highest estimate-per-sigma, then AAA. CCC fails the
    # probability gate, DDD the return gate, and EEE the two-alert cap.
    assert alerted == ["BBB", "AAA"]
    assert len(alerted) == config.portfolio.max_new_alerts_per_origin

    reasons = {decision.symbol: decision.reasons for decision in outcome.decisions}
    assert any("probability" in reason for reason in reasons["CCC"])
    assert any("estimate" in reason for reason in reasons["DDD"])
    assert any("capacity" in reason for reason in reasons["EEE"])

    # One position per sector.
    same_sector = engine.evaluate(
        origin_session=ORIGIN,
        candidates=[
            _candidate(config, "AAA", probability=0.70, estimate=0.020, sector="alpha"),
            _candidate(config, "BBB", probability=0.70, estimate=0.010, sector="alpha"),
        ],
        correlation=lambda left, right: 0.1,
    )
    assert [decision.symbol for decision in same_sector.alerts] == ["AAA"]

    # The correlation cap and missing history both block.
    correlated = engine.evaluate(
        origin_session=ORIGIN,
        candidates=[
            _candidate(config, "AAA", probability=0.70, estimate=0.020, sector="alpha"),
            _candidate(config, "BBB", probability=0.70, estimate=0.010, sector="beta"),
        ],
        correlation=lambda left, right: 0.95,
    )
    assert [decision.symbol for decision in correlated.alerts] == ["AAA"]
    no_source = engine.evaluate(
        origin_session=ORIGIN,
        candidates=[
            _candidate(config, "AAA", probability=0.70, estimate=0.020, sector="alpha"),
            _candidate(config, "BBB", probability=0.70, estimate=0.010, sector="beta"),
        ],
        correlation=None,
    )
    assert [decision.symbol for decision in no_source.alerts] == ["AAA"]


def test_policy_reports_zero_alerts_and_respects_the_pause(config):
    engine = PolicyEngine(config)
    quiet = engine.evaluate(
        origin_session=ORIGIN,
        candidates=[_candidate(config, "AAA", probability=0.10, estimate=0.0001)],
        correlation=lambda left, right: 0.0,
    )
    assert quiet.alert_count == 0
    assert any("valid outcome" in note for note in quiet.origin_notes)

    paused = engine.evaluate(
        origin_session=ORIGIN,
        candidates=[_candidate(config, "AAA", probability=0.90, estimate=0.05)],
        correlation=lambda left, right: 0.0,
        paused=True,
    )
    assert paused.alert_count == 0
    assert any("paused" in note for note in paused.origin_notes)

    suppressed = engine.evaluate(
        origin_session=ORIGIN,
        candidates=[_candidate(config, "AAA", probability=0.90, estimate=0.05)],
        correlation=lambda left, right: 0.0,
        suppressed_sectors=frozenset({"alpha"}),
    )
    assert suppressed.alert_count == 0


def test_dedup_key_is_issuer_scoped():
    left = dedup_key("v1", "ISSUER-A", ORIGIN, "2_sessions")
    right = dedup_key("v1", "ISSUER-A", ORIGIN, "2_sessions")
    assert left == right
    assert left != dedup_key("v1", "ISSUER-B", ORIGIN, "2_sessions")
    assert left != dedup_key("v2", "ISSUER-A", ORIGIN, "2_sessions")
    assert left == Ledger.dedup_key("v1", "ISSUER-A", ORIGIN, "2_sessions")


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #


def test_drawdown_pause_is_absorbing(config):
    book = ReferencePortfolio(config=config, variant="pause-check")
    request = EntryRequest(
        signal_id="s",
        symbol="AAA",
        issuer_id="I-AAA",
        sector="alpha",
        signal_session=dt.date(2025, 6, 9),
        entry_session=dt.date(2025, 6, 10),
        planned_exit_session=dt.date(2025, 6, 12),
        reference_open_price=100.0,
    )
    position = book.allocate(request, equity_at_open=100_000.0, gross_value_at_open=0.0)
    assert position is not None

    book.mark(dt.date(2025, 6, 10), {"AAA": 100.0})
    # A 60% collapse on a 10% position is a 6% portfolio drawdown.
    day = book.mark(dt.date(2025, 6, 11), {"AAA": 40.0})
    assert day.drawdown < -config.portfolio.pause_new_alerts_at_portfolio_drawdown
    assert book.paused
    assert book.pause_session == dt.date(2025, 6, 11)

    # Recovery does not lift the pause, and no new allocation is accepted.
    recovered = book.mark(dt.date(2025, 6, 12), {"AAA": 130.0})
    assert recovered.drawdown > -0.01
    assert book.paused
    assert book.allocate(
        dataclasses.replace(request, symbol="BBB", issuer_id="I-BBB", sector="beta"),
        equity_at_open=recovered.equity,
        gross_value_at_open=0.0,
    ) is None

    # But an existing position keeps its registered time exit.
    closed = book.close_due(dt.date(2025, 6, 12), {"AAA": 130.0})
    assert len(closed) == 1
    assert closed[0].status == "closed"


def test_existing_exposure_may_drift_above_the_cap_without_a_rebalance(config):
    book = ReferencePortfolio(config=config, variant="drift-check")
    for index, symbol in enumerate(("AAA", "BBB", "CCC")):
        book.allocate(
            EntryRequest(
                signal_id=f"s{index}",
                symbol=symbol,
                issuer_id=f"I-{symbol}",
                sector=f"sector{index}",
                signal_session=dt.date(2025, 6, 9),
                entry_session=dt.date(2025, 6, 10),
                planned_exit_session=dt.date(2025, 6, 12),
                reference_open_price=100.0,
            ),
            equity_at_open=100_000.0,
            gross_value_at_open=index * 10_000.0,
        )
    assert len(book.positions) == 3
    # Prices double: gross exposure is now far above the 30% new-allocation cap.
    day = book.mark(dt.date(2025, 6, 11), {s: 200.0 for s in ("AAA", "BBB", "CCC")})
    assert day.gross_exposure > config.portfolio.max_gross_exposure
    # No rebalancing exit was invented: all three are still open.
    assert len(book.positions) == 3
    assert day.open_positions == 3


def test_missing_exit_price_keeps_the_trade_in_the_ledger(config):
    book = ReferencePortfolio(config=config, variant="delisting-check")
    book.allocate(
        EntryRequest(
            signal_id="s",
            symbol="AAA",
            issuer_id="I-AAA",
            sector="alpha",
            signal_session=dt.date(2025, 6, 9),
            entry_session=dt.date(2025, 6, 10),
            planned_exit_session=dt.date(2025, 6, 12),
            reference_open_price=100.0,
        ),
        equity_at_open=100_000.0,
        gross_value_at_open=0.0,
    )
    closed = book.close_due(dt.date(2025, 6, 12), {})  # no price available
    assert len(closed) == 1
    assert closed[0].status == "exit_price_unavailable"
    assert closed[0].proceeds is None
    assert book.summary()["unresolved_exits"] == 1


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


def test_post_checkpoint_history_is_currently_insufficient(calendar, config):
    """The registered earliest origin cannot yet support a study.

    This is the honest state of the protocol, not a failure: with the frozen
    checkpoint dated late October 2025, fewer origins exist than the final test
    alone reserves.
    """
    with pytest.raises(InsufficientHistory) as excinfo:
        build_schedule(calendar, config, latest_data_session=dt.date(2025, 12, 19))
    assert "collect more dates" in str(excinfo.value)

    inspected = build_schedule(
        calendar,
        config,
        latest_data_session=dt.date(2025, 12, 19),
        require_folds=False,
    )
    assert inspected.folds == ()
    assert inspected.development_origins == ()
    assert inspected.notes


def test_refit_point_uses_only_matured_history(calendar, config):
    schedule = build_schedule(
        calendar,
        config,
        latest_data_session=dt.date(2025, 12, 19),
        earliest_origin=dt.date(2024, 7, 1),
    )
    effective = schedule.test_origins[0]
    point = build_refit_point(
        calendar,
        config,
        available_origins=schedule.labelled_origins,
        effective_from=effective,
    )
    assert point.effective_from == effective
    assert len(point.calibration_sessions) == config.validation.calibration_origin_sessions
    assert len(point.fit_sessions) >= config.validation.min_fit_origin_sessions
    assert len(point.purged_sessions) == config.validation.purge_sessions_per_boundary
    # Nothing in the fit or calibration blocks reaches the scoring origin.
    assert max(point.calibration_sessions) < effective
    assert set(point.purged_sessions).isdisjoint(point.fit_sessions)
    assert set(point.purged_sessions).isdisjoint(point.calibration_sessions)

    with pytest.raises(InsufficientHistory):
        build_refit_point(
            calendar,
            config,
            available_origins=schedule.labelled_origins[:10],
            effective_from=effective,
        )


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #


def test_only_five_variants_are_registered():
    assert set(REGISTERED_VARIANTS) == {"B0", "C128", "C256", "C512", "U256"}
    assert variant_spec("C512").context_length == 512
    assert variant_spec("U256").include_covariates is False
    assert variant_spec("B0").uses_forecast is False
    with pytest.raises(VariantError):
        variant_spec("LGBM")


def test_momentum_control_respects_the_same_capacity_rules(config):
    candidates = [
        MomentumCandidate("AAA", "I-AAA", "alpha", 0.05),
        MomentumCandidate("BBB", "I-BBB", "alpha", 0.04),
        MomentumCandidate("CCC", "I-CCC", "beta", 0.03),
        MomentumCandidate("DDD", "I-DDD", "gamma", -0.01),
    ]
    picks = rank_momentum_candidates(candidates, config=config)
    assert [pick.symbol for pick in picks] == ["AAA", "CCC"]  # one per sector, top two

    with_open = rank_momentum_candidates(
        candidates,
        config=config,
        open_sectors={"alpha": 1},
        open_issuers=["I-CCC"],
        open_position_count=2,
    )
    assert [pick.symbol for pick in with_open] == []


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def test_predictive_metrics():
    levels = [0.1, 0.5, 0.9]
    predictions = np.array([[-1.0], [0.0], [1.0]])
    assert pinball_loss(levels, predictions, [0.0]) == pytest.approx(
        (0.1 * 1.0 + 0.0 + 0.1 * 1.0) / 3.0
    )
    assert interval_coverage([-1, -1, -1], [1, 1, 1], [0.0, 2.0, -0.5]) == pytest.approx(
        2 / 3
    )
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    table = reliability_table([0.05, 0.15, 0.95], [0, 0, 1], bins=2)
    assert sum(row["count"] for row in table) == 3
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert math.isnan(spearman_correlation([1.0, 2.0], [1.0, 2.0]))


def test_block_bootstrap_preserves_block_structure():
    sessions = tuple(dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(60))
    series = DailySeries(
        name="s", sessions=sessions, returns=np.full(60, 0.001, dtype=float)
    )
    estimate = moving_block_bootstrap(
        series, block_sessions=10, samples=500, confidence=0.95
    )
    # A constant series has no sampling variability at all.
    assert estimate.point == pytest.approx(0.001)
    assert estimate.lower == pytest.approx(0.001)
    assert estimate.upper == pytest.approx(0.001)
    assert estimate.excludes_zero_above

    noisy = DailySeries(
        name="n",
        sessions=sessions,
        returns=np.random.default_rng(1).normal(0.0, 0.01, 60),
    )
    wide = moving_block_bootstrap(noisy, block_sessions=10, samples=1000, confidence=0.95)
    assert wide.lower < wide.point < wide.upper
    assert not wide.excludes_zero_above


def test_trade_metrics_report_no_alert_origins():
    trades = [
        TradeRecord(dt.date(2025, 1, 6), "AAA", "I-AAA", "alpha", 0.02, 0.015, 0.01),
        TradeRecord(dt.date(2025, 1, 7), "BBB", "I-BBB", "beta", -0.01, -0.015, -0.02),
    ]
    metrics = trade_metrics(trades, origins_scanned=20)
    assert metrics.trades == 2
    assert metrics.distinct_sessions == 2
    assert metrics.no_alert_origins == 18
    assert metrics.alert_coverage == pytest.approx(0.1)
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(2.0)

    empty = trade_metrics([], origins_scanned=20)
    assert empty.trades == 0
    assert empty.no_alert_origins == 20
    assert math.isnan(empty.win_rate)


def test_concentration_and_block_profitability():
    trades = [
        TradeRecord(dt.date(2025, 1, 6), "AAA", "I-AAA", "alpha", 0.10, 0.09, 0.08),
        TradeRecord(dt.date(2025, 1, 7), "BBB", "I-BBB", "beta", 0.01, 0.005, 0.0),
    ]
    concentration = contributor_concentration(trades)
    assert concentration["top_key"] == "I-AAA"
    assert concentration["top_share"] == pytest.approx(0.10 / 0.11)
    assert concentration["profitable_excluding_best_trade"] is True

    sessions = tuple(dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(45))
    series = DailySeries(
        name="s",
        sessions=sessions,
        returns=np.concatenate([np.full(20, 0.001), np.full(25, -0.001)]),
    )
    blocks = block_profitability(series, block_sessions=20)
    assert len(blocks) == 3
    assert blocks[0]["total_return"] > 0
    assert blocks[1]["total_return"] < 0


def test_promotion_gates_fail_closed(config):
    sessions = tuple(dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(20))
    series = DailySeries(name="C256", sessions=sessions, returns=np.zeros(20))
    report = evaluate_promotion_gates(
        checkpoint="unit-test",
        historical_trades=[],
        forward_trades=[],
        forward_sessions=0,
        portfolio_series=series,
        baseline_series={},
        bootstrap={},
        calibration_brier=float("nan"),
        base_rate_brier=float("nan"),
        coverage_p10_p90=float("nan"),
        reliability=[],
        config_min_forward_sessions=config.validation.forward_min_sessions,
        config_min_forward_signals=config.validation.forward_min_executed_signals,
        config_min_combined_signals=config.validation.combined_evaluation_min_executed_signals,
        config_min_combined_sessions=config.validation.combined_evaluation_min_signal_sessions,
        config_min_profitable_blocks=config.validation.promotion_min_profitable_blocks,
        config_block_sessions=config.validation.promotion_block_sessions,
        historical_origins_scanned=20,
        assessed_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    assert not report.passed
    assert report.status_label == "RESEARCH / UNVALIDATED"
    assert len(report.results) == 8
    assert len(report.failures()) == 8
    # Gate 8 does not pass without a recorded ablation, by construction.
    gate_eight = next(result for result in report.results if result.number == 8)
    assert not gate_eight.passed
    assert "recorded ablation" in gate_eight.detail


# --------------------------------------------------------------------------- #
# Forecaster provenance
# --------------------------------------------------------------------------- #


def test_stub_forecaster_declares_itself_not_a_model(config):
    stub = DeterministicStubForecaster(quantile_levels=config.model.quantile_levels)
    provenance = stub.provenance()
    assert provenance["is_frozen_checkpoint"] is False
    assert "no predictive content" in provenance["warning"]
    assert stub.is_frozen_checkpoint is False


def test_real_forecaster_requires_the_package(config):
    from chronos2_signal.forecaster import Chronos2Forecaster, ForecastError

    forecaster = Chronos2Forecaster(
        model_id=config.model.id,
        revision=config.model.revision,
        quantile_levels=config.model.quantile_levels,
    )
    assert forecaster.is_frozen_checkpoint is True
    with pytest.raises(ForecastError) as excinfo:
        forecaster.load()
    message = str(excinfo.value)
    assert "chronos-forecasting is not installed" in message
    assert "refuses to fabricate" in message


def test_ledger_has_every_required_table(tmp_path):
    from chronos2_signal.storage import LEDGER_TABLES

    ledger = Ledger(tmp_path / "led.sqlite")
    tables = set(ledger.tables())
    assert set(LEDGER_TABLES) <= tables
    ledger.close()
