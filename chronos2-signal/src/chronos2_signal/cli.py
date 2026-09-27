"""Command line entry points.

The commands follow the protocol's build order. Anything that would need data,
weights or history the project does not have refuses with the reason and the
build-order step that supplies it, rather than producing a number.

    chronos2-signal verify            structural self-check and provenance
    chronos2-signal schedule          the chronology a given data edge supports
    chronos2-signal audit             the data-audit plan (dry by default)
    chronos2-signal smoke             checkpoint compatibility (needs the model extra)
    chronos2-signal demo-study        an offline plumbing run on synthetic data
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Sequence

from .calendar_spec import CALENDAR_CHANNEL_NAMES, ExchangeCalendar
from .config import ConfigError, load_design
from .features import CHRONOS_CHANNEL_NAMES, FEATURE_NAMES, PREPROCESSING_COLUMNS
from .provenance import capture_environment
from .storage import LEDGER_TABLES, Ledger
from .variants import REGISTERED_VARIANTS

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronos2-signal",
        description=(
            "Free-data, long-only research scanner for a frozen US watchlist. "
            "Design chronos2_hourly_v1, unvalidated: no edge is claimed."
        ),
    )
    # The global options are also attached to every subcommand, so both
    # "chronos2-signal --json verify" and "chronos2-signal verify --json" work.
    # SUPPRESS on the subcommand copies keeps them from clobbering a value given
    # before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="path to the registered design configuration (default: bundled)",
    )
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit machine-readable output",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to the registered design configuration (default: bundled)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable output"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "verify",
        parents=[common],
        help="structural self-check, provenance and dependency state",
    )

    schedule = subparsers.add_parser(
        "schedule",
        parents=[common],
        help="the folds, holdout and refit points a data edge supports",
    )
    schedule.add_argument(
        "--latest-session",
        required=True,
        help="most recent session with complete data (YYYY-MM-DD)",
    )
    schedule.add_argument(
        "--earliest-origin",
        default=None,
        help=(
            "override the registered earliest primary origin. Only for a "
            "separately labelled diagnostic study, never for post-checkpoint evidence."
        ),
    )

    audit = subparsers.add_parser(
        "audit",
        parents=[common],
        help="print the data-audit plan; fetches only with --live",
    )
    audit.add_argument("--roster", type=Path, required=True, help="candidate roster CSV")
    audit.add_argument(
        "--live",
        action="store_true",
        help="actually contact the provider (requires the 'provider' extra)",
    )
    audit.add_argument(
        "--out", type=Path, default=Path("artifacts"), help="artifact directory"
    )

    subparsers.add_parser(
        "smoke",
        parents=[common],
        help="checkpoint compatibility smoke test; requires the 'model' extra",
    )

    demo = subparsers.add_parser(
        "demo-study",
        parents=[common],
        help="offline plumbing run on synthetic data; produces no performance evidence",
    )
    demo.add_argument("--variant", default="C256", choices=sorted(REGISTERED_VARIANTS))
    demo.add_argument(
        "--out", type=Path, default=Path("artifacts"), help="artifact directory"
    )
    demo.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="reduced sample count so the demo stays quick",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # SUPPRESS leaves the attribute absent when a subcommand copy was not used.
    args.json = getattr(args, "json", False)
    args.config = getattr(args, "config", None)
    try:
        config = load_design(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    handlers = {
        "verify": _verify,
        "schedule": _schedule,
        "audit": _audit,
        "smoke": _smoke,
        "demo-study": _demo_study,
    }
    return handlers[args.command](args, config)


# --------------------------------------------------------------------------- #


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for inner_key, inner_value in value.items():
                print(f"  {inner_key}: {inner_value}")
        elif isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")


def _verify(args: argparse.Namespace, config) -> int:
    """Confirm the code, the configuration and the environment agree."""
    calendar = ExchangeCalendar(config.runtime.exchange_calendar)
    environment = capture_environment()
    problems: list[str] = []

    if len(CHRONOS_CHANNEL_NAMES) != 12:
        problems.append(f"expected 12 task channels, found {len(CHRONOS_CHANNEL_NAMES)}")
    if len(CALENDAR_CHANNEL_NAMES) != 5:
        problems.append(
            f"expected 5 calendar channels, found {len(CALENDAR_CHANNEL_NAMES)}"
        )
    if len(FEATURE_NAMES) != 20 or len(PREPROCESSING_COLUMNS) != 21:
        problems.append(
            f"expected 20 features and 21 preprocessing columns, found "
            f"{len(FEATURE_NAMES)} and {len(PREPROCESSING_COLUMNS)}"
        )

    # A normal session is seven bars ending at the close; a half day is fewer.
    probe = dt.date(2025, 11, 25)
    if calendar.bars_per_session(probe) != 7:
        problems.append(
            f"{probe} should hold 7 scheduled bars, calendar says "
            f"{calendar.bars_per_session(probe)}"
        )
    horizon = calendar.horizon(probe, config.model.horizon_sessions)
    if horizon.prediction_length != 11 or horizon.terminal_indices != (6, 10):
        problems.append(
            "the horizon across the 2025 Thanksgiving half day should be 11 bars with "
            f"terminals (6, 10); calendar says {horizon.prediction_length} and "
            f"{horizon.terminal_indices}"
        )

    missing = environment.missing_packages()
    ledger_tables_ok = True
    try:
        import tempfile

        with (
            tempfile.TemporaryDirectory() as directory,
            Ledger(Path(directory) / "probe.sqlite") as ledger,
        ):
            ledger_tables_ok = set(LEDGER_TABLES) <= set(ledger.tables())
    except Exception as exc:  # pragma: no cover - environment dependent
        problems.append(f"ledger schema could not be created: {exc}")
    if not ledger_tables_ok:
        problems.append("ledger schema is missing required tables")

    payload: dict[str, object] = {
        "design_version": config.design_version,
        "status": config.status,
        "is_validated": config.is_validated,
        "design_fingerprint": config.fingerprint(),
        "config_path": str(config.source_path),
        "calendar": calendar.provenance(),
        "channels": {
            "task_channels": len(CHRONOS_CHANNEL_NAMES),
            "calendar_channels": len(CALENDAR_CHANNEL_NAMES),
            "economic_features": len(FEATURE_NAMES),
            "preprocessing_columns": len(PREPROCESSING_COLUMNS),
        },
        "registered_variants": sorted(REGISTERED_VARIANTS),
        "environment": environment.to_dict(),
        "absent_optional_packages": missing,
        "problems": problems,
        "notes": [
            "Absent provider or model packages are expected: the research pipeline "
            "and its integrity tests run offline.",
            "A live fetch needs the 'provider' extra; a forecast needs the 'model' "
            "extra and the pinned checkpoint revision.",
        ],
    }
    _emit(payload, as_json=args.json)
    return 1 if problems else 0


def _schedule(args: argparse.Namespace, config) -> int:
    """Report the chronology a given data edge supports."""
    from .protocol import InsufficientHistory, build_schedule

    calendar = ExchangeCalendar(config.runtime.exchange_calendar)
    latest = dt.date.fromisoformat(args.latest_session)
    earliest = (
        dt.date.fromisoformat(args.earliest_origin) if args.earliest_origin else None
    )
    diagnostic = earliest is not None and earliest < config.validation.earliest_primary_origin

    try:
        schedule = build_schedule(
            calendar,
            config,
            latest_data_session=latest,
            earliest_origin=earliest,
            require_folds=False,
        )
    except InsufficientHistory as exc:  # pragma: no cover - require_folds=False
        print(f"insufficient history: {exc}", file=sys.stderr)
        return 1

    payload = dict(schedule.describe())
    payload["earliest_origin"] = (earliest or config.validation.earliest_primary_origin).isoformat()
    payload["latest_data_session"] = latest.isoformat()
    payload["label"] = (
        "DIAGNOSTIC: origins precede the frozen checkpoint, so this is not evidence "
        "of historical deployability"
        if diagnostic
        else "post-checkpoint study window"
    )
    _emit(payload, as_json=args.json)
    return 0


def _audit(args: argparse.Namespace, config) -> int:
    """Print, and optionally execute, the data audit of build-order step 2."""
    from .universe import CandidateRoster, UniverseError

    try:
        roster = CandidateRoster.from_csv(args.roster)
    except UniverseError as exc:
        print(f"roster error: {exc}", file=sys.stderr)
        return 2

    calendar = ExchangeCalendar(config.runtime.exchange_calendar)
    today = dt.date.today()
    hourly_start = today - dt.timedelta(days=config.data.hourly_request_lookback_days)
    archive_start = today - dt.timedelta(days=config.data.archive_initial_lookback_days)
    context_symbols = sorted(
        {config.runtime.market_proxy, *(c.sector_etf for c in roster.candidates)}
    )

    plan: dict[str, object] = {
        "roster_source": roster.source,
        "roster_hash": roster.hash(),
        "candidates": len(roster),
        "sectors": sorted(roster.sectors()),
        "context_symbols": context_symbols,
        "requests": [
            {
                "interval": config.data.hourly_interval,
                "start": hourly_start.isoformat(),
                "end": today.isoformat(),
                "note": "inside the provider's ~730 calendar-day hourly window",
            },
            {
                "interval": "1d",
                "start": (
                    today - dt.timedelta(days=365 * config.data.daily_request_years)
                ).isoformat(),
                "end": today.isoformat(),
                "note": "slow features, risk estimates and the liquidity screens",
            },
            {
                "interval": config.data.archive_interval,
                "start": archive_start.isoformat(),
                "end": today.isoformat(),
                "note": "forward archive only; not a version-1 signal input",
            },
        ],
        "request_options": {
            "prepost": config.data.prepost,
            "auto_adjust": config.data.auto_adjust,
            "actions": config.data.actions,
            "repair": config.data.automatic_repair,
            "max_parallel_requests": config.data.max_parallel_requests,
            "max_retries": config.data.max_retries,
        },
        "checks": [
            "history depth per symbol against the expected bar schedule",
            "timestamp anchoring of every intraday bar",
            "split and volume convention against known corporate actions",
            "dividend-adjusted close kept separate from execution OHLC",
            "eligibility gates in as-of units",
        ],
        "calendar": calendar.provenance(),
        "unresolved_before_this_step": [
            "actual usable bars per symbol",
            "provider revisions",
            "the current 50-issuer eligibility list",
            "installed package compatibility",
            "measured runtime and memory",
        ],
    }

    if not args.live:
        plan["mode"] = "dry run: no request was made"
        _emit(plan, as_json=args.json)
        return 0

    try:
        from .collector import Ingestor, YahooCollector
        from .storage import SnapshotStore
    except Exception as exc:  # pragma: no cover - optional extra
        print(f"live audit unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        collector = YahooCollector(
            max_parallel_requests=config.data.max_parallel_requests
        )
    except Exception as exc:
        print(f"provider unavailable: {exc}", file=sys.stderr)
        print(
            "install the 'provider' extra before a live audit; the dry run above is "
            "the plan it would execute",
            file=sys.stderr,
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(args.out / "ledger.sqlite")
    ingestor = Ingestor(
        calendar=calendar,
        ledger=ledger,
        snapshots=SnapshotStore(args.out / "snapshots"),
        collector=collector,
    )
    from .collector import FetchRequest

    records = []
    for symbol in [*context_symbols, *(c.symbol for c in roster.candidates)]:
        for interval, start in (
            (config.data.hourly_interval, hourly_start),
            ("1d", today - dt.timedelta(days=365 * config.data.daily_request_years)),
        ):
            record = ingestor.ingest(
                FetchRequest(
                    symbol=symbol,
                    interval=interval,
                    start=start,
                    end=today,
                    prepost=config.data.prepost,
                    auto_adjust=config.data.auto_adjust,
                    actions=config.data.actions,
                    repair=config.data.automatic_repair,
                )
            )
            records.append(
                {
                    "symbol": record.symbol,
                    "interval": record.interval,
                    "status": record.status,
                    "rows": record.rows,
                    "anchoring_problems": (
                        len((record.anchoring or {}).get("unexpected", []))
                        + len((record.anchoring or {}).get("missing", []))
                    ),
                    "message": record.message,
                }
            )
    ledger.close()
    plan["mode"] = "live"
    plan["results"] = records
    plan["artifacts"] = str(args.out)
    _emit(plan, as_json=args.json)
    return 0 if all(row["status"] == "ok" for row in records) else 1


def _smoke(args: argparse.Namespace, config) -> int:
    """Checkpoint compatibility: shapes, quantile ordering and dates.

    This is a non-performance smoke run. Per the protocol it must not be used
    to tune a trading rule, and it only runs after the integrity invariants
    pass.
    """
    from .forecaster import Chronos2Forecaster, ForecastError

    forecaster = Chronos2Forecaster(
        model_id=config.model.id,
        revision=config.model.revision,
        quantile_levels=config.model.quantile_levels,
        batch_size_channels=config.model.batch_size_channels,
        dtype=config.model.dtype,
        cross_learning=config.model.cross_learning,
        expected_sha256=config.model.safetensors_sha256,
    )
    try:
        api = forecaster.describe_pipeline_api()
    except ForecastError as exc:
        payload = {
            "status": "unavailable",
            "reason": str(exc),
            "next_step": (
                "install the 'model' extra, then re-run; do not substitute another "
                "checkpoint or another entry point"
            ),
        }
        _emit(payload, as_json=args.json)
        return 2

    calendar = ExchangeCalendar(config.runtime.exchange_calendar)
    from .fixtures import make_default_market
    from .features import build_chronos_task, recommended_panel_bars

    origin = calendar.previous_sessions(dt.date.today(), 1)[0]
    bars = max(
        recommended_panel_bars(config.model.context_length),
        config.universe.common_hourly_history_bars,
    )
    sessions_needed = bars // 4 + 40
    market = make_default_market(
        calendar,
        start=calendar.previous_sessions(origin, sessions_needed)[0],
        end=origin,
    )
    task = build_chronos_task(
        symbol="AAA",
        origin_session=origin,
        calendar=calendar,
        stock=market.panel_ending_at("AAA", origin, bars),
        market=market.panel_ending_at("SPY", origin, bars),
        sector=market.panel_ending_at("XLA", origin, bars),
        context_length=config.model.context_length,
        horizon_sessions=config.model.horizon_sessions,
    )
    import time

    started = time.perf_counter()
    result = forecaster.predict([task])[0]
    elapsed = time.perf_counter() - started

    payload = {
        "status": result.status.value,
        "detail": result.detail,
        "pipeline_api": api,
        "provenance": forecaster.provenance(),
        "shape": list(result.paths.shape),
        "expected_shape": [
            len(config.model.quantile_levels),
            task.prediction_length,
        ],
        "terminal_indices": list(result.terminal_indices),
        "horizon_sessions": [bar.session.isoformat() for bar in task.horizon_bars],
        "diagnostics": dict(result.diagnostics),
        "measured_seconds_single_task": round(elapsed, 3),
        "notes": [
            "Synthetic input: this measures compatibility, shapes and runtime only.",
            "It is not a forecast of anything and must not be used to tune a rule.",
        ],
    }
    _emit(payload, as_json=args.json)
    return 0 if result.usable else 1


def _demo_study(args: argparse.Namespace, config) -> int:
    """An offline plumbing run against synthetic data.

    Present so the wiring can be exercised without a network or a checkpoint.
    Its numbers describe a seeded random walk scored by an arithmetic fixture,
    and carry no information about markets.
    """
    import datetime as _dt

    from .fixtures import SyntheticSpec, SyntheticMarket
    from .forecaster import DeterministicStubForecaster
    from .operations import render_markdown, run_study
    from .pipeline import FixtureMarketSource, ResearchPipeline
    from .protocol import build_schedule
    from .universe import Candidate, CandidateRoster, select_watchlist

    calendar = ExchangeCalendar(config.runtime.exchange_calendar)
    symbols = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
    labels = ("alpha", "beta", "gamma")
    sectors = {s: labels[i % len(labels)] for i, s in enumerate(symbols)}
    sector_etfs = {label: f"XL{label[0].upper()}" for label in labels}
    end = _dt.date(2025, 12, 19)
    spec = SyntheticSpec(
        symbols=symbols,
        sectors=sectors,
        market_proxy=config.runtime.market_proxy,
        sector_etfs=sector_etfs,
        start=_dt.date(2023, 6, 1),
        end=end,
        seed=424242,
        base_price={s: 40.0 + 11.0 * i for i, s in enumerate(symbols)},
    )
    market = SyntheticMarket(calendar, spec)
    roster = CandidateRoster(
        candidates=tuple(
            Candidate(
                symbol=s,
                issuer_id=f"ISSUER-{s}",
                exchange="XNYS",
                sector=sectors[s],
                sector_etf=sector_etfs[sectors[s]],
            )
            for s in symbols
        ),
        source="synthetic demo roster",
        declared_at=_dt.datetime.now(_dt.timezone.utc),
    )
    dollar_volume = {}
    for symbol in symbols:
        daily = market.daily(symbol).up_to(end)
        dollar_volume[symbol] = float(
            (daily.column("close")[-20:] * daily.column("volume")[-20:]).mean()
        )
    watchlist = select_watchlist(
        roster,
        config=config,
        selection_session=end,
        dollar_volume=dollar_volume,
    )

    spec_variant = REGISTERED_VARIANTS[args.variant]
    forecaster = (
        DeterministicStubForecaster(
            quantile_levels=config.model.quantile_levels,
            batch_size_channels=config.model.batch_size_channels,
        )
        if spec_variant.uses_forecast
        else None
    )
    pipeline = ResearchPipeline(
        config=config,
        calendar=calendar,
        source=FixtureMarketSource(market),
        watchlist=watchlist,
        forecaster=forecaster,
    )
    # A diagnostic window: these origins precede the frozen checkpoint, so the
    # run is labelled accordingly and is never post-checkpoint evidence.
    schedule = build_schedule(
        calendar,
        config,
        latest_data_session=end,
        earliest_origin=_dt.date(2024, 7, 1),
        require_folds=False,
    )
    if not schedule.folds:
        print("no folds available for the demo window", file=sys.stderr)
        return 1

    study = run_study(
        pipeline=pipeline,
        schedule=schedule,
        variant=args.variant,
        checkpoint="offline-demo",
        fold_index=0,
        bootstrap_samples=args.bootstrap_samples,
        notes=(
            "SYNTHETIC DATA. Seeded random walk, arithmetic forecaster fixture, "
            "pre-checkpoint diagnostic window. Produces no evidence about markets.",
        ),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = study.write(args.out / f"demo-study-{args.variant}.json")
    markdown = render_markdown(study)
    markdown_path = args.out / f"demo-study-{args.variant}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    if args.json:
        _emit(
            {
                "artifacts": [str(json_path), str(markdown_path)],
                "summary": study.candidate.summary(),
                "gates_passed": study.gates.passed,
            },
            as_json=True,
        )
    else:
        print(markdown)
        print(f"artifacts: {json_path}, {markdown_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
