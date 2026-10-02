"""Positive control: does the registered protocol find an edge that is really there?

A synthetic market with a planted, causal, persistent relative-strength effect:
every stock carries a latent daily drift alpha_d that follows a stationary AR(1)
(std ``alpha_bps``, persistence ``phi``), spread evenly over the session's bars.
Nothing about alpha is visible except through the stock's own returns, so the
only way to profit from it is the hypothesis the design states: relative
strength persists. Overnight gaps carry no alpha, so the reference strategy
cannot capture it outside its holding window.

The run mirrors the real protocol's chronology (about 224 labelled origins: two
development folds and a 60-origin final test) and measures, for three alert
rules, what the pipeline does with the edge.

Run: ``python analysis/positive_control.py --alpha 20 --out pc_a20.json``
(about 15 minutes for 40 stocks). See analysis/README.md.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import math
import sys
import time

import numpy as np

from chronos2_signal.calendar_spec import ExchangeCalendar
from chronos2_signal.config import load_design
from chronos2_signal.evaluation import paired_block_bootstrap, predictive_report, trade_metrics
from chronos2_signal.fixtures import SyntheticMarket, SyntheticSpec
from chronos2_signal.forecaster import DeterministicStubForecaster
from chronos2_signal.holdout import AccessMode
from chronos2_signal.operations import _fold_schedule, _guarded, _label_predictions, _refit_schedule
from chronos2_signal.pipeline import ResearchPipeline
from chronos2_signal.policy import PolicyEngine
from chronos2_signal.protocol import build_schedule, label_available_at
from chronos2_signal.simulation import PolicySelector, WalkForwardRunner, simulate
from chronos2_signal.sources import FixtureMarketSource
from chronos2_signal.universe import Candidate, CandidateRoster, select_watchlist
from chronos2_signal.variants import cash_series


class PlantedMarket(SyntheticMarket):
    """The fixture market plus a latent, persistent per-stock daily drift."""

    def __init__(self, calendar, spec, *, alpha_bps: float, phi: float) -> None:
        self._alpha_bps = alpha_bps
        self._phi = phi
        self._alpha_cache: dict[str, np.ndarray] = {}
        super().__init__(calendar, spec)
        self.spec_hash = f"{self.spec_hash}-alpha{alpha_bps}-phi{phi}"

    @property
    def planted(self) -> bool:
        return self._alpha_bps != 0.0

    def daily_alpha(self, symbol: str) -> np.ndarray:
        if symbol not in self._alpha_cache:
            rng = self._rng("planted-alpha", symbol)
            sigma = self._alpha_bps / 10_000.0
            innovation = sigma * math.sqrt(max(0.0, 1.0 - self._phi**2))
            alpha = np.empty(len(self.sessions))
            alpha[0] = rng.normal(0.0, sigma)
            for day in range(1, len(self.sessions)):
                alpha[day] = self._phi * alpha[day - 1] + rng.normal(0.0, innovation)
            self._alpha_cache[symbol] = alpha
        return self._alpha_cache[symbol]

    def _log_path(self, symbol):
        path, steps = super()._log_path(symbol)
        if symbol not in self.spec.symbols or self._alpha_bps == 0.0:
            return path, steps
        alpha = self.daily_alpha(symbol)
        index = {session: i for i, session in enumerate(self.sessions)}
        counts = collections.Counter(bar.session for bar in self.bars)
        per_bar = np.asarray(
            [alpha[index[bar.session]] / counts[bar.session] for bar in self.bars], dtype=float
        )
        return path + np.cumsum(per_bar), steps + per_bar


def thresholds(config, rule: str):
    if rule == "registered":
        return config
    decision = config.decision
    if rule == "economic":
        decision = dataclasses.replace(
            decision, min_probability=0.50, min_estimated_net_return=1e-9,
            require_positive_stress_estimate=False,
        )
    elif rule == "rank_only":
        decision = dataclasses.replace(
            decision, min_probability=0.0, min_estimated_net_return=-1.0,
            require_positive_stress_estimate=False,
        )
    else:
        raise ValueError(rule)
    return dataclasses.replace(config, status="plumbing_test_only_not_registered", decision=decision)


def summarise(result, pipeline, *, confidence: float, samples: int) -> dict:
    metrics = trade_metrics(result.trades, origins_scanned=result.origins_scanned)
    cash = cash_series("cash", result.daily.sessions)
    boot = paired_block_bootstrap(
        [result.daily, cash], block_sessions=10, samples=samples, confidence=confidence,
        differences=[(result.daily.name, "cash")],
    )
    estimate = boot[result.daily.name]
    return {
        "alerts": result.alerts,
        "trades": metrics.trades,
        "signal_sessions": metrics.distinct_sessions,
        "mean_net_trade_return": metrics.mean_net_return,
        "stress_mean_net_trade_return": metrics.stress_mean_net_return,
        "win_rate": metrics.win_rate,
        "portfolio_mean_daily": float(np.mean(result.daily.returns)),
        "portfolio_total_return": float(np.prod(1.0 + result.daily.returns) - 1.0),
        "portfolio_sessions": len(result.daily.sessions),
        "bootstrap_lower": estimate.lower,
        "bootstrap_upper": estimate.upper,
        "bootstrap_confidence": confidence,
        "max_drawdown": result.portfolio.summary()["max_drawdown"],
    }


def score_distribution(predictions) -> dict:
    p = np.asarray([r.calibrated_probability for r in predictions], dtype=float)
    e = np.asarray([r.estimated_net_return for r in predictions], dtype=float)
    if not p.size:
        return {"rows": 0}
    return {
        "rows": int(p.size),
        "p_max": float(p.max()),
        "p_q99": float(np.quantile(p, 0.99)),
        "p_q50": float(np.quantile(p, 0.50)),
        "share_p_ge_060": float(np.mean(p >= 0.60)),
        "est_max": float(e.max()),
        "est_q99": float(np.quantile(e, 0.99)),
        "share_est_ge_0003": float(np.mean(e >= 0.003)),
        "share_both": float(np.mean((p >= 0.60) & (e >= 0.003))),
    }


def oracle_edge(market: PlantedMarket, pipeline, sessions, phi: float) -> dict:
    """What a perfect knower of today's alpha could expect from its top three."""
    calendar = pipeline.calendar
    index = {session: i for i, session in enumerate(market.sessions)}
    symbols = list(market.spec.symbols)
    gross = []
    for origin in sessions:
        timing = label_available_at(calendar, origin, horizon_sessions=2)
        day = index[origin]
        if not market.planted:
            gross.append(0.0)
            continue
        expected = {s: market.daily_alpha(s)[day] * (phi + phi**2) for s in symbols}
        top = sorted(symbols, key=lambda s: -expected[s])[:3]
        e1, e2 = index[timing.entry_session], index[timing.exit_session]
        realised = [float(market.daily_alpha(s)[e1] + market.daily_alpha(s)[e2]) for s in top]
        gross.append(float(np.mean(realised)))
    return {"oracle_top3_mean_gross_alpha_2s": float(np.mean(gross)) if gross else float("nan")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True, help="stationary std of daily alpha, bps")
    parser.add_argument("--phi", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--symbols", type=int, default=40)
    parser.add_argument("--sectors", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--variants", default="B0,C256")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()

    config = load_design()
    calendar = ExchangeCalendar(config.runtime.exchange_calendar)
    labels = [chr(ord("a") + i) for i in range(args.sectors)]
    symbols = tuple(f"S{i:02d}" for i in range(args.symbols))
    sector_of = {s: labels[i % len(labels)] for i, s in enumerate(symbols)}
    etfs = {label: f"XL{label.upper()}" for label in labels}
    spec = SyntheticSpec(
        symbols=symbols, sectors=sector_of, market_proxy="SPY", sector_etfs=etfs,
        start=dt.date(2023, 10, 2), end=dt.date(2025, 9, 26), seed=1000 + args.seed,
        base_price={s: 40.0 + 3.0 * i for i, s in enumerate(symbols)},
    )
    market = PlantedMarket(calendar, spec, alpha_bps=args.alpha, phi=args.phi)
    roster = CandidateRoster(
        candidates=tuple(
            Candidate(symbol=s, issuer_id=f"ISSUER-{s}", exchange="XNYS", sector=sector_of[s],
                      sector_etf=etfs[sector_of[s]], listing_history="synthetic")
            for s in symbols
        ),
        source="positive control", declared_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    selection = market.sessions[-1]
    dollar_volume = {}
    for s in symbols:
        daily = market.daily(s).up_to(selection)
        dollar_volume[s] = float((daily.column("close")[-20:] * daily.column("volume")[-20:]).mean())
    watchlist = select_watchlist(
        roster, config=config, selection_session=selection, dollar_volume=dollar_volume,
        frozen_at=dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc),
    )
    schedule = build_schedule(
        calendar, config, latest_data_session=dt.date(2025, 9, 26), earliest_origin=dt.date(2024, 10, 31)
    )
    pipeline = ResearchPipeline(
        config=config, calendar=calendar, source=FixtureMarketSource(market), watchlist=watchlist,
        forecaster=DeterministicStubForecaster(
            quantile_levels=config.model.quantile_levels,
            batch_size_channels=config.model.batch_size_channels,
        ),
    )
    variants = args.variants.split(",")
    report: dict = {
        "alpha_bps": args.alpha, "phi": args.phi, "seed": args.seed, "symbols": len(watchlist.members),
        "schedule": {
            "labelled": len(schedule.timings), "development": len(schedule.development_origins),
            "folds": len(schedule.folds), "test": len(schedule.test_origins),
        },
    }

    phases = {
        "development": (AccessMode.DEVELOPMENT, sorted({s for f in schedule.folds for s in f.validation_sessions}), 0.90),
        "final_test": (AccessMode.FINAL_TEST, list(schedule.test_origins), 0.95),
    }
    for phase, (mode, sessions, confidence) in phases.items():
        guarded = _guarded(pipeline, schedule, mode)
        runner = WalkForwardRunner(pipeline=guarded)
        cache: dict = {}
        schedules = {
            v: (_fold_schedule(runner, v, schedule.folds, cache) if mode is AccessMode.DEVELOPMENT
                else _refit_schedule(runner, v, schedule, cache))
            for v in variants
        }
        phase_report: dict = {"sessions": len(sessions), **oracle_edge(market, pipeline, sessions, args.phi)}
        for variant in variants:
            entry: dict = {}
            for rule in ("registered", "economic", "rank_only"):
                engine = PolicyEngine(thresholds(config, rule))
                selector = PolicySelector(
                    pipeline=guarded, variant=variant, schedule=schedules[variant], batches=cache, engine=engine
                )
                result = simulate(pipeline=guarded, selector=selector, score_sessions=sessions)
                entry[rule] = summarise(result, guarded, confidence=confidence, samples=args.samples)
                if rule == "rank_only":
                    entry["scores"] = score_distribution(result.predictions)
                    predictive = predictive_report(
                        _label_predictions(guarded, result.predictions),
                        quantile_levels=config.model.quantile_levels,
                    )
                    entry["rank_ic_mean"] = predictive.rank_ic_mean
                    entry["brier"] = predictive.brier
                    entry["base_rate_brier"] = predictive.base_rate_brier
            phase_report[variant] = entry
        report[phase] = phase_report
    report["seconds"] = round(time.time() - started, 1)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(json.dumps({"alpha": args.alpha, "seconds": report["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
