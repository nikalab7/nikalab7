# chronos2-signal

A free-data, long-only **research scanner** for a frozen watchlist of up to 50 liquid
US common stocks, built to the preregistered protocol in
[`docs/DESIGN.md`](docs/DESIGN.md).

> **Status: `design_only_unvalidated`. No edge is claimed and none has been tested.**
>
> No market data has been downloaded, no Chronos-2 weights have been loaded, no
> forecast has been produced by the real checkpoint, and no notification channel is
> configured. Every output is labelled `RESEARCH / UNVALIDATED`, and the label fails
> closed. See [`docs/STATUS.md`](docs/STATUS.md) for the full account of what does and
> does not exist.

The scanner reads completed regular-session hourly bars, runs once after each exchange
session, and evaluates a fixed two-session holding period: enter at the next session's
official open, exit at the second following session's official close. **Zero alerts is
a valid outcome**, and the code reports it rather than lowering a threshold.

## The hypothesis, and what would count as an edge

The research hypothesis is *conditional continuation of relative strength*: under some
market conditions, a stock's strength relative to its sector, supported by trading
activity and a useful conditional forecast, may persist into the next two sessions.
This is a hypothesis to reject or retain, not an asserted inefficiency.

An edge means a positive expected return **after realistic costs, under the exact alert
and execution policy**, plus useful improvement over simple alternatives. A positive
price forecast, a high hit rate, a general forecasting benchmark, or a profitable
rising-market backtest is not sufficient. The protocol is designed so that "Chronos-2
adds no useful information" is a reachable conclusion.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Two optional extras are deliberately absent by default. The research pipeline and all
of its integrity tests run offline without them:

```bash
python -m pip install -e '.[provider]'   # yfinance, for a live data audit
python -m pip install -e '.[model]'      # chronos-forecasting + torch
```

Versions are pinned in `requirements.txt` to what the test suite actually ran against.
The provider and model extras are pinned later, at the data audit, before any result is
examined — see [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Commands

```bash
chronos2-signal verify                                   # self-check and provenance
chronos2-signal schedule --latest-session 2026-09-25     # the chronology available
chronos2-signal audit --roster config/roster.csv         # data-audit plan (dry)
chronos2-signal audit --roster config/roster.csv --live   # actually fetch
chronos2-signal smoke                                    # checkpoint compatibility
chronos2-signal demo-study --variant C256                # offline plumbing run
```

`verify` confirms the code and the registered configuration agree, and prints the
environment fingerprint. `schedule` reports how many folds a given data edge supports —
and refuses to shrink a block when the answer is "not enough". `demo-study` runs the
whole pipeline on synthetic data so the wiring can be exercised without a network; its
own report says that its numbers describe a seeded random walk.

## Tests

```bash
pytest -q
```

90 tests, fully offline. The twelve integrity invariants of the protocol's section 16
are in `tests/test_invariants.py` and are mapped to what they protect in
[`docs/INVARIANTS.md`](docs/INVARIANTS.md). Among other things they prove that
appending future prices cannot change an earlier origin's features, that a split
creates no wealth, that the overnight gap contributes no captured return, that
reordering forecast tasks does not change a prediction, and that the candidate and its
baselines share one date index, one set of fills and one cost model.

`tests/test_system.py` checks the same guarantees **through the pipeline** —
`build_batch`, `simulate`, the operations entry points — rather than by calling each
component directly. That distinction is the point: an earlier version of this repository
had correct components that the orchestration never called, and a component-level suite
stayed green regardless. Each system test was mutation-checked by reintroducing the
defect it guards against and confirming that it fails.

## Layout

```
config/design.yaml            the registered protocol, machine-readable
src/chronos2_signal/
  config.py         strict loader; rejects drift from the protocol
  calendar_spec.py  sessions, hourly bars, half days, DST, horizons
  collector.py      bounded provider fetches, immutable snapshots, suppression
  storage.py        SQLite ledger (append-only forecasts) + Parquet snapshots
  holdout.py        the final-test lock
  market.py         panels aligned to the expected bar schedule
  sources.py        session-bounded, split-consistent views: fixture and ledger
  actions.py        split audit, as-of units, wealth accounting
  quality.py        per-origin eligibility, one mask for every variant
  universe.py       candidate roster, selection rule, freeze guard
  features.py       12 task channels; 20 features + 1 missingness flag
  forecaster.py     frozen-checkpoint adapter; deterministic offline stub
  forecast_store.py forecasts recorded to the append-only ledger before outcomes
  decision.py       ridge + logistic heads, disjoint-block Platt calibration
  policy.py         thresholds, ranking, capacity, deduplication
  portfolio.py      reference paper portfolio and its accounting
  protocol.py       labelled origins, folds, purges, holdout, refits
  variants.py       the five registered variants and the fixed controls
  evaluation.py     metrics, paired date-block bootstrap, promotion gates
  pipeline.py       origin batches and labels, guard-checked
  simulation.py     one session engine and one scoring path for every system
  operations.py     after-close run, development comparison, study, final test
  notifier.py       renders persisted signals; contacts nothing
  cli.py            command line entry points
  fixtures.py       deterministic synthetic market for tests
docs/                 DESIGN, STATUS, BUILD_ORDER, INVARIANTS, ENVIRONMENT, AMENDMENTS
```

## Design choices worth knowing before reading the code

- **Bars, not hours.** Two trading sessions are 14 hourly bars on a normal pair and
  fewer around a half day. Prediction length and both terminal indices come from the
  exchange calendar, never from a fixed offset. A normal session is seven bars, the
  last of which lasts 30 minutes.
- **The `mean` output is the median.** It is stored as `forecast_median` and never as
  an expected profit. A quantile is not a calibrated probability, and marginal
  quantiles at several future times do not describe a joint path.
- **Crossed quantiles block a forecast.** They are not silently sorted, because that
  would report an output the model did not produce.
- **Batch size counts channels, not symbols**, and `cross_learning` is off — otherwise
  a prediction would depend on which other stocks shared its batch.
- **The overnight gap is not captured.** The forecast is measured from the D0 close but
  entry is the D1 open, so the gap is an input feature only.
- **Splits are audited, not assumed.** `auto_adjust=False` does not prove a field is an
  untouched quote. A split that cannot be resolved quarantines the episode rather than
  manufacturing a return, and screens run in as-of units so a later split cannot fail a
  price floor the stock actually passed. A position held across an ex-date is
  re-expressed in the new units, never marked in two.
- **Dates are the unit of uncertainty.** Fifty stocks on one day are not fifty
  independent observations; every interval resamples date blocks, paired across
  competing systems.
- **The holdout is locked on every path.** Batch construction, labels, realised
  outcomes, simulation and ledger reads all check the guard, so development-mode access
  to a reserved origin raises before any data is read. Studies then *verify* the result
  — every origin they touched against the reserved set — and report the count, instead
  of asserting that nothing was read.
- **One engine, one scoring path.** Candidates, B0 and the momentum control run through
  the same session loop, and the backtest scores origins with the same function as the
  live after-close run. A backtest therefore describes the code that would run.

## Limitations that implementation cannot fix

Today's surviving watchlist carries survivorship and selection bias, so results on it
are conditional watchlist evidence. A post-checkpoint window reduces pretraining
contamination but does not remove vendor revisions or a researcher's prior exposure to
these years. The clean history currently supports two development folds in one broad
market regime, and a bootstrap reports sampling variability within the observed history
rather than inventing regimes it never saw. Unscheduled news and overnight gaps remain
material unmodelled risks, and the p10 quantile is not a loss bound.

## Scope

No broker connection, no automatic orders, no paid API, no cloud endpoint and no LLM
call. This is research code for evaluating a hypothesis. It is not investment advice
and its allocations are evaluation conventions, not a recommendation for anyone's
account.
