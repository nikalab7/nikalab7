# Can the registered design produce a credible positive result?

**Analysis of 2 October 2026, before any market data has been examined.** Nothing here
is a performance claim: every number comes from synthetic markets or simulation, and
the scripts are in [`analysis/`](../analysis/README.md).

## Summary

- **The pipeline works end to end.** Given a large planted edge, it learns it, calibrates,
  alerts, trades, and the date-block bootstrap confirms it at the registered confidence
  levels. No silent defect is suppressing signals.
- **Under the registered rules, a realistic edge produces no alerts and cannot be
  confirmed.** On real data the expected outcome is "no edge demonstrated" whether or
  not Chronos-2 adds information. That is a valid result, but the experiment, as
  registered, can hardly produce any other.
- Three things cause this: an alert rule that only very large edges can reach, an
  evaluation sample too small for the portfolio test, and a development history too
  short for the decision heads to learn a moderate effect.

## 1. Positive control: an edge that is really there

A synthetic market of 40 stocks in 8 sectors carries a planted, causal, persistent
relative-strength effect: each stock has a latent daily drift following a stationary
AR(1) with persistence 0.95, visible only through its own returns, absent from
overnight gaps. The run mirrors the real chronology — 224 labelled origins, two
development folds, a 60-origin final test — and uses the deterministic stub forecaster,
so it tests the decision heads, the alert rule, the portfolio and the statistics rather
than Chronos-2. Three alert rules are compared on the same fitted models:

- **registered**: p ≥ 0.60, estimated net return ≥ 0.30%, positive at stress costs;
- **economic**: p ≥ 0.50 and a positive estimated net return;
- **rank-only**: always take the top-ranked names the capacity rules allow.

Results for C256 (B0 behaves the same, since the stub forecaster adds nothing beyond
B0's features). "Best IC" is the ceiling from section 2; "learned IC" is the
out-of-sample cross-sectional rank IC of the fitted heads.

| Planted edge | Best IC | Learned IC, development → test | Registered rule: trades, development / test | Lower bound above zero (90% / 95%) |
| --- | --- | --- | --- | --- |
| none | 0 | −0.07 → +0.04 | 1 / 0 | no / no |
| 10 bps/day | 0.03 | −0.08 → +0.03 | 1 / 0 | no / no |
| **20 bps/day** | **0.10** | **−0.02 → +0.06** | **3 / 0** | **no / no** |
| 40 bps/day | 0.28 | +0.21 → +0.29 | 43 / 64, +0.61% net per trade | yes / yes |

What the rows show:

- **With no edge**, the registered rule correctly stays out, and the looser rules lose
  money to costs: the negative control holds.
- **At 20 bps/day** the edge is real and tradable — a well-informed model nets about
  +0.13% per trade (section 2) — yet the registered rule makes three trades in
  development and **none in the final test**. The heads, fitted on 80–100 development
  sessions, learn almost nothing (IC −0.02); with the longer refit history of the test
  they reach 0.06 of a possible 0.10, still short of the bar. Under the economic rule
  C256 made +0.38% per trade over 39 test trades, and the 95% lower bound still stayed
  just below zero (−0.002% a day): a profitable rule, unconfirmable in 60 sessions.
- **Only at 40 bps/day** — an IC near 0.3, far beyond what short-horizon large-cap
  signals achieve — does the registered rule trade, and then both lower bounds clear
  zero. The machinery itself is sound.

## 2. The ceiling: what any model could do with that edge

The optimal causal estimate of the planted drift is a Kalman filter. Its quality bounds
every model, Chronos-2 included:

| Planted edge | Best achievable IC | Best top-3-of-40 trade, gross | After 0.20% round-trip costs |
| --- | --- | --- | --- |
| 10 bps/day | 0.03 | +0.10% | −0.10% |
| 20 bps/day | 0.10 | +0.33% | +0.13% |
| 40 bps/day | 0.28 | +0.93% | +0.73% |

An information coefficient of 0.10 is already exceptional for short-horizon signals on
large US stocks. Even then, the best trades net about +0.13% over two sessions. A
calibrated hit probability of 0.60 over a two-session hold with about 1.7% noise
requires an expected net return near +0.4% per trade — an edge close to the
unrealistic last row.

## 3. Power: what the registered sample sizes can confirm

Probability that the lower bound of the mean daily portfolio return clears zero, for a
true net edge per trade, with real-world large-cap volatility (1.8% daily, 1.0% market)
and the registered capacity (three positions of 10%). The bound is a normal
approximation of the registered block bootstrap, which is close here because
overlapping two-session holds leave the daily series only weakly autocorrelated:

| True net edge per trade | Development (40 sessions, 90%) | Final test (60, 95%) | 120 sessions | 250 sessions |
| --- | --- | --- | --- | --- |
| 0.05% | 6% | 4% | 4% | 5% |
| 0.10% | 8% | 6% | 7% | 9% |
| 0.20% | 11% | 9% | 13% | 23% |
| 0.40% | 25% | 21% | 37% | 67% |
| 0.70% | 51% | 53% | 82% | 99% |

With no edge the test rejects 2.25% of the time, against a nominal 2.5%. Even an
extraordinary +0.40% per trade is confirmed by the final test about one time in five.

The same dates carry far more information when every eligible stock is used. Power of a
test on the mean daily cross-sectional rank IC (50 names per date):

| True IC | Development (40 dates, 90%) | Final test (60 dates, 95%) |
| --- | --- | --- |
| 0.02 | 23% | 19% |
| 0.03 | 36% | 33% |
| 0.05 | 68% | 70% |
| 0.10 | 99% | 100% |

The pipeline already computes this IC, but only as a diagnostic.

## 4. Options for the protocol owner

These change the registered protocol and must be decided — and recorded in
[AMENDMENTS.md](AMENDMENTS.md) under a new `design_version` — **before any real outcome
is examined**. None has been, so each is still legitimate.

1. **Answer the research question with an instrument that can answer it.** Make the
   paired date-block bootstrap of the daily rank IC, candidate minus B0, the primary
   development and final-test criterion for "does Chronos-2 add information". Keep the
   portfolio, its thresholds and all eight gates unchanged for any trading claim.
2. **Make the alert rule reachable**, for example a positive estimated net return after
   stress costs, ranked by estimate over volatility, so that the paper portfolio trades
   and evidence accumulates. The qualified label would still require every gate.
3. **Collect more evaluation sessions.** Lengthen the forward paper period well beyond
   60 sessions and use the full 50-issuer universe. Power grows with the square root of
   the sample, slowly.
4. **Keep the registration as it is**, and accept "no edge demonstrated" as the likely
   and honest outcome.

These are not mutually exclusive; 1 and 3 together address the research question
without loosening anything a trading claim depends on.
