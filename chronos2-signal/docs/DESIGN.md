# Chronos-2 stock signal system — design and preregistered research protocol

**Prepared for Nika · 27 September 2026, Asia/Tbilisi · Version 1.0**

**Amended 3 October 2026 as `chronos2_hourly_v2`:** the development selection rule of
section 12 and the primary information test of section 14. Both changes are marked in
place; the original text and the reason are in [AMENDMENTS.md](AMENDMENTS.md).

**Status: registered protocol.** No market dataset was downloaded, no model weights
loaded, no forecasting or backtest run, and no notification schedule activated for
this document. Official documentation, source code and research were inspected. All
thresholds below are engineering starting assumptions, not empirically optimal
settings and not measured performance.

This file is the registered record. The code in `src/chronos2_signal/` cites its
section numbers, and `config/design.yaml` is the machine-readable form of section 19.
Changing any value here requires a protocol amendment recorded in
[AMENDMENTS.md](AMENDMENTS.md) and a new `design_version`.

---

## 1. Decision and definition of success

Build a free-data, long-only research scanner for a frozen watchlist of up to 50
liquid US common stocks. The primary model uses completed regular-session hourly
bars. It runs once after each exchange session and evaluates a fixed two-session
holding period. Its valid outcomes include zero alerts.

Initial configuration: Chronos-2, 256 hourly observations, log-price target,
market/sector/volume covariates, and a small regularized decision model. Archive
15-minute observations for a later experiment. Use daily history for regime features
and risk estimates, not an additional Transformer in version 1.

The research hypothesis is conditional continuation of relative strength: under some
market conditions, a stock's recent strength relative to its sector, supported by
trading activity and a useful conditional forecast, may persist into the next two
sessions. This is a hypothesis to reject or retain, not an asserted market
inefficiency. Recent strength may also reverse, and public information may already be
priced in.

An edge means positive expected return after realistic costs under the exact alert
and execution policy, plus useful improvement over relevant simple alternatives. A
positive price forecast, high hit rate, general forecasting benchmark, or profitable
rising-market backtest is insufficient. No architecture guarantees an edge. A failed
experiment must be allowed to conclude that Chronos-2 adds no useful information.

## 2. Sources and the free-data boundary

| Source | Use | Restriction or design consequence |
| --- | --- | --- |
| Yahoo Finance through yfinance | Hourly and daily OHLCV; actions; current event metadata | Unofficial access; gaps, revisions and access limits require audit and caching |
| Yahoo 15-minute series | Forward archive and later resolution experiment | Approximately last 60 calendar days currently available |
| Exchange calendar / pandas-market-calendars | Sessions, holidays, early closes and DST | Pin calendar version and test representative dates |
| SEC EDGAR | Timestamped filing metadata for future event annotations | Filings are observable after publication; they are not a complete advance earnings calendar |
| Federal Reserve and BLS calendars | Scheduled macro-event annotations | Archive what was known at the decision time |
| FRED/ALFRED, optional later | Low-frequency macro regime experiments | Use release availability and historical vintages, never revised data presented as contemporaneous |

yfinance's source treats 1h/60m history as a roughly 730-day window, while 15m uses
roughly 60 days [S4]. These are calendar days, not trading sessions, and not a
service guarantee. The generic download documentation's blanket intraday wording is
less specific than the implementation. Splitting requests cannot recover older data
the provider no longer retains.

Yahoo alone supplies the numerical version-1 signal. Other sources are optional
annotations until archived, point-in-time data support a separate evaluation. Do not
add scraped news sentiment, current fundamentals, analyst revisions or reconstructed
earnings-surprise features to an old backtest without proof of historical
availability.

No paid API, cloud endpoint or LLM call is required. Local CPU inference is the
fallback; an existing GPU is optional. Free notebook sessions can support experiments
but are not assumed to provide a reliable unattended scheduler. Compute runtime
remains unmeasured.

## 3. Why hourly is the primary resolution

| Resolution | Approximate regular-session observations/day | Available historical breadth | Version-1 decision |
| --- | --- | --- | --- |
| 15 minutes | 26 | About 40–43 sessions within 60 calendar days | Archive; insufficient regime breadth for selecting many configurations |
| Yahoo 60 minutes | 7, including a final 30-minute bar | Roughly 500 sessions within 730 calendar days | Primary input |
| Daily | 1 | Several years, depending on listing | Slow features, risk and diagnostic baseline |

The bar counts are exchange-session arithmetic, not a live Yahoo availability test.
At ingestion, validate the provider's actual timestamps against this expected schema.
Do not silently use differently anchored bars.

The hourly choice balances resolution, historical depth and execution cost for a
two-session trade. It does not assert that hourly is inherently best for Transformer
models. Fifteen-minute data can contain useful intraday information, but also noisier
movement; finer observations do not create additional independent market regimes.

Context candidates at hourly resolution:

| Bars | Approximate full sessions | Role |
| --- | --- | --- |
| 128 | 18.3 | Short-history challenger |
| 256 | 36.6 | Default |
| 512 | 73.1 | Longer-history challenger |

The corresponding 256 bars at 15m cover only about 9.8 sessions. Thus comparing 256
hourly with 256 quarter-hourly bars confounds sampling resolution with elapsed
history. A later fair comparison uses the same decision dates, approximately the same
session span, and the same exits/cost assumptions.

Longer context preserves hourly sampling; it adds older observations. It may supply
regime context or introduce stale dynamics. We will not assume that maximum context
is optimal. Nor will we assume attention can recover information absent from the
chosen bars.

## 4. Chronos-2 facts and their consequences

The official checkpoint is a 120M-parameter encoder-only model with time/group
attention and direct quantile outputs [S1, S2]. Its input patches have size/stride
16; maximum context is 8192. The checkpoint has 12 layers, 12 attention heads and
hidden size 768 [S3]. Those architecture values stay unchanged.

Patches contain the ordered observations; they are not simply average prices. A
16-observation hourly patch spans about 2.3 ordinary sessions. Enlarging context does
not enlarge this patch. Internal standardization and an arcsinh transform are
retained [S2]. These properties do not establish financial predictive power.

Important implementation traps found in the current pipeline [S5]:

- The output named `mean` is actually the median. Store it as `forecast_median`,
  never as expected profit.
- Batch size counts target and covariate series, not only stock symbols.
- `cross_learning=True` shares across tasks and makes results depend on batch
  composition. Default it to false; covariates within each stock task can still
  interact.
- The DataFrame convenience path assumes a regular timestamp grid. Use the
  array/dictionary interface with our explicit exchange-bar calendar instead.

A median or a quantile is not a calibrated success probability. Marginal quantiles at
several future times do not define a joint price path, maximum drawdown or the
probability that a take-profit beats a stop. Do not sum per-bar return quantiles or
subtract two marginal quantiles and call the result a valid return distribution.

## 5. Watchlist and eligibility

Before looking at forecast performance, freeze one manifest of up to 50 common-stock
issuers, with symbol, stable issuer identifier, exchange, sector proxy,
listing/rename history where known, and selection timestamp. Use a candidate roster
chosen independently of this experiment's returns. One share class per issuer; no
leveraged/inverse ETFs, options or penny-stock spike universe. ETFs are contextual
inputs, not the trade universe.

This document does not pretend to have screened a current 50-symbol list: actual
coverage, liquidity and issuer status require the future data audit. At that audit,
choose the 50 highest trailing-20-session dollar-volume eligible issuers from a
declared candidate roster, subject to at most 10 per sector; freeze before producing
any model performance metric. Underfilled sectors are acceptable. This chooses a
current watchlist, not a historically unbiased whole-market universe.

At each decision origin, require:

- As-of trade price at least USD 10; trailing-20-session median dollar volume at
  least USD 50 million.
- At least 200 valid daily observations and 512 scheduled hourly observations for a
  common experiment mask.
- At least 99% observed target bars in the last 512 expected bars; no missing target
  or required market/sector bars in the latest complete session.
- Valid issuer mapping, finite OHLCV, and no unresolved corporate-action discrepancy.

No minimum recent return, hand-picked winning tickers, ticker-specific thresholds or
ticker-specific model training. Relative strength is a feature, not a post hoc
membership rule. Symbols failing eligibility are skipped; do not replace them after
seeing results.

Today's surviving watchlist produces survivorship and selection limitations in
historical results. Without a point-in-time historical universe and delisting
outcomes, report only conditional watchlist evidence. New delistings, bankruptcies
and missing exit prices must remain in the forward ledger; never drop a failed trade
because its data disappeared.

## 6. Time, ingestion and adjustment contract

Internal timestamps are timezone-aware UTC; session logic uses America/New_York.
Display may use Asia/Tbilisi. Never hard-code a Tbilisi hour for the US close because
US DST changes the offset.

Schedule preparation at exchange close plus 30 minutes, normally 16:30 New York.
Accept only completed bars. The schedule is a design, not an activated task. A
provider lag can still exceed this buffer; snapshot freshness must be checked.

Normal expected Yahoo-style bar starts: 09:30, 10:30, 11:30, 12:30, 13:30, 14:30 and
15:30; ends: 10:30 through 15:30, then 16:00. The last bar has 30-minute duration.
Half days have a different calendar-derived count. Retain real session gaps; do not
synthesize overnight/weekend flat-price bars. Missing scheduled observations get
masks, not fake prices. Chronos receives the sequence in trading-bar order, with
calendar covariates exposing unequal elapsed time.

Initial retrieval intent: 1h with a safe request start inside the 730-day boundary
(for example, 729 days), 1d for eight years where available, and 15m inside 59 days.
Set `prepost=False`, explicit adjustment options, and request actions. Cache
overlapping incremental fetches; current results must not overwrite earlier decision
snapshots.

Store three distinct representations:

1. Immutable provider response, including retrieval time and declared adjustment
   mode.
2. Canonical, split-consistent feature series with the action provenance recorded.
3. Execution/accounting series and action ledger in the correct share/cash units.

`auto_adjust=False` does not establish that every returned field is an untouched
historical quote. Validate provider split conventions and volume units against known
actions. Never apply a split twice. Keep dividend-adjusted close separate from
execution OHLC; Yahoo documents that adjusted close incorporates splits and
distributions [S8].

Feature calculations must use actions effective/available at the origin. For
historical backfills, current vendor data cannot fully reproduce the original
provider vintage: flag that limitation. If split history permits reconstruction,
audit it; otherwise quarantine affected episodes rather than manufacture apparent
returns. Convert dividends and quantities consistently for P&L, and do not add
dividends again to an already total-return-adjusted label. Price/volume eligibility
must use as-of units, not a later split's nominal price.

For transient fetch failure: at most three retries with exponential backoff/jitter,
respect provider errors, and keep concurrency modest (initially two requests). On
rate limiting, stop aggressively retrying. If required SPY data fail, suppress the
entire scan; if a sector input fails, suppress that sector. Do not fill failures from
a stale snapshot and still call the signal current. No valid snapshot by close plus
120 minutes means `DATA_UNAVAILABLE` for that origin.

## 7. Exact input channels and inference settings

One task per stock. Default target at origin `t`:

```
y[j] = 100 * ln(C_split[j] / C_split[t]),  for past bars j
```

The final observed value is zero. This retains cumulative price structure while
putting forecasts into a common relative scale. All rebasing uses values already
known at `t`. Do not fit a full-dataset scaler. For a future target quantile `q`, the
implied price quantile is `C_split[t] * exp(q/100)`.

Past-only channels, aligned exactly to the target bars:

| Channel | Definition |
| --- | --- |
| Market price | SPY, same relative-log transform |
| Sector price | Frozen sector ETF, same transform |
| Relative volume | `log(current volume / median volume of the previous 20 matching session slots)`; exclude current observation from denominator |
| Range | `100 * ln(high/low)` |
| Close location | `2*(close-low)/(high-low)-1`; zero when `high=low` |
| Intrabar move | `100 * ln(close/open)` |

Match volume comparisons by slot and bar duration; omit incompatible
shortened-session observations from the denominator. Invalid/zero required volumes
trigger quality handling, not infinite features.

Five known-future calendar channels, with matching historical values:

1. Minutes since session open divided by 390.
2. Bar duration in minutes divided by 60.
3. Elapsed hours since the preceding scheduled bar end.
4. Sine of weekday position, using a seven-day cycle.
5. Cosine of weekday position, using a seven-day cycle.

Only the calendar extends into the future. Future SPY, sector, volume, high/low and
stock prices remain unknown. The default task has 12 channels: one target, six
past-only channels and five calendar channels.

| Setting | Initial choice |
| --- | --- |
| Checkpoint | `amazon/chronos-2` |
| Weight revision | `95a9710e2596287d08352589f42634fa5abdf0a7` [S6] |
| Inference API | `Chronos2Pipeline.predict_quantiles`, dictionary/array input |
| Context | 256; research alternatives 128 and 512 |
| Prediction length | Calendar-derived bars in next two sessions; normally 14 |
| Quantiles requested | 0.10, 0.25, 0.50, 0.75, 0.90 |
| Cross-learning across stock tasks | False |
| Channel batch budget | 128, reduced to 64/32 if memory requires |
| Numeric precision | Float32 initially, CPU or available CUDA GPU |
| Model state | Eval mode, inference/no-gradient mode |
| Fine-tuning | Disabled |
| Temperature / top-p / generated samples | Not knobs in this direct-quantile design |

Early-close horizons use the correct number of future bars, for example 11 if a
seven-bar session and a four-bar session follow. Both days' terminal indices come
from the calendar. Do not blindly equate 14 arbitrary wall-clock hours to two trading
days. Padded output values beyond the requested horizon are discarded.

Pin the actual package versions, source commit, calendar version and hardware
metadata during implementation, before any results are examined. Their versions are
intentionally not fabricated here. A compatibility smoke test must verify shape,
quantile ordering and dates. If the package cannot load the frozen weight revision
correctly, stop and resolve compatibility; do not silently switch weights.

## 8. Tradable target, costs and the overnight gap

Let D0 be the completed signal session. Entry reference is the next session's
official open, D1. Exit reference is the official close of the second following
session, D2. The decision is generated after D0 and does not know the D1 open.
Simulated participation assumes an order could be submitted before that open; manual
fills need their actual times/prices tracked separately.

For no corporate action during the hold, the base label is:

```
R_net = (C_D2 * (1-s_sell)) / (O_D1 * (1+s_buy)) - 1 - explicit_fee_fraction
```

For corporate actions, calculate the same wealth change using quantity adjustments
and dividends actually earned. Cash entitlement depends on holding through the
relevant ex-date boundary. The accounting engine, not a forecast quantile, determines
realized return.

Base spread/slippage assumption: 10 basis points per side; stress: 25 per side;
severe stress: 50 per side. Broker commissions and material fixed/FX charges are
additional when relevant. These are test assumptions, not measured spreads or
guaranteed execution. Report the break-even round-trip cost.

The Chronos terminal forecast relative to D0 close includes an overnight move that
may occur before entry. Therefore it is only an input feature. The second-stage
estimator learns the realized D1-open-to-D2-close return. Never count
D0-close-to-D1-open profit as captured by this strategy. Never subtract a median
future open from a median future close and call that the median holding return.

Version 1 uses a fixed time exit, no optimized stop or take-profit, and no look-ahead
gap rejection at the open. An intraday stop/gap filter would change the labels and
require a new predeclared execution experiment. Forecast p10 is not a guaranteed loss
bound. Record realized maximum adverse excursion for diagnosis, without using it as
an unearned exit price.

## 9. Small decision model before a flexible one

The earlier suggestion of immediately using LightGBM is revised. With roughly two
years of hourly history and much less history after the frozen checkpoint existed,
independent market dates are scarce. Start with regularized linear models. A flexible
nonlinear model remains a challenger for a later research version, not an automatic
upgrade.

Use two heads trained on exactly the same origin rows:

- **Return head:** Ridge regression, alpha 10, intercept enabled. Target is
  `R_net / sigma_2d`, where `sigma_2d = max(sqrt(2)*stdev(last 20 daily log returns),
  0.005)`. Multiply its prediction by current `sigma_2d` to obtain an estimated net
  return. This volatility scale is a feature normalization, not an assumption that
  returns are Gaussian.
- **Probability head:** Logistic regression, L2 regularization, `C=1`, lbfgs,
  `max_iter=2000`, no class rebalancing. Target is `1[R_net > 0]`. Fit sigmoid/Platt
  calibration on a later, disjoint time block. Do not use random-split calibration
  [S11].

For each head, fit feature medians and standardization only on its training window.
Sample weight per eligible row is `1 / eligible_stock_count_that_date`, so each date
has total weight one. This reduces domination by dates with more rows; it does not
make correlated stocks statistically independent. Clip standardized feature values to
`[-8,8]` under a fixed rule and record clipping frequency. Never trim realized test
losses. All required price/forecast features must be valid rather than silently
imputed.

Twenty predefined features:

| # | Feature |
| --- | --- |
| 1–2 | Chronos median forecast return at D1 and D2 close, divided by `sigma_2d` |
| 3 | D2 forecast p90–p10 width, divided by `sigma_2d` |
| 4 | D2 quantile asymmetry: `(p90+p10-2*p50)/max(p90-p10,1e-4)`, using return units |
| 5 | D2 median forecast return / `max(p90-p10,1e-4)` |
| 6 | Difference between D2 and D1 median returns, divided by `sigma_2d`; a curve-shape feature only |
| 7–9 | Stock trailing 1-, 5-, and 20-session returns |
| 10 | Stock minus sector trailing 5-session return |
| 11 | Current daily volume / preceding 20-session median volume |
| 12 | Stock realized daily volatility over 20 sessions |
| 13 | Ratio of 20-session to 60-session volatility |
| 14 | Distance from a 20-session EMA, divided by daily volatility |
| 15 | Drawdown from the preceding 20-session high |
| 16 | Trailing-60-session daily beta to SPY |
| 17–18 | SPY trailing 5-session return and 20-session volatility |
| 19 | Sector minus SPY trailing 5-session return |
| 20 | Fraction of eligible frozen-watchlist sector peers above their 20-session EMA |

Use the last completed daily session for all daily features. Calculate beta with an
intercept using past aligned returns; mask if variance is degenerate. Breadth is
explicitly watchlist breadth, not total-market breadth; mask sector breadth when
fewer than three eligible issuers exist. Add a fixed missingness indicator for that
optional breadth feature and median-impute it from training only. Thus the
preprocessing matrix has 21 columns, while the economic feature list has 20.

No issuer embeddings, current market cap, hundreds of overlapping indicators,
discretionary LLM score or hand-tuned per-ticker models in version 1. The
return/probability models do not receive the future opening price.

## 10. Alert policy and portfolio accounting

Default research thresholds, fixed before results:

1. All data/eligibility checks pass.
2. Calibrated estimated probability of positive net return is at least 0.60.
3. Estimated base-cost net return is at least 0.003, i.e. 0.30%.
4. Estimated return remains positive when repriced under the stress-cost assumption.
5. No active position for the same issuer; capacity/correlation rules pass.

These thresholds do not mean that 60% is achievable. If the estimator rarely or never
qualifies, report that. No forced daily top pick and no lowering thresholds to make a
dashboard look active.

Rank eligible candidates by estimated net return divided by `sigma_2d`, then
estimated net return, then symbol for deterministic ties. At most two new alerts per
origin, three simultaneously open positions, one per sector, and no newly added pair
with trailing-60-session daily correlation above 0.80. Missing required correlation
history blocks the new allocation.

For the reference paper portfolio, target 10% of current equity for each new
position, bounded by remaining cash and a 30% gross-exposure allocation limit.
Calculate the actual allocation at the reference opening fill using then-observable
equity and existing positions. The limit governs new allocations; existing
marked-to-market exposure may drift above it, in which case no new position is added.
Do not invent an untested rebalancing exit to enforce a continuous cap. Do not lever,
short or pyramid. A position remains open through its D2 close; new next-open entries
use the state after that close. The remainder earns zero in the reference comparison,
with identical treatment for baselines. These are evaluation allocations, not a
recommendation for the user's personal account.

Trade-level returns and portfolio returns are different outputs. Mark open positions
daily, include overlapping holds, and report cash exposure. Do not annualize an
average trade return as though all capital could compound independently on every
overlapping signal.

Deduplication key: `(strategy_version, issuer_id, signal_session, holding_period)`.
One alert per key; retries must not create duplicates. A nightly alert expires at the
next session's opening execution window. A late user action is a distinct observed
fill, not evidence for the reference opening-fill strategy.

A useful alert contains: symbol, data timestamp, entry/exit convention, estimated net
return, calibrated probability explicitly called a model estimate, forecast range with
observed coverage context, risk/event flags, and the number of out-of-sample
comparable observations and dates. Feature summaries are descriptive, not causal
explanations. Never fill a sample alert with invented current prices or performance.

Before statistical validation, all output is labeled `RESEARCH / UNVALIDATED`. After
validation, the label can be `QUALIFIED SIGNAL`; it must never become
`GUARANTEED RISE`.

## 11. Event information without retrospective leakage

Version-1 historical signals use the price/volume/calendar contract above. Forward
collection adds event records with `published_at`, `retrieved_at`, `first_seen_at`,
`source` and `revision`. SEC offers public submissions/filings APIs [S9]. Federal
Reserve/BLS schedules can annotate known macro dates [S10]. FRED's historical-vintage
controls matter if macro levels are later introduced [S12].

An earnings calendar's current contents cannot be retroactively treated as the
schedule known on an old date. A historical actual earnings date is also not proof of
advance knowledge. `UNKNOWN` event status must remain distinct from confirmed absence.

Initially, earnings/FOMC/CPI information is an annotation in the research alert, not
an untested signal filter. A later veto on known events creates a separately
versioned policy tested prospectively from the first archived calendar snapshot.
Report both core and event-overlay performance. Do not present a price-only historical
backtest as validation of the event-filtered live strategy.

News processing is excluded initially: free historical news coverage and reliable
as-of timestamps are uneven, and language-model summaries do not supply missing market
data. Unscheduled news and overnight gaps remain material unmodeled risks.

## 12. Limited, predeclared experiments

Exactly five initial learned variants, all sharing dates, universe mask, features
where applicable, labels, costs, decision-model settings and alert policy:

| ID | Variant | Question |
| --- | --- | --- |
| B0 | Decision model with 14 non-Chronos features and the breadth-missingness flag | Is market/volume information enough? |
| C128 | Full Chronos channels, 128-bar context | Does short history help? |
| C256 | Full Chronos channels, 256-bar context | Default hypothesis |
| C512 | Full Chronos channels, 512-bar context | Is extra historical context useful? |
| U256 | Chronos target and calendar only, 256-bar context | Do market/volume covariates add value? |

Also report a fixed simple relative-momentum benchmark: rank positive trailing
five-session stock-minus-sector returns; take up to two subject to the same
eligibility/capacity rules; identical hold and costs. Report an exposure-matched SPY
benchmark and zero-return cash. These controls are not candidates from which to
retroactively choose the easiest benchmark.

For predictive diagnostics, compare terminal log-price forecasts with persistence and
a trailing-volatility reference. For trading, assess B0 and momentum as well as
profitability itself. A model may improve price error without improving trading
selection.

Do not grid-search horizons, stops, thresholds, dozens of indicators, sector
definitions and context simultaneously. Record every variant, including failures.
Selection after many trials inflates apparent performance; statistical correction and
a fresh test are both needed [S13].

Development selection *(amended in `chronos2_hourly_v2`, 3 October 2026)*: whether a
Chronos candidate adds information beyond B0 is answered by the primary information
test of section 14, not by the three-position reference portfolio, which cannot answer
it at these sample sizes. A Chronos candidate
is eligible with at least 25 development-validation dates carrying at least three rows
that every compared system scored, a positive mean daily rank-IC gain over B0, and a
gain that survives removing its single most favourable date. Rank eligible candidates
by the 90% paired date-block-bootstrap lower bound of that gain. If the leader's gain
over the runner-up is inconclusive at 90%, retain C256 for continued research, without
declaring it superior. Claim development evidence of added information only when the
selected candidate's lower bound against B0 is above zero and the frozen checkpoint
produced its forecasts; otherwise do not claim a Transformer edge. The trading floors —
at least 40 executed development-validation trades across 25 distinct signal sessions,
positive base/stress net returns, and no dependence on one lucky outlier — are still
computed and reported for every candidate; they no longer select.

LightGBM and fine-tuning are outside these five variants. Their later admission
requires a new registered version, additional untouched evaluation data and an
explicit capacity justification. Do not silently replace a weak Chronos result with an
unrelated successful strategy under the same project claim.

## 13. Chronology, pretraining and validation

Pin checkpoint revision `95a9710e2596287d08352589f42634fa5abdf0a7`. The inspected
repository commit dates its weights/config to 30 October 2025 [S6]. Use the next
trading session, 31 October 2025, as the conservative earliest forecast origin for the
main post-checkpoint study, rather than relying on an earlier announcement date.
Verify weight checksum when eventually downloading. Original stored safetensors
SHA256: `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42` [S6].

Earlier hourly/daily data can provide historical context and separately labeled
diagnostics, but are not clean proof of historical deployability of this model. A
post-checkpoint evaluation reduces pretraining contamination; it does not fix
current-watchlist bias, vendor revisions or researcher's prior exposure to historical
outcomes.

Only include an origin in outcome analysis once its D2 exit is observed. For a latest
market-data date of 25 September, the final two session origins do not yet have
two-session outcomes. Use calendar arithmetic, not calendar-day subtraction.

Reserve the last 60 fully labeled post-checkpoint origin sessions as the final
historical test. No candidate selection or threshold changes may read their results.
Earlier post-checkpoint origins form development.

Development folds use expanding chronological blocks:

- At least 80 distinct sessions for fitting the decision heads.
- A two-session purge/gap, also checking actual entry/exit timestamps.
- The next 30 distinct sessions for probability calibration.
- Another two-session purge/gap.
- The next 20 sessions for validation, then advance by 20.

All symbols of a date belong to the same split. Drop any training/calibration label
whose outcome would not have been known at the next stage's fit time; timestamp
checking overrides fixed gap lengths. Historical feature windows may share older
public prices across a boundary; overlapping outcome windows may not leak future
labels across it.

This clean history supports only a small number of such folds, not an
impressive-looking five-year Chronos-2 validation. If development has too few
sessions, do not shorten blocks after seeing performance: stay in research and collect
more dates.

Chronos is frozen, but its forecasts must still be generated with data ending at each
origin. Decision-head training uses these causal historical forecasts. Generate
forecasts for all eligible rows, including weak signals; selectively keeping only
successful/alerted outcomes biases fitting and calibration.

Once a candidate is frozen, evaluate the final 60-session test once. The
deployment/update policy is prequential: every 20 sessions, automatically refit from
matured preceding history using the same fit/purge/calibration recipe, without
changing features, thresholds or hyperparameters. Refit schedules and all training
timestamps are replayed identically in the final test. Earlier test outcomes may enter
a later scheduled fit only after they would have been observed; no manual test-driven
redesign is allowed.

After the historical test, start a timestamped forward paper ledger. A failed test
cannot be relabeled development and reused as a new clean test. A new research version
needs genuinely new evaluation dates. Historical results already investigated in other
projects are not magically unknown because this pipeline has a new name.

## 14. Evidence required before calling an edge credible

Metrics must include alert coverage, trade count, distinct signal dates, win rate,
mean/median net return, profit factor, payoff ratio, realized downside, portfolio
drawdown, exposure, turnover, daily net portfolio return, and excess return versus the
fixed controls. Report failures and no-alert dates.

Predictive metrics: pinball loss, terminal median error, empirical p10–p90 coverage,
cross-sectional rank association, Brier score and calibration reliability. Calculate
intervals on date blocks, not by pretending all stock rows are independent.

Use 2,000 moving-block bootstrap samples of complete daily portfolio records, initial
block length 10 sessions, with 5/20-session sensitivity. Resample the same date blocks
across stocks and competing systems, preserving contemporaneous dependence. Report
statistical uncertainty and the short-sample limitation; a bootstrap does not create
missing regimes.

Primary information test *(added in `chronos2_hourly_v2`, 3 October 2026)*. Whether
Chronos-2 adds information beyond B0 is decided by the daily cross-sectional rank IC:
on each date, the Spearman correlation between the registered ranking score (estimated
net return over `sigma_2d`) and the realised base-cost net return, computed only on the
rows that every compared system scored — the candidate and B0 in the final test, B0 and
every candidate in development. A date with fewer than three such rows is dropped for
all of them, so the series share one date index. The statistic is the candidate's mean
daily IC minus B0's; its interval comes from the same moving-block bootstrap — 2,000
samples, block length 10 sessions — resampling identical dates for every system.
Development uses 90% (section 12); the final historical test uses 95%, on the
reserved origins, once. Information is demonstrated when the lower bound is above zero
and the frozen checkpoint produced the candidate's forecasts. This is a claim about
ranking information, not about trading: it leaves the alert policy, the portfolio and
all eight promotion requirements below unchanged, and it is not Transformer-specific
alpha, which still needs requirement 4's incremental portfolio evidence. Information
demonstrated with failed gates is reported as exactly that.

Predeclared promotion requirements:

1. Final historical base-cost and stress-cost results are positive under the reference
   execution policy.
2. Forward paper evaluation lasts at least 60 sessions and includes at least 50
   executed signals; otherwise continue collecting.
3. Combined untouched historical-test and forward evidence contains at least 100
   executed trades across at least 60 distinct signal sessions. These are minimum
   bookkeeping floors, not guaranteed adequate statistical power.
4. The 95% date-block-bootstrap lower bound for mean daily net portfolio return is
   above zero; report incremental performance versus B0, momentum and exposure-matched
   SPY. If incremental evidence is inconclusive, say so and do not claim
   Transformer-specific alpha.
5. At least three separate 20-session evaluation blocks are profitable, and removing
   the single best trade does not eliminate all net profitability.
6. Report results with the strongest contributor removed; if one issuer/sector explains
   almost everything, restrict the claim rather than generalizing to 50 stocks.
7. Calibrated probabilities beat the training-derived constant base rate on Brier score
   out of sample, and reliability/coverage diagnostics are acceptable. Do not attach a
   success probability to the phrase "strong" without this evidence.
8. Core improvements survive model removal/feature ablation, costs and the actual
   capital constraints. Any failed gate means the research remains unvalidated.

These gates are conservative research acceptance rules, not a theorem that future
returns will be positive. Slow signal arrival may require substantially more than
three calendar months. A 55% win rate can still lose money if the losses are larger
than the gains.

## 15. Runtime structure, persistence and monitoring

Use a local Python application with these isolated responsibilities:

| Component | Responsibility |
| --- | --- |
| Collector | Fetch bounded increments; retain immutable retrieval snapshots |
| Calendar/quality validator | Establish completed bars, corporate-action consistency and eligibility |
| Feature builder | Produce origin-bounded arrays and daily features |
| Forecaster | Load frozen weights once; batch inference; validate outputs |
| Decision service | Apply the registered preprocessing, fitted heads and calibration |
| Policy engine | Apply eligibility, thresholds, capacity and deduplication |
| Ledger/evaluator | Save forecasts before outcomes, process fills/actions and score matured outcomes |
| Notifier | Render the same persisted signal to a configured personal channel |

Parquet stores market snapshots and forecasts; SQLite stores manifests, run state,
signal IDs, positions and outcomes. Back up artifacts locally. No external database
subscription is necessary. Dataset manifests record hashes; cache keys include
checkpoint revision, package/source version, schema version, symbol, origin, context,
horizon and source snapshot hash.

Minimum tables/files: `instruments`, `source_snapshots`, `bars`, `corporate_actions`,
`event_snapshots`, `origin_features`, `forecasts`, `fit_manifests`, `signals`,
`paper_positions`, `fills`, `outcomes`, `run_log`. Each price/event record has
observation/event time and retrieval/availability time. Forecast tables are
append-only and distinct from subsequently attached outcome tables.

Holdout outcome access is disabled in development reports. The experiment manifest
records all evaluations, seeds, rejected configurations and known historical exposure.
A release manifest binds data, code, model weights, policy and costs together. Do not
silently revise old forecast values when Yahoo corrects history.

Operations:

- After close: fetch, validate, forecast, score, commit records, then notify. Commit
  must precede notification so retries are idempotent.
- Each next session: record the reference opening execution; expire unexecuted manual
  alerts separately.
- After each session: mark positions and finalize any D2 exits. Query only matured
  labels during scheduled refits.
- Every 20 sessions: perform the preregistered decision-head refit. The Transformer
  remains frozen.
- Archive overlapping 15m data daily, verifying duplicates/revisions without
  overwriting the original as-of archive.

Monitor data lag, missing bars, action anomalies, failed tickers, inference latency,
nonfinite/crossed quantiles, feature clipping, signal counts, probability calibration,
realized costs and drawdown. Crossing quantiles block that forecast in version 1; do
not sort them silently and pretend the model issued the repaired output.

Daily operational checks do not justify daily strategy retuning. Investigate drift at
scheduled reviews. Suspend new qualified alerts if the reference portfolio drawdown
reaches 5% from its peak, a fitted artifact is invalid, or the latest origin fails
mandatory data checks. A pause does not close an existing position at an imaginary
protected price: existing reference positions retain their registered time exits. A
drawdown pause is absorbing within a registered evaluation run and is not reset at
fold/report boundaries. Re-enabling after that pause requires a documented new policy
version and fresh evaluation. A transient data-quality failure can recover at a later
origin when the original checks pass; missed alerts are not backdated.

Statistical performance gates are assessed at predeclared evaluation checkpoints, not
repeatedly until a confidence interval happens to turn positive. Additional looks and
strategy revisions must be recorded as new research decisions. There is no automatic
loop searching for a passing result.

No broker connection or automatic orders in this scope. A personal Telegram/email
adapter can be implemented later; no channel is contacted or configured by this
document. Secrets would belong in environment variables, never source files or
reports.

## 16. Meaningful verification before any large run

Implementation should prove these invariants on small controlled fixtures before
fetching a large universe or running a backtest:

1. Appending/changing future prices cannot change any feature or forecast input at an
   earlier origin.
2. Changing a future outcome cannot change an earlier fit, calibration or alert.
3. A normal session, an early close, a DST transition, a weekend and a holiday produce
   correct bar/exit timestamps.
4. The incomplete current bar is excluded; synthetic overnight flat bars are never
   introduced.
5. Split/dividend fixtures preserve economic wealth and do not create false momentum,
   incorrect liquidity or double-counted cash.
6. A positive D0-close-to-D1-open gap contributes no captured return to the reference
   strategy.
7. Every stock of an origin shares its temporal split; no overlapping unknown label
   crosses a training boundary.
8. With cross-learning disabled, reordering independent tasks or changing the memory
   batch budget does not materially change predictions beyond numerical tolerance.
9. Quantile axes, levels, terminal indices and log-price inverse transforms are
   correct; the API point forecast is treated as a median.
10. Duplicate runs produce one persisted signal/notification, and failed data cannot
    generate a fresh-looking signal.
11. Portfolio cash, simultaneous holds, correlation limits and corporate-action
    accounting reconcile exactly on a small hand-checkable ledger.
12. Baseline and candidate tests use identical dates, fills, costs and capital
    accounting.

These tests address financial/model-integrity risks rather than mirroring trivial
code. Only after they pass should a small non-performance smoke run establish package
compatibility, memory use and measured runtime. Such a smoke run must not be used to
tune a trading rule on the final test.

See [INVARIANTS.md](INVARIANTS.md) for the mapping from each numbered invariant to the
test that proves it.

## 17. What could fail and the planned response

| Weakness | Observable symptom | Response |
| --- | --- | --- |
| No usable information in the forecast | B0/momentum equal or better after costs | Reject Transformer-specific edge; do not hide the result |
| Price-level persistence looks accurate | Low price error, little return/rank skill | Evaluate actual holding returns and persistence baseline |
| Context too stale or too short | 128/256/512 disagree and performance unstable | Use the registered comparison; no discretionary daily context switching |
| Apparent confidence is miscalibrated | Reliability/coverage poor out of sample | Suppress numerical success claims; recalibrate only under registered updates |
| Free feed misses execution detail | Results vanish at higher assumed costs | Require stronger margin or reject; no invented tight spreads |
| Gaps consume the forecast move | Close-to-close looks good; next-open labels fail | Reject that tradable signal |
| Short history / one market regime | Wide block intervals; insufficient signal dates | Collect prospective data; do not treat stock count as independent evidence |
| Market beta explains profit | Exposure-matched SPY/B0 matches results | Describe market exposure, not forecasting alpha |
| Watchlist or delisting bias | Strong survivors drive retrospective gains | Restrict claim; preserve forward failures and missing exits |
| Contextual inputs add noise | U256 outperforms multivariate candidates | Prefer simpler validated variant; do not force every input |
| Model/update selection leaks | Performance changes when folds are repaired | Invalidate affected results and rerun only with a declared protocol |
| Unscheduled corporate news | Gap or drawdown outside model range | Acknowledge residual risk; no quantile-based guarantee |

All favorable mechanisms here are plausible starting hypotheses. Research on
financial-return forecasting also warns that gains against simple benchmarks can be
small and asset-dependent [S14]; general forecasting success cannot be transferred into
a guaranteed trading claim.

## 18. Later 15m and nonlinear experiments

Store 15m observations now when implementation begins. Do not infer a long-history win
from the present 60-day window. After at least 180 archived sessions, register a
separate resolution experiment with a new future evaluation block. For approximately
20-session history, compare around 140 hourly bars with 520 quarter-hourly bars; use
identical origins and labels, and let the model handle patch padding. This isolates
elapsed history more fairly than equal bar counts.

One possible subsequent experiment adds 15m intraday shape as a separate predictor to
the already frozen hourly model. It must be compared on dates where both feeds exist,
after costs, without selecting a different holding period to make it win. An ensemble
is justified only by incremental out-of-sample performance; model agreement is not
independent evidence when both consume almost the same prices.

Admit LightGBM only in a separately registered experiment with substantially more
independent development dates, preferably at least 250, and a fresh test. Initial
constrained candidate: binary objective for profitability; 200 estimators, learning
rate 0.03, `num_leaves` 7, `max_depth` 3, `min_child_samples` 200, `reg_lambda` 10,
`colsample_bytree` 0.8, no class weights, fixed seed 42, sigmoid calibration on
held-out dates. Regressor comparison is a separate recorded trial. Verify
weight-normalization effects on leaf constraints. These parameters are capacity
controls, not a claimed optimum [S15].

Fine-tuning 120M weights on this small watchlist is not the first solution to weak
signals. Reconsider only if frozen forecasts show transferable value and additional
diverse training data plus untouched evaluation justify adaptation. LoRA would reduce
trainable parameters, not eliminate overfitting or data leakage.

## 19. Declarative configuration to implement

This is a design configuration, not an executable implementation. Values remain
unchanged until a registered protocol amendment.

The machine-readable form lives in [`config/design.yaml`](../config/design.yaml) and is
loaded and validated by `chronos2_signal.config.load_design`. The loader rejects an
unknown key, a missing key, or a value outside its declared domain, so a silent drift
away from this document is not possible.

## 20. Build order and unresolved facts

1. Freeze this protocol, obtain the independent candidate roster, and create the
   current-universe manifest without consulting strategy returns.
2. Audit small Yahoo samples for history depth, timestamp anchoring and split/volume
   behavior. Freeze dependencies and provenance.
3. Implement calendar, action/wealth accounting and leakage invariants; pass controlled
   fixtures.
4. Build persistence/momentum/B0 baselines and frozen causal forecast cache, keeping
   final-test outcomes hidden.
5. Run only the registered development comparisons. Freeze the selected version or
   record no winner.
6. Execute the final historical protocol once, including the predefined refit schedule
   and costs.
7. Start forward paper recording, then assess at registered checkpoints. Qualify alerts
   only if evidence supports them.

Facts that cannot honestly be resolved before those steps: actual usable bars per
symbol, provider revisions, the current 50-issuer eligibility list, installed package
compatibility, measured runtime/memory, actual broker costs, the winning context,
calibration quality, and whether an edge exists. These are explicit measurement tasks,
not hidden assumptions. No paid-data workaround is presupposed.

See [BUILD_ORDER.md](BUILD_ORDER.md) for which steps the current repository has
completed and which remain.

## Primary references inspected

- **S1** — Model card: `amazon/chronos-2`.
- **S2** — Architecture and training paper: *Chronos-2: From Univariate to Universal
  Forecasting*.
- **S3** — Actual checkpoint configuration: `config.json`.
- **S4** — yfinance history implementation: `history.py`; compare with the
  less-specific download reference.
- **S5** — Chronos inference implementation: `pipeline.py`.
- **S6** — Frozen checkpoint provenance and hash: weights/config commit.
- **S7** — Session calendars: NYSE hours and calendars; pandas-market-calendars.
- **S8** — Yahoo adjustment definition: *What is the adjusted close?*.
- **S9** — Public filing data: SEC EDGAR APIs.
- **S10** — Official scheduled events: Federal Reserve FOMC calendar; BLS schedule.
- **S11** — Calibration: scikit-learn probability calibration; Ridge reference.
- **S12** — Historical macro vintages: FRED real-time periods.
- **S13** — Selection bias: Bailey and Lopez de Prado, *The Deflated Sharpe Ratio*.
- **S14** — Finance-specific evidence, limited scope: *Pretrained Time-Series
  Foundation Models for Financial Return Forecasting*. Its five-equity study is not a
  validation of this proposed watchlist or strategy.
- **S15** — Later nonlinear candidate parameters: LightGBM parameter documentation.

Primary-source facts above are distinct from the author's proposed defaults. No source
is cited as proof that this system will be profitable.
