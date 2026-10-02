# Status

**As of 30 September 2026. Design version `chronos2_hourly_v1`, status
`design_only_unvalidated`.**

## What exists

The protocol in [DESIGN.md](DESIGN.md) is implemented as a working, tested research
pipeline, including the orchestration for build-order steps 4–6. Steps 1, 2 (live), 4, 5,
6 and 7 need data, weights or elapsed time the project does not have.

- All twelve section-16 integrity invariants pass on controlled fixtures
  ([INVARIANTS.md](INVARIANTS.md)).
- 105 tests pass offline: no network, no checkpoint, no market data. 32 of them are
  system-level tests that go through the pipeline rather than calling components
  directly, and each was mutation-checked against the defect it guards.
- The collector's ledger reaches the pipeline through a session- and vintage-bounded
  data source, and a test proves it reproduces the fixture's features exactly.
- The development comparison (all five variants, B0 always included), the
  single-candidate study and the prequential final test all run end to end on
  synthetic data.
- The CLI runs `verify`, `schedule`, `audit --roster ...` (dry), `smoke` and
  `demo-study`.

## What has *not* happened

This is the important half of the status.

- **No market data has been downloaded.** Yahoo Finance and Hugging Face are both
  blocked by the egress policy of the environment this was built in (403 on CONNECT);
  per that policy they were reported rather than routed around.
- **No model weights have been loaded.** `chronos-forecasting` and `torch` are not
  installed; `chronos2-signal smoke` reports `unavailable` and refuses to substitute
  anything.
- **No forecast has been produced by Chronos-2.** The offline tests use
  `DeterministicStubForecaster`, an arithmetic fixture that declares itself not a model
  and carries no predictive content.
- **No backtest of any real strategy has been run.** `demo-study` scores a seeded random
  walk and says so in its own report. At the registered thresholds it produces zero
  alerts, which is the correct result on noise.
- **No watchlist has been frozen.** `config/candidate_roster.example.csv` is a template
  with placeholder rows.
- **No notification channel is configured.** Nothing contacts Telegram, email or a
  broker.
- **Runtime and memory remain unmeasured** for the real checkpoint.

**No edge is claimed, and none has been tested.**

## Unverified against the real checkpoint

The weights are unreachable (Hugging Face is blocked), so the checkpoint's numerical
behaviour cannot be checked here. The adapter's mapping to and from the library *was*
checked on 2 October 2026 against the source of `chronos-forecasting` 2.3.2, which
PyPI does serve:

- **Input — matches.** The library documents `past_covariates` as "past-only
  covariates or past values of known future covariates" and requires every
  `future_covariates` key to appear in `past_covariates`: exactly the adapter's
  mapping. Its batch budget counts series, covariates included, as the adapter's
  channel batching assumes, and NaN marks a missing value.
- **Output — matches.** `predict_quantiles` returns `(quantiles, mean)`, each a list
  with one tensor per task of shape `(n_variates, prediction_length, n_quantiles)`,
  and the library states that "the median is returned as the mean". The adapter
  unpacks the pair and transposes that shape correctly.
- **Still unverified:** whether the pretrained weights actually *use* every channel.
  `smoke` perturbs the calendar history, the calendar future and the past-only
  covariates in turn and fails if any group leaves the output unchanged.

## The chronology the protocol currently supports

From the registered earliest primary origin (31 October 2025, the session after the
frozen checkpoint's weight date) to a data edge of 25 September 2026:

| Quantity | Value |
| --- | --- |
| Labelled post-checkpoint origins (D2 exit observed) | 224 |
| Reserved for the single final historical test | 60 |
| Purged between development and the test | 2 |
| Available for development | 162 |
| Development folds at the registered block sizes | **2** |
| Prequential refit points across the test block | 3 |

Two folds. That is the honest size of the clean history: enough to run the registered
development comparison, not enough for a wide-ranging search. Per section 13, the
response to too little history is to collect more dates, never to shorten a block after
seeing performance — and the code enforces this by raising `InsufficientHistory` rather
than shrinking.

The two purged origins are new since 27 September. Without them, the last development
origins' outcome windows reached into the first test origins' holding windows. The
folds were already purged; this boundary was not.

Reproduce with:

```
chronos2-signal schedule --latest-session 2026-09-25
```

## Known gaps in the implementation

- **Unexplained price jumps are not detected.** A split that shows in the prices as a
  raw step but is missing from the provider's action feed has no audit, so the step
  passes through as a genuine move. (Only a held position whose entry quote a later
  view restates is caught, by the unit check, and left unpriced.) Section 5's "no
  unresolved corporate-action discrepancy" needs a jump detector that flags large moves
  with no corresponding action.
- **"Evaluate the final test once" is enforced within a process only.** The guard's
  unlock is one-way inside a process, but nothing records a completed final test in the
  ledger and refuses a second one later.
- **A missing exit price writes the position down to zero.** The trade stays in the
  ledger as unresolved, as the protocol requires, but the portfolio carries no value for
  it. That is conservative for any performance claim; it is not an accurate mark. The
  same applies to a position whose units a later view cannot reconcile. A 10%
  position written off is a 10% drawdown, which trips the absorbing drawdown pause, so
  one data failure can stop a variant trading for the rest of a run.
- **Two readings of the text are open.** Feature f15 measures the drawdown from a
  20-session high that includes the origin day, while f11's "preceding" excludes it;
  the design text allows either. And section 14's moving-block bootstrap, as
  registered, gives the first and last sessions about a tenth of the weight of the
  middle ones; a circular block bootstrap would not.
- **A view from before a split can disagree with its own action ledger.** A view
  restates only the splits inside its data. If the provider left a *later* split
  unrestated, a view from before it quotes pre-split prices while the ledger it returns
  already records the split. The conversions that read the ledger — as-of price floors,
  nominal share counts, and a dividend that falls before the split — then assume the
  wrong units. Returns and features are unaffected. Data whose splits the provider
  restated, read at the latest vintage, never reaches this state; the fixture's raw-split
  markets and a raw-price provider can.
- **No `predict` command.** Scoring a live origin exists as `operations.run_after_close`
  but is not exposed on the CLI, and fitted decision models are not yet serialised.

## Next steps, in protocol order

1. **Step 1 — freeze the universe.** Obtain a candidate roster chosen independently of
   this experiment's returns, then run the audit and `universe.select_watchlist`. The
   freeze must happen before any performance metric exists; `select_watchlist` refuses
   once outcomes are in the ledger.
2. **Step 2 — live data audit.** `chronos2-signal audit --roster <file> --live` after
   installing the `provider` extra, from a network that can reach Yahoo. Then pin the
   exact versions into `requirements.txt` per [ENVIRONMENT.md](ENVIRONMENT.md).
3. **Compatibility smoke test.** Install the `model` extra and run
   `chronos2-signal smoke`. It records the introspected `predict_quantiles` signature,
   verifies shapes, quantile ordering, terminal indices and dates, checks that every
   channel group reaches the model, and measures single-task runtime. If the package
   cannot load the pinned revision, stop and resolve compatibility.
4. **Steps 4–6 — the registered comparison, then the single final test.**
5. **Step 7 — forward paper ledger**, read at the point-in-time vintage and assessed only
   at the predeclared checkpoints.

## Known limitations that no amount of implementation fixes

- Today's surviving watchlist carries survivorship and selection bias. Historical
  results on it are conditional watchlist evidence, not whole-market evidence.
- A post-checkpoint window reduces pretraining contamination. It does not remove vendor
  revisions, current-watchlist bias, or the researcher's own prior exposure to these
  years of market history. A historical backfill read at the latest vintage cannot
  exclude revisions either, and every report produced that way says so.
- Two development folds and one broad market regime. A block bootstrap reports sampling
  variability within the observed history; it cannot invent the regimes the sample
  never contained.
- Unscheduled news and overnight gaps stay material unmodelled risks. The p10 quantile is
  not a loss bound.
