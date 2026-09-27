# Status

**As of 27 September 2026. Design version `chronos2_hourly_v1`, status
`design_only_unvalidated`.**

## What exists

The protocol in [DESIGN.md](DESIGN.md) is implemented as a working, tested research
pipeline. Build-order steps 3 and (in its dry form) 2 are done; steps 1, 4, 5, 6 and 7
need data the project does not yet have.

- All twelve section-16 integrity invariants pass on controlled fixtures
  ([INVARIANTS.md](INVARIANTS.md)).
- 57 tests pass offline: no network, no checkpoint, no market data.
- The registered configuration loads under a strict validator that rejects drift.
- The CLI runs `verify`, `schedule`, `audit --roster ...` (dry), `smoke` and
  `demo-study`.

## What has *not* happened

This is the important half of the status.

- **No market data has been downloaded.** No Yahoo request has been made.
- **No model weights have been loaded.** `chronos-forecasting` and `torch` are not
  installed; `chronos2-signal smoke` reports `unavailable` and refuses to substitute
  anything.
- **No forecast has been produced by Chronos-2.** The offline tests use
  `DeterministicStubForecaster`, an arithmetic fixture that declares itself not a model
  and carries no predictive content.
- **No backtest of any real strategy has been run.** The `demo-study` command scores a
  seeded random walk and says so in its own report.
- **No watchlist has been frozen.** `config/candidate_roster.example.csv` is a template
  with placeholder rows.
- **No notification channel is configured.** Nothing contacts Telegram, email or a
  broker.
- **Runtime and memory remain unmeasured** for the real checkpoint. The only measured
  figure is the synthetic pipeline: roughly 0.2–0.9 s per origin for 8 symbols on CPU,
  which says nothing about 120M-parameter inference over 50 symbols.

**No edge is claimed, and none has been tested.**

## The chronology the protocol currently supports

From the registered earliest primary origin (31 October 2025, the session after the
frozen checkpoint's weight date) to a data edge of 25 September 2026:

| Quantity | Value |
| --- | --- |
| Labelled post-checkpoint origins (D2 exit observed) | 224 |
| Reserved for the single final historical test | 60 |
| Available for development | 164 |
| Development folds at the registered block sizes | **2** |
| Prequential refit points across the test block | 3 |

Two folds. That is the honest size of the clean history: enough to run the registered
development comparison, not enough for a wide-ranging search. Per section 13, the
response to too little history is to collect more dates, never to shorten a block
after seeing performance — and the code enforces this by raising
`InsufficientHistory` rather than shrinking.

Reproduce with:

```
chronos2-signal schedule --latest-session 2026-09-25
```

For reference, `build_schedule` raised `InsufficientHistory` for a data edge of
19 December 2025 (33 labelled origins, fewer than the 60 the final test reserves
alone). That is the same guard, working.

## Next steps, in protocol order

1. **Step 1 — freeze the universe.** Obtain a candidate roster chosen independently of
   this experiment's returns, then run the audit and
   `universe.select_watchlist`. The freeze must happen before any performance metric
   exists; `select_watchlist` refuses once outcomes are in the ledger.
2. **Step 2 — live data audit.** `chronos2-signal audit --roster <file> --live` after
   installing the `provider` extra. Then pin the exact versions into
   `requirements.txt` per [ENVIRONMENT.md](ENVIRONMENT.md).
3. **Compatibility smoke test.** Install the `model` extra and run
   `chronos2-signal smoke`. It records the introspected `predict_quantiles` signature
   rather than assuming it, verifies shapes, quantile ordering, terminal indices and
   dates, and measures single-task runtime. If the package cannot load the pinned
   revision, stop and resolve compatibility.
4. **Steps 4–6 — the registered comparison, then the single final test.**
5. **Step 7 — forward paper ledger**, assessed only at the predeclared checkpoints.

## Known limitations that no amount of implementation fixes

- Today's surviving watchlist carries survivorship and selection bias. Historical
  results on it are conditional watchlist evidence, not whole-market evidence.
- A post-checkpoint window reduces pretraining contamination. It does not remove
  vendor revisions, current-watchlist bias, or the researcher's own prior exposure to
  these years of market history.
- Two development folds and one broad market regime. A block bootstrap reports
  sampling variability within the observed history; it cannot invent the regimes the
  sample never contained.
- Unscheduled news and overnight gaps stay material unmodelled risks. The p10 quantile
  is not a loss bound.
