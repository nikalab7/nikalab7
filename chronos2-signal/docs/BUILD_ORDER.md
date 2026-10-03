# Build order

Section 20 of the [registered protocol](DESIGN.md). Each step lists what the
repository provides and what still requires data, weights or elapsed time.

> **Correction, 30 September 2026.** The first version of this file marked steps 4–6
> *implemented, not run*. That overstated it: the components existed, but the
> orchestration that would run them did not — no ledger-backed data source reached the
> pipeline, forecasts were never written to the ledger, B0 was absent from the study
> runner, and nothing replayed the refit schedule. An architecture review found this;
> the orchestration now exists and the labels below describe it.

## 1. Freeze the protocol and the universe — *partly done*

Freeze the protocol, obtain the independent candidate roster, and create the
current-universe manifest without consulting strategy returns.

- **Done:** the protocol is registered in `docs/DESIGN.md` and
  `config/design.yaml`, loaded through a strict validator
  (`chronos2_signal.config`). `chronos2_signal.universe` implements the roster,
  the selection rule and the freeze guard: `select_watchlist` raises
  `FreezeError` if outcomes already exist in the ledger.
- **Outstanding:** a real candidate roster. `config/candidate_roster.example.csv`
  holds placeholder rows. The roster must be chosen independently of this
  experiment's returns — for example a published constituent list captured at a
  recorded date — and its hash goes into every release manifest.

## 2. Audit the provider — *dry run done, live run outstanding*

Audit small Yahoo samples for history depth, timestamp anchoring and split/volume
behaviour. Freeze dependencies and provenance.

- **Done:** `chronos2-signal audit --roster <file>` prints the exact request plan
  and the checks it would run, without contacting anything.
  `chronos2_signal.collector` implements bounded requests, retry with backoff and
  jitter, a hard stop on rate limiting, content-addressed immutable snapshots, and
  bar-anchoring validation against the exchange schedule.
- **Outstanding:** `--live` with the `provider` extra installed, then pinning the
  measured versions per `docs/ENVIRONMENT.md`. Facts that only this step can
  settle: actual usable bars per symbol, provider revisions, real split and volume
  conventions, and the current eligible issuer list.

## 3. Calendar, wealth accounting and leakage invariants — *done*

Implement calendar, action/wealth accounting and leakage invariants; pass
controlled fixtures.

- **Done:** `calendar_spec`, `actions`, `market`, `features`, `quality`,
  `decision`, `policy`, `portfolio`, `protocol`, `storage`. All twelve invariants
  of section 16 pass — see `docs/INVARIANTS.md`.
- **Note:** only after these pass may a non-performance smoke run establish
  package compatibility, memory use and runtime. That is
  `chronos2-signal smoke`, and it must not be used to tune a trading rule.

## 4. Baselines and the causal forecast cache — *implemented, not run*

Build persistence/momentum/B0 baselines and the frozen causal forecast cache,
keeping final-test outcomes hidden.

- **Done:** B0, the fixed relative-momentum control, the exposure-matched market
  benchmark and zero-return cash all run through one session engine
  (`simulation.simulate`) with identical fills, costs and capital accounting. The
  collector's ledger reaches the pipeline through `sources.LedgerMarketSource`, at a
  point-in-time or a declared latest vintage. Every forecast is written to the
  append-only `forecasts` table when its batch is built, before any label for that
  origin exists (`forecast_store.ForecastCache`), under a key covering the checkpoint
  revision, package version, input schema, symbol, origin, context, horizon, channel
  set, cross-learning flag and source snapshot hash. `holdout.HoldoutGuard` is checked
  on every research path, not only on ledger reads.
- **Outstanding:** running it, which needs step 2's data and the `model` extra.

## 5. Registered development comparison — *implemented, not run*

Run only the registered development comparisons. Freeze the selected version or
record no winner.

- **Done:** `operations.run_development_comparison` runs every registered variant —
  B0 always included — across the development folds, one model per fold and one
  portfolio per system carried across fold boundaries, with the fixed controls on the
  same dates. The paired bootstrap resamples identical date blocks for every system
  and reports each candidate against B0, against momentum, against its own
  exposure-matched benchmark and against every other candidate.
  `operations.select_candidate` applies the section-12 rule as amended in
  `chronos2_hourly_v2`: candidates are ranked by the paired rank-IC gain over B0 from
  `evaluation.information_test`, on the rows every system scored, and the rule records
  *no winner* — retaining C256 without declaring it superior — whenever the evidence
  does not separate the leader. The trading floors of 40 trades across 25 sessions are
  reported beside the selection and no longer decide it. The
  out-of-sample predictive report (Brier against each model's own training base rate,
  reliability, p10–p90 coverage, terminal pinball against a trailing-volatility
  reference, median error against persistence, and cross-sectional rank IC) now feeds
  gate 7, which previously received hard-coded `NaN`.
- **Outstanding:** the actual comparison, on real data.

## 6. The single final historical test — *implemented, not run*

Execute the final historical protocol once, including the predefined refit
schedule and costs.

- **Done:** `protocol.build_schedule` reserves the last 60 labelled origins, purges
  the two origins before them so that no development outcome window reaches the test
  block, and builds the 20-session prequential refit points from the schedule alone.
  `operations.run_final_test` replays exactly those refit points — each one fitted
  from history that had matured by its deadline — under a deliberately unlocked
  guard, for the candidate and B0 alike. Its report leads with the primary
  information test at 95%, beside the eight gates rather than inside them.
- **Outstanding:** the single pass. A failed test cannot be relabelled development
  and reused. Nothing yet stops a second pass *across processes*: the guard's
  unlock is one-way within a process only, and "once" is still a discipline rather
  than a mechanism.

## 7. Forward paper ledger — *not started*

Start forward paper recording, then assess at registered checkpoints. Qualify
alerts only if evidence supports them.

- **Done:** `operations.run_after_close` is the operational flow (validate,
  forecast, score, commit, then notify), and the ledger schema records fills,
  positions, outcomes and daily marks.
- **Outstanding:** at least 60 forward sessions with at least 50 executed signals.
  This is elapsed time, not engineering. Promotion needs every gate in section 14
  to pass; `evaluation.evaluate_promotion_gates` fails closed on absent evidence.

## Facts that cannot honestly be resolved before those steps

Actual usable bars per symbol; provider revisions; the current 50-issuer
eligibility list; installed package compatibility; measured runtime and memory;
actual broker costs; the winning context; calibration quality; and whether an edge
exists. These are explicit measurement tasks, not hidden assumptions.
