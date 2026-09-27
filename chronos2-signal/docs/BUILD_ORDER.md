# Build order

Section 20 of the [registered protocol](DESIGN.md). Each step lists what the
repository provides and what still requires data, weights or elapsed time.

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
  benchmark and zero-return cash all run through the same portfolio, fills and
  costs (`variants`, `pipeline`, `portfolio`). The forecast cache key covers the
  checkpoint revision, package version, schema version, symbol, origin, context,
  horizon, channel set, cross-learning flag and source snapshot hash
  (`provenance.forecast_cache_key`), and `forecasts` is append-only at the
  database level. `holdout.HoldoutGuard` makes the reserved origins unreadable in
  development mode.
- **Outstanding:** running it, which needs step 2's data and the `model` extra.

## 5. Registered development comparison — *implemented, not run*

Run only the registered development comparisons. Freeze the selected version or
record no winner.

- **Done:** the five variants are the only ones `variant_spec` accepts; anything
  else raises. `operations.run_study` runs one fold against all controls on one
  shared date index with the paired date-block bootstrap.
- **Outstanding:** the actual comparison. Development selection also requires at
  least 40 executed development-validation trades across 25 distinct sessions;
  "no winner" is a permitted and recorded outcome.

## 6. The single final historical test — *implemented, not run*

Execute the final historical protocol once, including the predefined refit
schedule and costs.

- **Done:** `protocol.build_schedule` reserves the last 60 labelled origins and
  builds the 20-session prequential refit points from the schedule alone, so the
  same points are replayed whether the test is being planned or executed.
  `HoldoutGuard.unlock_final_test` is a deliberate, one-way action within a
  process.
- **Outstanding:** the single pass. A failed test cannot be relabelled development
  and reused.

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
