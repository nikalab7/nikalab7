# Protocol amendments

Every change to a value in `config/design.yaml` or to a rule in
[DESIGN.md](DESIGN.md) is recorded here, with its date, its reason, and the new
`design_version`. Values remain unchanged until such an amendment exists.

An amendment that changes features, thresholds, hyperparameters or the execution
policy invalidates results produced under the previous version for promotion
purposes. It does not retroactively relabel a failed test as development: a new
research version needs genuinely new evaluation dates.

| Date | design_version | Change | Reason |
| --- | --- | --- | --- |
| 2026-09-27 | `chronos2_hourly_v1` | Initial registration | Protocol frozen before implementation |
| 2026-10-03 | `chronos2_hourly_v2` | The daily rank-IC gain over B0 becomes the primary criterion for "does Chronos-2 add information", in development selection (section 12) and in the final test (section 14). Trading thresholds, the portfolio and all eight gates are unchanged | The registered portfolio test cannot detect a realistic edge at these sample sizes; see below |

## `chronos2_hourly_v2` — 3 October 2026

**Decided before any real outcome was examined.** No market data has been downloaded
and no Chronos-2 forecast produced. The decision rests only on synthetic positive
controls and simulation ([ANALYSIS.md](ANALYSIS.md)); the protocol owner chose this
option among the four that analysis listed.

**What changed.** Section 12's development selection rule read:

> Development selection: require at least 40 executed development-validation trades
> across 25 distinct signal sessions, positive base/stress net returns, and no
> dependence on one lucky outlier. Rank eligible Chronos candidates by the 90%
> date-block-bootstrap lower bound of mean daily reference-portfolio net return. If
> differences are inconclusive, retain C256 for continued research, without declaring
> it superior. A candidate must also show useful incremental evidence over B0; if not,
> do not claim a Transformer edge.

It now ranks candidates by the 90% paired date-block-bootstrap lower bound of the mean
daily rank-IC gain over B0, on the rows every compared system scored. Eligibility is at
least 25 such dates, a positive mean gain, and a gain that survives removing the single
best date. The no-winner rule is kept: if the leader is not separated from the
runner-up, C256 is retained without being declared superior. Section 14 gains the
primary information test that defines the statistic, at 90% in development and 95% in
the final test. A claim also needs the frozen checkpoint to have produced the
forecasts: a test fixture's result is reported, never credited to Chronos-2. The full
text is in [DESIGN.md](DESIGN.md), marked in place.

Three keys are added to `validation` in `config/design.yaml`:
`information_criterion: daily_rank_ic_candidate_minus_b0`,
`information_min_common_rows_per_date: 3` and `development_min_information_dates: 25`.
The loader rejects any other criterion, and rejects a configuration whose
`design_version` is not the one the code implements.

**What did not change.** Every feature, threshold, hyperparameter, cost, the alert
policy, the reference portfolio, the chronology, the bootstrap settings and all eight
promotion gates. The trading floors of section 12 are still computed and reported for
every candidate. A trading claim — the qualified label — still needs every gate.

**Why.** The original rule asked whether Chronos-2 adds information through a
three-position portfolio with a two-session hold. At the registered sample sizes that
test can barely detect even an extraordinary edge: a true +0.40% net per trade clears
the development bound about one time in four and the final test about one in five.
Scoring every eligible stock every date uses the same dates far more efficiently.
Power of the adopted test (simulated, 50 names per date; the gain is the realised mean
daily IC difference):

| Mean daily IC gain over B0 | Development, 40 dates, 90% | Final test, 60 dates, 95% |
| --- | --- | --- |
| none planted (−0.004 realised) | 3% | 2% |
| +0.013 | 15% | 12% |
| +0.026 | 35% | 32% |
| +0.042 | 68% | 70% |
| +0.069 | 95% | 97% |

These numbers assume 50 names; a smaller watchlist has less power. At a gain of exactly
zero the nominal false-positive rates are 5% in development and 2.5% in the final test;
the first row is a candidate slightly worse than B0. `analysis/power_ic_difference.py`
reproduces them.

**What this does not buy.** Information is not profit. A candidate can rank stocks
better than B0 and still lose money after costs, or never reach the alert thresholds;
the report then says exactly that. Section 14's "Transformer-specific alpha" still needs
gate 4's incremental portfolio evidence.

**Effect on earlier results.** None exist: no result was produced under
`chronos2_hourly_v1`.

## Not amendments

For the record, these were implementation decisions inside the protocol's latitude,
not changes to it. They are listed so that a later reader can tell them apart from an
amendment.

- **`sigma_2d_floor: 0.005` and `sigma_daily_window_sessions: 20`** were moved into
  `config/design.yaml` as named keys. Both values are stated in section 9; only their
  location changed.
- **`explicit_fee_fraction: 0.0`** makes the section-8 label's fee term explicit and
  configurable. Section 8 already treats broker commissions as separately accounted.
- **`initial_equity: 100000.0`** names the reference portfolio's starting capital.
  Section 10 specifies allocations as fractions of current equity, so the level itself
  is a reporting convention.
- **Development and promotion floors** from sections 12 and 14 (40 trades / 25
  sessions, the 90% and 95% confidence levels, three profitable 20-session blocks) were
  given config keys instead of being hard-coded.
- **The full 20-session volume lookback is required**, rather than a partial window, for
  the relative-volume channel. Section 7 says "the previous 20 matching session slots";
  a shorter window is masked instead of estimated from fewer observations.
  `features.recommended_panel_bars` loads the extra history so the trimmed context is
  unaffected.
- **Fractional shares** are permitted in the reference paper portfolio, so that position
  sizing is exactly 10% of equity rather than 10% rounded to a whole share. This is an
  evaluation convention; it is not a claim about a real account.
- **Scoring after the close** (30 September 2026). The walk-forward loop now scores an
  origin after that session's exits and daily mark. The earlier order contradicted
  section 10 — "new next-open entries use the state after that close" — so this corrects
  the implementation toward the text rather than changing it.
- **The development selection rule** (2 October 2026). The implementation required the
  leading candidate's own 90% lower bound to be above zero before selecting it. Section
  12 contains no such condition: it ranks eligible candidates by that lower bound and
  retains C256 only "if differences are inconclusive". The extra condition was removed;
  whether a selected candidate's return is credibly positive is the final test's
  question.
- **Refits and block-length sensitivity** (2 October 2026). The prequential refits now
  use the folds' fit/purge/calibration recipe, which section 13 requires ("the same
  fit/purge/calibration recipe"), and the 5/20-session bootstrap sensitivity of section
  14 is now computed. Both bring the implementation to the registered text.
- **No interval from fewer than two blocks** (3 October 2026). Section 14's moving-block
  bootstrap cannot vary on a series that holds a single block: every replicate is the
  series itself, and the interval collapsed onto the point estimate. A zero-width
  interval reads as certainty, so a fixture with no information was "demonstrated" on a
  ten-date test. A series shorter than two full blocks now gets no interval, and every
  rule that reads a bound fails closed. The registered sample sizes are unaffected: 40
  development dates and 60 test origins are at least two blocks at every registered
  block length (5, 10 and 20 sessions).

## Clarification that departs from the letter of the text

Listed separately because, unlike the entries above, it changes which origins belong to
which block. If a reader judges it an amendment, it should be registered as one.

- **A purge between development and the final test** (30 September 2026). Section 13
  says the last 60 labelled origins form the test and "earlier post-checkpoint origins
  form development". Taken literally, the last development origins' two-session outcome
  windows then reach into the first test origins' holding windows, so development
  selection can read part of a reserved outcome. Section 13 also states the principle
  that "overlapping outcome windows may not leak future labels across" a boundary, and
  applies a two-session purge at every fold boundary for exactly this reason. The same
  purge is now applied at the development/test boundary: the two origins immediately
  before the test belong to neither block. On the current data edge this moves
  development from 164 to 162 origins and leaves the fold count at two; the test block
  and its 60 origins are unchanged.
