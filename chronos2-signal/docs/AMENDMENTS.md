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
