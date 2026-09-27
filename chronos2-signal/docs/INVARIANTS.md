# The twelve integrity invariants

Section 16 of the [registered protocol](DESIGN.md) requires these to pass on small
controlled fixtures **before** a large universe is fetched or a backtest is run. They
run offline: no network, no checkpoint, no performance claim.

```
pytest tests/test_invariants.py -v
```

| # | Invariant | Test | What would break if it failed |
| --- | --- | --- | --- |
| 1 | Appending or changing future prices cannot change a feature or forecast input at an earlier origin | `test_invariant_01_future_prices_cannot_change_earlier_inputs` | Every backtest number would embed knowledge the decision could not have had |
| 2 | Changing a future outcome cannot change an earlier fit, calibration or alert | `test_invariant_02_future_outcomes_cannot_change_an_earlier_fit`, `test_invariant_02b_holdout_outcomes_are_unreadable_in_development` | The model would be fitted on labels that had not happened yet |
| 3 | A normal session, an early close, a DST transition, a weekend and a holiday produce correct bar and exit timestamps | `test_invariant_03_calendar_edge_cases_produce_correct_timestamps` | Horizons and exits would land on the wrong day around half days and DST |
| 4 | The incomplete current bar is excluded; synthetic overnight flat bars are never introduced | `test_invariant_04_incomplete_and_synthetic_bars_are_excluded` | The model would read a partial bar as complete, and unequal time steps would look contiguous |
| 5 | Split and dividend fixtures preserve economic wealth and create no false momentum, wrong liquidity or double-counted cash | `test_invariant_05_splits_and_dividends_preserve_wealth` | A corporate action would manufacture a return |
| 6 | A positive D0-close-to-D1-open gap contributes no captured return | `test_invariant_06_overnight_gap_is_not_captured` | The strategy would be credited with a move that happened before it could enter |
| 7 | Every stock of an origin shares its temporal split; no overlapping unknown label crosses a training boundary | `test_invariant_07_dates_are_atomic_and_labels_do_not_cross_boundaries` | Outcome windows would leak across the purge and inflate validation |
| 8 | With cross-learning disabled, reordering tasks or changing the batch budget does not change predictions | `test_invariant_08_task_order_and_batch_budget_are_irrelevant` | A forecast would depend on which other stocks shared its batch |
| 9 | Quantile axes, levels, terminal indices and log-price inverse transforms are correct; the point forecast is treated as a median | `test_invariant_09_quantile_axes_and_inverse_transforms` | A median would be read as an expected profit, or padding as signal |
| 10 | Duplicate runs produce one persisted signal and one notification; failed data cannot generate a fresh-looking signal | `test_invariant_10_duplicate_runs_and_failed_data` | A retry would double-alert, and an outage would emit a stale signal as current |
| 11 | Portfolio cash, simultaneous holds, correlation limits and corporate-action accounting reconcile exactly | `test_invariant_11_portfolio_ledger_reconciles` | Reported returns would not match the ledger that produced them |
| 12 | Baseline and candidate tests use identical dates, fills, costs and capital accounting | `test_invariant_12_baselines_share_dates_fills_costs_and_capital` | An "improvement over baseline" would be a comparison of samples, not of systems |

## Where each guarantee is enforced

The tests verify behaviour; these are the places that make it true.

| Guarantee | Enforced in |
| --- | --- |
| Origin-bounded inputs | `features.build_chronos_task`, `market.DailyPanel.up_to`, `calendar_spec.ExchangeCalendar.bars_ending_at` |
| Label availability timestamps | `protocol.label_available_at`, `decision._usable_rows`, `protocol._assert_no_label_leak` |
| Holdout lock | `holdout.HoldoutGuard`, used by `storage.Ledger` and `protocol.ProtocolSchedule.guard` |
| Bar schedule and half days | `calendar_spec.ExchangeCalendar.session_bars`, `.horizon` |
| No synthetic bars | `market.BarPanel.aligned` (rejects unscheduled timestamps, masks missing ones) |
| Split convention audit | `actions.audit_split_convention`, `actions.to_split_consistent` (quarantines rather than guesses) |
| As-of units for screens | `actions.to_asof_units`, `quality.liquidity_metrics` |
| Entry at the next open | `pipeline.WalkForwardRunner.run`, `portfolio.ReferencePortfolio.allocate` |
| Median naming, crossed quantiles | `forecaster.ForecastResult.forecast_median`, `forecaster.validate_quantile_paths` |
| Batching by channels, not symbols | `forecaster.batch_tasks_by_channels` |
| Append-only forecasts | SQLite triggers in `storage._SCHEMA`, plus `Ledger.record_forecast` |
| One alert per deduplication key | `policy.dedup_key`, `Ledger.record_signal` (unique index) |
| Commit before notify | `operations.run_after_close`, `notifier.Notifier.notify_session` |
| Identical accounting for controls | every system runs through `portfolio.ReferencePortfolio`; `evaluation.paired_block_bootstrap` refuses mismatched date indices |
| Output label fails closed | `config.DesignConfig.is_validated` (allowlist), `policy.PolicyEngine.output_label` |

## Two defects these tests caught during implementation

Recorded because a test suite is only credible if it has actually rejected something.

1. **The output label failed open.** `is_validated` was written as a denylist, so any
   status string it did not recognise — a typo, a local experiment — read as
   *validated* and promoted the output to `QUALIFIED SIGNAL`. It is now an allowlist,
   and the loader rejects an unrecognised status outright.
   (`test_output_label_fails_closed_on_an_unknown_status`)
2. **Promotion gate 6 passed on absent evidence.** With no executed trades there is no
   contributor share to measure, and the gate reported a pass. Gates now fail closed on
   absent evidence. (`test_promotion_gates_fail_closed`)
