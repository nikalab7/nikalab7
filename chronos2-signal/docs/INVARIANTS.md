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
| Origin-bounded inputs | `sources.MarketSource.view` — every panel a view returns ends at its session — plus `features.build_chronos_task` and `calendar_spec.ExchangeCalendar.bars_ending_at` |
| Vintage-bounded inputs | `sources.LedgerMarketSource` at `Vintage.POINT_IN_TIME`, reading `storage.Ledger.read_bars(as_of=...)`; a `LATEST` backfill says so in every report |
| Label availability timestamps | `protocol.label_available_at`, `decision._usable_rows`, `protocol._assert_no_label_leak`, `protocol._assert_development_precedes_test` |
| Holdout lock | `holdout.HoldoutGuard`, checked by `pipeline.ResearchPipeline.build_batch`, `.label`, `.realised_log_return`, `simulation.simulate` and `storage.Ledger`; studies verify afterwards with `operations._verify_holdout` |
| Bar schedule and half days | `calendar_spec.ExchangeCalendar.session_bars`, `.horizon` |
| No synthetic bars | `market.BarPanel.aligned` (rejects unscheduled timestamps, masks missing ones) |
| Split convention audit | `sources._SplitConsistentView` audits every split in a view's data and restates the ones the provider did not; `pipeline._audits_in_window` passes the rest to eligibility, which quarantines |
| As-of units for screens | `actions.to_asof_units`, `quality.liquidity_metrics` |
| Entry at the next open | `simulation.simulate`, `portfolio.ReferencePortfolio.allocate` |
| One set of units per position | `simulation._unit_change`, `portfolio.ReferencePortfolio.restate_units` |
| Scoring after the close | `simulation.simulate` — exits and the mark precede scoring |
| Median naming, crossed quantiles | `forecaster.ForecastResult.forecast_median`, `forecaster.validate_quantile_paths` |
| Output orientation and unpacking | `forecaster.Chronos2Forecaster._to_arrays`, `forecaster._orient_levels_by_horizon` |
| Batching by channels, not symbols | `forecaster.batch_tasks_by_channels` |
| Forecasts before outcomes | `forecast_store.ForecastCache.predict`, called from `pipeline.ResearchPipeline.build_batch` |
| Append-only forecasts | SQLite triggers in `storage._SCHEMA`, plus `Ledger.record_forecast` |
| One alert per deduplication key | `policy.dedup_key`, `Ledger.record_signal` (unique index) |
| Commit before notify | `operations.run_after_close` (one transaction per origin), `notifier.Notifier.notify_session` |
| One scoring path | `simulation.score_origin`, used by the backtest and by `operations.run_after_close` |
| Identical accounting for controls | every system runs through `simulation.simulate`; `evaluation.paired_block_bootstrap` refuses mismatched date indices |
| Output label fails closed | `config.DesignConfig.is_validated` (allowlist), `policy.PolicyEngine.output_label` |

## Components that were correct but not wired in

An architecture review on 30 September 2026 found that several guarantees in the table
above were implemented as library functions that the orchestration never called. The
component tests passed regardless, because they called the components directly. The
split audit never ran; the holdout guard protected ledger reads but not the research
path; the ledger-backed data source did not exist; forecasts were never written to the
ledger; B0 was missing from the study runner; gate 7 received hard-coded `NaN`; and the
live after-close run and the backtest scored through two separate copies of the same
logic.

`tests/test_system.py` now checks each of these through the pipeline. Every one of its
tests was **mutation-checked**: the defect it guards against was reintroduced, the test
was confirmed to fail, and the code was restored. The first attempt at the ordering
mutation was itself wrong — it added an early scoring call rather than moving the
existing one, so the selector's last observation was still correct and the test passed.
Only a faithful mutation shows whether a test catches anything.

## Defects the tests caught

Recorded because a test suite is only credible if it has actually rejected something.

1. **The output label failed open.** `is_validated` was written as a denylist, so any
   status string it did not recognise — a typo, a local experiment — read as
   *validated* and promoted the output to `QUALIFIED SIGNAL`. It is now an allowlist,
   and the loader rejects an unrecognised status outright.
   (`test_output_label_fails_closed_on_an_unknown_status`)
2. **Promotion gate 6 passed on absent evidence.** With no executed trades there is no
   contributor share to measure, and the gate reported a pass. Gates now fail closed on
   absent evidence. (`test_promotion_gates_fail_closed`)

Found while wiring the components in, 30 September 2026:

3. **Scoring ran before the close.** The walk-forward loop scored an origin *before*
   that session's exits, so positions leaving at the close still counted against
   capacity — three exiting positions blocked every new alert for a day — and the
   drawdown pause lagged a session.
   (`test_scoring_happens_after_the_close_against_post_exit_state`)
4. **A retried quiet origin notified twice.** The zero-alert notice went straight to
   the channel, bypassing the idempotency the alert path had. Invariant 10's test only
   covered the alerting case. (`test_a_retried_quiet_origin_sends_one_notice`)
5. **A two-task forecast batch would have been misread.** `predict_quantiles` returns a
   `(quantiles, mean)` pair; a generic sequence branch would have read it as "task one
   = quantiles, task two = mean" whenever a batch held exactly two tasks.
   (`test_two_task_batch_is_not_read_as_quantiles_and_mean`)
6. **Re-recording an identical forecast raised.** Its idempotency check compared the
   generation timestamp, which always differs between runs, so the documented no-op
   retry was impossible. (`test_forecasts_are_written_to_the_ledger_at_batch_time`)
7. **The development/test boundary was not purged.** The last development origins'
   outcome windows reached into the first test origins' holding windows, so development
   selection could see part of a reserved outcome. (`test_the_dev_test_boundary_is_purged`)
8. **Correlation compared different days.** Each symbol's own last sixty returns were
   correlated by position, so one missing session in either series shifted them against
   each other. (`test_correlation_is_computed_on_shared_sessions`)
9. **A missing close was marked at the entry fill**, hiding any drawdown since entry.
   (`test_a_missing_close_is_marked_at_the_last_observed_close`)

Found by reading the new session engine before it was committed. No test covered it;
the test below was written first and failed against the unfixed engine:

10. **A split inside a hold was booked as a 50% loss.** Each view is split-consistent
    within itself, but a view before an ex-date quotes pre-split prices and one after
    it post-split prices — always at a point-in-time vintage, and at any vintage when
    the provider left the split unrestated. The engine entered from one view and
    exited from the other, so a 2-for-1 split turned a +0.04% trade into −49.98%, with
    a −4.9% portfolio day. The engine now re-reads each position's entry quote from
    the current view and re-expresses the position when the units changed by exactly
    the splits the ledger records inside the hold. A change that no recorded split
    explains leaves the position unpriced rather than valued in the wrong units.
    (`test_a_split_inside_a_hold_changes_units_not_value`,
    `test_a_requoted_entry_is_neither_a_split_nor_a_return`)
