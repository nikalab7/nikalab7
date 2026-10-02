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
| One universe mask for every variant | `pipeline.ResearchPipeline.panel_bars` — every variant loads the panel the longest registered context needs |
| Daily windows count sessions | `features.build_daily_features(calendar=...)`, `market.DailyPanel.on_sessions`; the origin's daily bar is required |
| No bar still forming at retrieval | `collector.Ingestor._persist_bars` |
| Exposure-matched benchmark | `variants.exposure_matched_series`, `operations._matched_series` — the same capital over the same intervals, entry day included |
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

That first pass also covered only about half of the tests, although this section
already claimed all of them. A second pass reintroduced a defect for every remaining
test — eighteen mutations, all caught — and found two tests that could not fail
against the defect they were credited with: the holdout check tested only the
wording of its note, and the forecast-retry check reused the original timestamp. Both
were strengthened, and the old versions were confirmed to pass under the mutation that
the new ones catch.

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
   retry was impossible. Invariant 10's test passed regardless because its "retry"
   reused the original timestamp; it now retries later, as a real retry does.
   (`test_invariant_10_duplicate_runs_and_failed_data`)
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

Found by an independent review on 2 October 2026: three reviewers, each taking a group
of the modules that determine results, every finding then reproduced before it was
accepted. Each fix has a test that failed before it.

11. **Refits skipped the purge between fit and calibration.** In the final test, the
    last fit origin and the first calibration origin shared a holding session; the
    folds had the registered two-session gap. Refits now use the fold recipe
    exactly, including its fit deadline.
    (`test_refits_follow_the_fold_recipe_including_both_purges`)
12. **The exposure-matched benchmark missed every entry day.** It applied the previous
    close's exposure, but a D1 entry is decided at D0's after-close run and is known
    in advance. A zero-skill strategy that simply bought SPY showed excess return over
    "the same exposure" on 17 of 35 sessions.
    (`test_holding_the_market_shows_no_excess_over_the_matched_benchmark`)
13. **C512 could see a different universe.** It loaded more history than the other
    variants, so an unresolvable split in that extra stretch removed the stock for
    C512 alone. (`test_every_variant_shares_one_universe_mask`)
14. **A missing origin-day daily bar was replaced by yesterday's**, and windows
    counted rows: one missing day turned a five-session return into a six-session
    one. (`test_a_missing_origin_daily_row_is_not_replaced_by_yesterday`,
    `test_trailing_returns_count_sessions_not_rows`)
15. **Gate 5 counted a two-session tail as one of the three 20-session blocks.**
    (`test_gate_five_counts_only_full_twenty_session_blocks`)
16. **An unresolved trade stood in for the best trade.** `argmax` returns a NaN's
    position, so the real best trade was never removed in the outlier check.
    (`test_an_unresolved_trade_does_not_stand_in_for_the_best_one`)
17. **Gate 6 ignored sector concentration** and passed a net loss as "not
    concentrated". (`test_gate_six_restricts_a_claim_explained_by_one_sector`)
18. **The selection rule had a condition section 12 does not contain** — a positive
    lower bound for the leader — so a candidate conclusively better than C256 could
    still be passed over. (`test_selection_follows_the_registered_rule`)
19. **The momentum control ignored the correlation cap**, one of "the same capacity
    rules". Both systems now use one function.
    (`test_momentum_control_respects_the_same_capacity_rules`)
20. **The registered 5/20-session bootstrap sensitivity was never computed.**
    (`test_studies_report_the_registered_block_length_sensitivity`)
21. **A bar still forming at retrieval was stored as complete**, so a pre-close fetch
    could feed a partial bar to the model as finished.
    (`test_a_bar_still_in_progress_at_retrieval_is_not_stored`)
22. **Calendar answers depended on the order of earlier lookups.**
    (`test_calendar_answers_do_not_depend_on_lookup_order`)

Latent — unreachable by any current caller, fixed so they cannot become live:

23. A training label without an availability time passed the leakage check.
    (`test_a_label_without_an_availability_time_is_dropped_under_a_deadline`)
24. An explicit broker fee was never charged by the portfolio.
    (`test_an_explicit_fee_is_charged_exactly_as_the_label_charges_it`)
25. A stress-cost run trained on stress labels, counting the stress step twice, and
    recorded its stress return as the base return.
    (`test_a_stress_cost_run_still_records_each_trade_at_every_cost_level`)
26. "Alert coverage" counted executed trades; alerts now come from the selector's
    own record.
