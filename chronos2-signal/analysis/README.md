# Analysis scripts

Offline experiments behind [docs/ANALYSIS.md](../docs/ANALYSIS.md). They answer one
question: **can the registered design produce a credible positive result?** None of
them reads market data or model weights, and none of their numbers is a performance
claim.

| Script | What it measures | Runtime |
| --- | --- | --- |
| `positive_control.py` | The whole registered pipeline on a synthetic market with a planted, causal, persistent relative-strength edge of known size | ~15 min for 40 stocks |
| `edge_ceiling.py` | The best any model could do with that planted edge (an optimal Kalman-filter estimate) | seconds |
| `power_portfolio.py` | How often the reference-portfolio test confirms a true per-trade edge, at the registered sample sizes | a few minutes |
| `power_rank_ic.py` | The same for a cross-sectional rank-IC test on the same dates | about a minute |

Run from the project root with the package installed:

```bash
for a in 0 10 20 40; do
  python analysis/positive_control.py --alpha $a --out pc_a$a.json &
done; wait
python analysis/edge_ceiling.py
python analysis/power_portfolio.py
python analysis/power_rank_ic.py
```

`positive_control.py` mirrors the real chronology: about 224 labelled origins, two
development folds and a 60-origin final test. It uses the deterministic stub
forecaster, so C256 carries no information beyond B0's features; the experiment tests
the decision heads, the alert rule, the portfolio and the statistics, not Chronos-2.
