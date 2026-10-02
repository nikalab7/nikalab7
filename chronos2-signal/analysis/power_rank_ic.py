"""Power of a mean daily cross-sectional rank-IC test, 50 names per date.

Run: ``python analysis/power_rank_ic.py`` (about a minute). See analysis/README.md.
"""

import numpy as np

from chronos2_signal.evaluation import spearman_correlation

rng = np.random.default_rng(5)
names = 50


def power(rho, dates, z, trials=1500):
    hits = 0
    for _ in range(trials):
        ics = np.empty(dates)
        for d in range(dates):
            signal = rng.normal(size=names)
            realised = rho * signal + np.sqrt(1 - rho**2) * rng.normal(size=names)
            ics[d] = spearman_correlation(signal, realised)
        hits += ics.mean() - z * ics.std(ddof=1) / np.sqrt(dates) > 0
    return hits / trials


print(f"{'true IC':>8} | dev 40 dates @90% | test 60 dates @95%")
for rho in (0.0, 0.02, 0.03, 0.05, 0.10):
    print(f"{rho:>8.2f} | {power(rho, 40, 1.645):>17.2f} | {power(rho, 60, 1.96):>18.2f}")
