"""Power of the amended primary test: daily rank IC, candidate minus B0, same rows.

Each date has 50 names. The realised return carries information B0 also sees
(``s_b``), information only the candidate sees (``s_x``) and noise. Both fitted
heads are noisy. The reported gain is the *realised* mean daily IC difference,
which is what the registered test estimates, not the planted coefficient.

Run: ``python analysis/power_ic_difference.py`` (a few minutes). See
analysis/README.md.
"""

import numpy as np

from chronos2_signal.evaluation import spearman_correlation

rng = np.random.default_rng(17)
NAMES = 50
B0_SIGNAL = 0.03


def daily_gain(planted: float) -> float:
    s_b, s_x, noise = rng.normal(size=(3, NAMES))
    realised = B0_SIGNAL * s_b + planted * s_x + noise
    b0 = s_b + 0.5 * rng.normal(size=NAMES)
    candidate = s_b + s_x + 0.5 * rng.normal(size=NAMES)
    return spearman_correlation(candidate, realised) - spearman_correlation(b0, realised)


def power(planted: float, dates: int, z: float, trials: int = 800) -> float:
    hits = 0
    for _ in range(trials):
        gains = np.asarray([daily_gain(planted) for _ in range(dates)])
        hits += gains.mean() - z * gains.std(ddof=1) / np.sqrt(dates) > 0
    return hits / trials


print(f"{'IC gain':>8} | dev 40 dates @90% | test 60 dates @95%")
for planted in (0.0, 0.03, 0.05, 0.08, 0.12):
    gain = float(np.mean([daily_gain(planted) for _ in range(6000)]))
    print(f"{gain:>+8.3f} | {power(planted, 40, 1.645):>17.2f} | {power(planted, 60, 1.96):>18.2f}")
