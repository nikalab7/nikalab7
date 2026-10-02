"""Power of the reference-portfolio test for a true net edge per trade.

Three slots of 10% each, two-session holds, a slot refilled whenever it frees up,
and real-world large-cap volatility (1.8% daily, of which 1.0% market). The test
is a normal approximation of the registered one: the lower bound of the mean
daily portfolio return must exceed zero. Overlapping two-session holds make the
daily series only weakly autocorrelated, so the approximation and the
registered block bootstrap give similar interval widths.

Run: ``python analysis/power_portfolio.py`` (a few minutes). See analysis/README.md.
"""

import numpy as np

rng = np.random.default_rng(11)
market_sd, idio_sd = 0.010, 0.015          # real-world large caps: ~1.8% total daily vol
weight, slots = 0.10, 3


def portfolio_returns(mu, sessions, slots=slots, weight=weight):
    """Fill every free slot each day; each trade earns mu over its two sessions."""
    market = rng.normal(0, market_sd, sessions + 2)
    open_trades = []                       # remaining sessions per open trade
    daily = np.zeros(sessions)
    for day in range(sessions):
        open_trades = [left for left in open_trades if left > 0]
        while len(open_trades) < slots:
            open_trades.append(2)
        r = 0.0
        for _ in open_trades:
            r += weight * (market[day] + rng.normal(0, idio_sd) + mu / 2)
        daily[day] = r
        open_trades = [left - 1 for left in open_trades]
    return daily


def power(mu, sessions, z, trials=2000, slots=slots, weight=weight):
    hits = 0
    for _ in range(trials):
        d = portfolio_returns(mu, sessions, slots, weight)
        se = d.std(ddof=1) / np.sqrt(sessions)
        hits += (d.mean() - z * se) > 0
    return hits / trials


print("P(lower bound > 0) for a true NET edge per 2-session trade")
print(f"{'edge/trade':>11} | dev 40s @90% | test 60s @95% | 120s @95% | 250s @95% | 250s @95%, 10 slots x3%")
for mu in (0.0005, 0.001, 0.002, 0.004, 0.007):
    row = [power(mu, 40, 1.645), power(mu, 60, 1.96), power(mu, 120, 1.96), power(mu, 250, 1.96),
           power(mu, 250, 1.96, slots=10, weight=0.03)]
    print(f"{mu*100:>10.2f}% | " + " | ".join(f"{p:>11.2f}" for p in row))
print("(size check, edge 0):", power(0.0, 60, 1.96))
