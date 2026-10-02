"""The best any model could do with the planted edge of positive_control.py.

The planted edge is a latent AR(1) daily drift observed only through noisy
returns, so its optimal causal estimate is a steady-state Kalman filter. This
script reports, per edge size, the IC of that optimal estimate against realised
holding returns and the expected gross return of its top three of forty names.

Run: ``python analysis/edge_ceiling.py`` (seconds). See analysis/README.md.
"""

import numpy as np

rng = np.random.default_rng(7)
phi = 0.95
obs_sd = np.sqrt(0.0093**2 + 0.0045**2)          # idiosyncratic session + overnight
hold_noise_sd = np.sqrt(2 * (0.0093**2 + 0.0053**2 + 0.0040**2) + 0.0045**2)  # long-only hold
for alpha_bps in (10, 20, 40):
    s = alpha_bps / 1e4
    n_stocks, n_days = 400, 1500
    q = s**2 * (1 - phi**2)
    a = np.zeros((n_stocks, n_days))
    a[:, 0] = rng.normal(0, s, n_stocks)
    for d in range(1, n_days):
        a[:, d] = phi * a[:, d - 1] + rng.normal(0, np.sqrt(q), n_stocks)
    y = a + rng.normal(0, obs_sd, a.shape)
    # steady-state Kalman filter
    p = s**2
    for _ in range(500):
        p_pred = phi**2 * p + q
        k = p_pred / (p_pred + obs_sd**2)
        p = (1 - k) * p_pred
    est = np.zeros_like(a)
    m = np.zeros(n_stocks)
    for d in range(n_days):
        m = phi * m
        m = m + k * (y[:, d] - m)
        est[:, d] = m
    origins = range(100, n_days - 3)
    ics, top3 = [], []
    for d in origins:
        signal = est[:, d]
        realised = a[:, d + 1] + a[:, d + 2] + rng.normal(0, hold_noise_sd, n_stocks)
        ics.append(np.corrcoef(signal, realised)[0, 1])
        # top 3 of a 40-name universe, expected gross alpha over the hold
        sub = rng.choice(n_stocks, 40, replace=False)
        best = sub[np.argsort(-signal[sub])[:3]]
        top3.append(np.mean(a[best, d + 1] + a[best, d + 2]))
    print(f"alpha {alpha_bps:>2} bps/day: best IC {np.mean(ics):.3f}  "
          f"top-3-of-40 expected gross per trade {np.mean(top3)*100:.3f}%  "
          f"(costs 0.20% round trip)")
