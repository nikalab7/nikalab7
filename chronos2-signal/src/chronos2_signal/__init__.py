"""Chronos-2 hourly stock signal system.

A free-data, long-only research scanner for a frozen watchlist of up to 50 liquid
US common stocks, implemented to the preregistered protocol in ``docs/DESIGN.md``.

**Status: unvalidated. No edge is claimed and none has been tested.** No market
data has been downloaded by this package, no Chronos-2 weights are bundled, and
no notification channel is configured. Every rendered output carries the
``RESEARCH / UNVALIDATED`` label until the registered evaluation gates pass, and
:attr:`chronos2_signal.config.DesignConfig.is_validated` fails closed so that an
unrecognised status cannot promote a label by accident.

The submodules mirror the protocol's component list (section 15):

==================== =========================================================
Module               Responsibility
==================== =========================================================
``config``           strict loader for the registered configuration
``calendar_spec``    sessions, hourly bars, half days, DST, horizons
``collector``        bounded provider fetches and immutable snapshots
``storage``          SQLite ledger and Parquet snapshot store
``holdout``          access control for the reserved final-test origins
``market``           panels aligned to the expected bar schedule
``actions``          split audits, as-of units and wealth accounting
``quality``          per-origin eligibility, one mask for every variant
``universe``         candidate roster, selection rule and freeze guard
``features``         the twelve task channels and the twenty-one columns
``forecaster``       frozen-checkpoint adapter and an offline stub
``decision``         the return and probability heads plus calibration
``policy``           thresholds, ranking, capacity and deduplication
``portfolio``        the reference paper portfolio
``protocol``         labelled origins, folds, purges, holdout and refits
``variants``         the five registered variants and the fixed controls
``evaluation``       metrics, the date-block bootstrap and promotion gates
``pipeline``         origin batches, labels and the walk-forward runner
``operations``       the after-close run and the registered study
``notifier``         renders persisted signals; contacts nothing
``provenance``       hashes, cache keys and release manifests
``fixtures``         a deterministic synthetic market, for tests only
==================== =========================================================

Nothing is imported eagerly here: the provider and model extras are optional, and
importing this package must not require them.
"""

from __future__ import annotations

__all__ = ["__version__", "DESIGN_VERSION", "STATUS", "load_design"]

__version__ = "0.1.0"

#: The registered protocol version this code implements.
DESIGN_VERSION = "chronos2_hourly_v2"

#: Short, deliberately blunt statement of what this package has established.
STATUS = (
    "design_only_unvalidated: no market data downloaded, no checkpoint loaded, "
    "no backtest run, no edge claimed"
)


def load_design(path=None):
    """Load and validate the registered design configuration.

    Thin re-export of :func:`chronos2_signal.config.load_design`, kept here so
    that the common entry point is available without importing a submodule.
    """
    from .config import load_design as _load_design

    return _load_design(path)
