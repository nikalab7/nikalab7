"""Origin-bounded feature construction.

Two families are built here and kept separate:

* **Chronos task channels** -- the twelve aligned series the frozen forecaster
  consumes: one target, six past-only channels and five known-future calendar
  channels. Only the calendar extends into the future. Future market, sector,
  volume, range and price values are unknown and are never supplied.
* **Decision features** -- the twenty economic features (six derived from the
  forecast, fourteen from price, volume and market context) plus one
  missingness indicator, giving a twenty-one column preprocessing matrix.

Every value is computed from data ending at the origin bar. The target is
rebased on the origin close, so no scaler is fitted on the full dataset and
appending later prices cannot change an earlier row.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .calendar_spec import (
    CALENDAR_CHANNEL_NAMES,
    BarSpec,
    ExchangeCalendar,
    calendar_channel_matrix,
)
from .market import BarPanel, DailyPanel

__all__ = [
    "FeatureError",
    "PAST_CHANNEL_NAMES",
    "TARGET_CHANNEL_NAME",
    "CHRONOS_CHANNEL_NAMES",
    "FEATURE_NAMES",
    "FORECAST_FEATURE_NAMES",
    "NON_FORECAST_FEATURE_NAMES",
    "BREADTH_FEATURE_NAME",
    "BREADTH_MISSING_FLAG",
    "PREPROCESSING_COLUMNS",
    "VOLUME_LOOKBACK_SESSIONS",
    "ChronosTask",
    "build_chronos_task",
    "recommended_panel_bars",
    "DailyFeatures",
    "build_daily_features",
    "sigma_2d",
    "forecast_features",
    "assemble_feature_row",
]


class FeatureError(RuntimeError):
    """Raised when a feature cannot be computed from origin-bounded data."""


#: Sessions of history used for the matching-slot volume median.
VOLUME_LOOKBACK_SESSIONS = 20

TARGET_CHANNEL_NAME = "target_rebased_log_close_x100"

PAST_CHANNEL_NAMES = (
    "market_rebased_log_close_x100",
    "sector_rebased_log_close_x100",
    "relative_volume_log",
    "bar_range_log_x100",
    "close_location",
    "intrabar_move_log_x100",
)

CHRONOS_CHANNEL_NAMES = (
    TARGET_CHANNEL_NAME,
    *PAST_CHANNEL_NAMES,
    *CALENDAR_CHANNEL_NAMES,
)

#: The six features derived from the forecast, in registered order 1-6.
FORECAST_FEATURE_NAMES = (
    "f01_median_return_d1_over_sigma",
    "f02_median_return_d2_over_sigma",
    "f03_d2_p90_p10_width_over_sigma",
    "f04_d2_quantile_asymmetry",
    "f05_d2_median_over_width",
    "f06_d2_minus_d1_median_over_sigma",
)

#: The fourteen features available without a forecast, in registered order 7-20.
NON_FORECAST_FEATURE_NAMES = (
    "f07_stock_return_1s",
    "f08_stock_return_5s",
    "f09_stock_return_20s",
    "f10_stock_minus_sector_return_5s",
    "f11_volume_ratio_20s",
    "f12_stock_daily_vol_20s",
    "f13_vol_ratio_20s_60s",
    "f14_distance_from_ema20_over_vol",
    "f15_drawdown_from_20s_high",
    "f16_beta_to_market_60s",
    "f17_market_return_5s",
    "f18_market_vol_20s",
    "f19_sector_minus_market_return_5s",
    "f20_sector_peer_breadth_above_ema20",
)

FEATURE_NAMES = (*FORECAST_FEATURE_NAMES, *NON_FORECAST_FEATURE_NAMES)

#: The one optional feature, and its fixed missingness indicator.
BREADTH_FEATURE_NAME = "f20_sector_peer_breadth_above_ema20"
BREADTH_MISSING_FLAG = "f20_missing_indicator"

#: Twenty economic features plus the indicator: the preprocessing matrix has
#: twenty-one columns while the economic feature list has twenty.
PREPROCESSING_COLUMNS = (*FEATURE_NAMES, BREADTH_MISSING_FLAG)

#: Minimum eligible sector peers before breadth is meaningful.
MIN_BREADTH_PEERS = 3


# --------------------------------------------------------------------------- #
# Chronos task channels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChronosTask:
    """One forecasting task: a single stock at a single origin.

    Attributes:
        symbol: Target symbol.
        origin_session: The completed signal session.
        origin_close: Split-consistent close of the final context bar, used to
            invert the rebased target back into prices.
        context_bars: The context schedule, oldest first.
        past: ``(context, n_channels)`` matrix of target and past-only channels.
        future_calendar: ``(horizon, 5)`` matrix of known-future calendar values.
        past_calendar: ``(context, 5)`` matrix of matching historical calendar values.
        channel_names: Names in column order of ``past``.
        diagnostics: Counts a quality monitor needs (missing bars, masked volumes).
    """

    symbol: str
    origin_session: dt.date
    origin_close: float
    context_bars: tuple[BarSpec, ...]
    horizon_bars: tuple[BarSpec, ...]
    past: np.ndarray
    past_calendar: np.ndarray
    future_calendar: np.ndarray
    channel_names: tuple[str, ...]
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @property
    def context_length(self) -> int:
        return self.past.shape[0]

    @property
    def prediction_length(self) -> int:
        return self.future_calendar.shape[0]

    @property
    def n_channels(self) -> int:
        """Total channels in the task, including the calendar ones."""
        return self.past.shape[1] + self.future_calendar.shape[1]

    @property
    def target(self) -> np.ndarray:
        return self.past[:, 0]

    def past_covariates(self) -> np.ndarray:
        """Past-only channels, excluding the target."""
        return self.past[:, 1:]

    def price_from_target(self, target_value: float) -> float:
        """Invert the rebased log target back into a price.

        ``C_origin * exp(q / 100)``. Applying this to a forecast quantile gives
        the implied price quantile at that horizon -- and nothing more: marginal
        quantiles at several future times do not describe a joint path.
        """
        return self.origin_close * math.exp(target_value / 100.0)

    def return_from_target(self, target_value: float) -> float:
        """Simple return implied by a rebased log target value."""
        return math.expm1(target_value / 100.0)

    def validate(self) -> None:
        """Structural checks that must hold before inference."""
        if self.past.shape[0] != len(self.context_bars):
            raise FeatureError(
                f"{self.symbol}: past matrix has {self.past.shape[0]} rows for "
                f"{len(self.context_bars)} context bars"
            )
        if self.past_calendar.shape[0] != self.past.shape[0]:
            raise FeatureError(f"{self.symbol}: calendar history misaligned with context")
        if self.future_calendar.shape[0] != len(self.horizon_bars):
            raise FeatureError(f"{self.symbol}: calendar future misaligned with horizon")
        if not np.isfinite(self.past_calendar).all() or not np.isfinite(self.future_calendar).all():
            raise FeatureError(f"{self.symbol}: calendar channels must be finite")
        if abs(float(self.past[-1, 0])) > 1e-9:
            raise FeatureError(
                f"{self.symbol}: the target's final value must be exactly zero after "
                f"rebasing on the origin close, got {self.past[-1, 0]!r}"
            )
        if not math.isfinite(self.origin_close) or self.origin_close <= 0.0:
            raise FeatureError(f"{self.symbol}: origin close must be finite and positive")


def recommended_panel_bars(context_length: int, *, bars_per_session: int = 7) -> int:
    """Bars to load so that every context bar has a full volume lookback.

    The matching-slot volume median needs twenty earlier sessions, so a context
    window alone is not enough history to compute the channel at its left edge.
    """
    return context_length + bars_per_session * (VOLUME_LOOKBACK_SESSIONS + 1)


def build_chronos_task(
    *,
    symbol: str,
    origin_session: dt.date,
    calendar: ExchangeCalendar,
    stock: BarPanel,
    market: BarPanel,
    sector: BarPanel,
    context_length: int,
    horizon_sessions: int = 2,
    include_covariates: bool = True,
) -> ChronosTask:
    """Assemble the twelve aligned channels for one stock at one origin.

    Args:
        stock: Split-consistent hourly panel ending at the origin bar. It should
            carry at least :func:`recommended_panel_bars` rows so the volume
            channel is defined across the whole context.
        market: Market proxy panel on the same schedule.
        sector: Frozen sector ETF panel on the same schedule.
        include_covariates: ``False`` builds the target-and-calendar-only task
            used by the registered ``U256`` variant.

    Raises:
        FeatureError: If the panels are misaligned, the context is short, or the
            origin bar itself has no observed close. An origin without a
            rebasing price cannot be forecast, and must not be padded.
    """
    if context_length < 1:
        raise FeatureError("context length must be positive")
    for name, panel in (("market", market), ("sector", sector)):
        if len(panel) != len(stock) or not panel.index.equals(stock.index):
            raise FeatureError(
                f"{symbol}: {name} panel is not aligned to the stock bar index; "
                "channels must share one schedule exactly"
            )
    if len(stock) < context_length:
        raise FeatureError(
            f"{symbol}: panel holds {len(stock)} bars, need at least {context_length}"
        )
    if stock.bars[-1].session != origin_session:
        raise FeatureError(
            f"{symbol}: panel ends on {stock.bars[-1].session.isoformat()}, expected the "
            f"origin {origin_session.isoformat()}"
        )

    closes = stock.column("close")
    if not stock.observed[-1] or not math.isfinite(closes[-1]) or closes[-1] <= 0.0:
        raise FeatureError(
            f"{symbol}: the origin bar has no observed close; the target cannot be "
            "rebased and this origin is skipped rather than padded"
        )
    origin_close = float(closes[-1])

    target = _rebased_log(closes, origin_close)
    diagnostics: dict[str, float] = {
        "missing_target_bars": float(int((~stock.observed[-context_length:]).sum())),
    }

    channels: list[np.ndarray] = [target]
    names: list[str] = [TARGET_CHANNEL_NAME]

    if include_covariates:
        market_closes = market.column("close")
        sector_closes = sector.column("close")
        market_anchor = _anchor_close(market_closes, market.observed, symbol, "market")
        sector_anchor = _anchor_close(sector_closes, sector.observed, symbol, "sector")
        relative_volume, masked_volume = _relative_volume(stock)
        channels.extend(
            [
                _rebased_log(market_closes, market_anchor),
                _rebased_log(sector_closes, sector_anchor),
                relative_volume,
                _bar_range(stock),
                _close_location(stock),
                _intrabar_move(stock),
            ]
        )
        names.extend(PAST_CHANNEL_NAMES)
        diagnostics["masked_relative_volume_bars"] = float(masked_volume)
        diagnostics["missing_market_bars"] = float(
            int((~market.observed[-context_length:]).sum())
        )
        diagnostics["missing_sector_bars"] = float(
            int((~sector.observed[-context_length:]).sum())
        )

    past_full = np.column_stack(channels).astype(np.float32)
    past = past_full[-context_length:, :]
    context_bars = stock.bars[-context_length:]

    past_calendar = calendar_channel_matrix(list(context_bars))
    if not np.isfinite(past_calendar).all():
        # Only the very first bar of a window can lack a predecessor, and
        # bars_ending_at seeds it. A NaN here means the window was built by hand.
        raise FeatureError(
            f"{symbol}: calendar history contains an undefined overnight gap; build "
            "the context through ExchangeCalendar.bars_ending_at"
        )
    horizon = calendar.horizon(origin_session, horizon_sessions)
    future_calendar = calendar_channel_matrix(list(horizon.bars))

    task = ChronosTask(
        symbol=symbol,
        origin_session=origin_session,
        origin_close=origin_close,
        context_bars=tuple(context_bars),
        horizon_bars=horizon.bars,
        past=past,
        past_calendar=past_calendar,
        future_calendar=future_calendar,
        channel_names=tuple(names),
        diagnostics=diagnostics,
    )
    task.validate()
    return task


def _rebased_log(values: np.ndarray, anchor: float) -> np.ndarray:
    """``100 * ln(value / anchor)``, propagating NaN for unobserved bars."""
    array = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * (np.log(array) - math.log(anchor))
    out[~np.isfinite(array) | (array <= 0.0)] = np.nan
    return out


def _anchor_close(
    closes: np.ndarray, observed: np.ndarray, symbol: str, label: str
) -> float:
    """The origin-bar close used to rebase a covariate channel."""
    if observed[-1] and math.isfinite(closes[-1]) and closes[-1] > 0.0:
        return float(closes[-1])
    raise FeatureError(
        f"{symbol}: {label} channel has no observed close on the origin bar; the "
        "latest session must be complete for required inputs"
    )


def _bar_range(panel: BarPanel) -> np.ndarray:
    high = panel.column("high")
    low = panel.column("low")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * (np.log(high) - np.log(low))
    invalid = ~np.isfinite(high) | ~np.isfinite(low) | (high <= 0.0) | (low <= 0.0)
    out[invalid] = np.nan
    return out


def _close_location(panel: BarPanel) -> np.ndarray:
    high = panel.column("high")
    low = panel.column("low")
    close = panel.column("close")
    span = high - low
    out = np.full(close.shape, np.nan, dtype=float)
    valid = np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    flat = valid & (span <= 0.0)
    sloped = valid & (span > 0.0)
    # A bar with no range has no location inside it; zero is the neutral value
    # the protocol fixes, rather than a division by zero.
    out[flat] = 0.0
    out[sloped] = 2.0 * (close[sloped] - low[sloped]) / span[sloped] - 1.0
    return out


def _intrabar_move(panel: BarPanel) -> np.ndarray:
    open_ = panel.column("open")
    close = panel.column("close")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * (np.log(close) - np.log(open_))
    invalid = ~np.isfinite(open_) | ~np.isfinite(close) | (open_ <= 0.0) | (close <= 0.0)
    out[invalid] = np.nan
    return out


def _relative_volume(
    panel: BarPanel, *, lookback: int = VOLUME_LOOKBACK_SESSIONS
) -> tuple[np.ndarray, int]:
    """Log volume relative to the median of matching earlier session slots.

    Matching means the same slot index *and* the same bar duration, so a
    shortened half-day bar is never compared against a full-length one -- on a
    half day the final 30-minute bar has no full-length counterpart and is
    masked. The current observation is excluded from its own denominator.

    The full ``lookback`` of matching observations is required. A shorter window
    would give an unstable median, so the left edge of a panel is masked
    instead; :func:`recommended_panel_bars` exists so that the trimmed context
    is unaffected. A zero or missing volume also yields ``NaN`` and is counted,
    never an infinite feature.
    """
    volumes = panel.column("volume")
    out = np.full(volumes.shape, np.nan, dtype=float)
    masked = 0

    keys = [(bar.slot, round(bar.duration_minutes, 6)) for bar in panel.bars]
    history: dict[tuple[int, float], list[float]] = {}

    for position, key in enumerate(keys):
        prior = history.get(key, [])
        volume = volumes[position]
        usable_prior = [value for value in prior[-lookback:] if value > 0.0]
        if math.isfinite(volume) and volume > 0.0 and len(usable_prior) >= lookback:
            median = float(np.median(usable_prior))
            if median > 0.0:
                out[position] = math.log(volume / median)
            else:
                masked += 1
        else:
            masked += 1
        if math.isfinite(volume):
            history.setdefault(key, []).append(float(volume))
    return out, masked


# --------------------------------------------------------------------------- #
# Daily and forecast decision features
# --------------------------------------------------------------------------- #


def sigma_2d(
    daily: DailyPanel,
    origin_session: dt.date,
    *,
    window: int = 20,
    floor: float = 0.005,
) -> float:
    """Two-session volatility scale used to normalise the return target.

    ``max(sqrt(2) * stdev(last `window` daily log returns), floor)``. This is a
    feature normalisation, not a claim that returns are Gaussian, and the floor
    keeps a quiet stretch from inflating a scaled target.
    """
    history = daily.up_to(origin_session)
    returns = history.log_returns()
    tail = returns[-window:]
    finite = tail[np.isfinite(tail)]
    if finite.size < 2:
        return float("nan")
    # Sample standard deviation, matching the usual realised-volatility estimate.
    daily_sigma = float(np.std(finite, ddof=1))
    if not math.isfinite(daily_sigma):
        return float("nan")
    return max(math.sqrt(2.0) * daily_sigma, floor)


@dataclass(frozen=True)
class DailyFeatures:
    """Features 7-20 for one symbol at one origin, plus validity."""

    symbol: str
    origin_session: dt.date
    values: Mapping[str, float]
    sigma_2d: float
    breadth_missing: bool
    invalid: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.invalid and math.isfinite(self.sigma_2d)


def build_daily_features(
    *,
    symbol: str,
    origin_session: dt.date,
    stock_daily: DailyPanel,
    market_daily: DailyPanel,
    sector_daily: DailyPanel,
    peer_daily: Mapping[str, DailyPanel],
    config_sigma_window: int = 20,
    config_sigma_floor: float = 0.005,
) -> DailyFeatures:
    """Build the fourteen non-forecast features from completed daily sessions.

    ``peer_daily`` holds the eligible frozen-watchlist members of this symbol's
    sector, including the symbol itself. Breadth is watchlist breadth, not
    whole-market breadth, and is masked when fewer than three peers are
    eligible.
    """
    stock = stock_daily.up_to(origin_session)
    market = market_daily.up_to(origin_session)
    sector = sector_daily.up_to(origin_session)

    values: dict[str, float] = {}
    invalid: list[str] = []

    scale = sigma_2d(
        stock_daily, origin_session, window=config_sigma_window, floor=config_sigma_floor
    )
    if not math.isfinite(scale):
        invalid.append("sigma_2d")

    def trailing_return(panel: DailyPanel, sessions: int) -> float:
        closes = panel.column("close")
        if closes.size < sessions + 1:
            return float("nan")
        start, end = closes[-(sessions + 1)], closes[-1]
        if not (math.isfinite(start) and math.isfinite(end)) or start <= 0.0:
            return float("nan")
        return float(end / start - 1.0)

    def realised_vol(panel: DailyPanel, window: int) -> float:
        returns = panel.log_returns()
        tail = returns[-window:]
        finite = tail[np.isfinite(tail)]
        if finite.size < 2:
            return float("nan")
        return float(np.std(finite, ddof=1))

    values["f07_stock_return_1s"] = trailing_return(stock, 1)
    values["f08_stock_return_5s"] = trailing_return(stock, 5)
    values["f09_stock_return_20s"] = trailing_return(stock, 20)

    stock_5 = values["f08_stock_return_5s"]
    sector_5 = trailing_return(sector, 5)
    market_5 = trailing_return(market, 5)
    values["f10_stock_minus_sector_return_5s"] = (
        stock_5 - sector_5 if math.isfinite(stock_5) and math.isfinite(sector_5) else float("nan")
    )

    volumes = stock.column("volume") if "volume" in stock.frame.columns else np.zeros(0)
    if volumes.size >= 21:
        current = volumes[-1]
        preceding = volumes[-21:-1]
        finite = preceding[np.isfinite(preceding) & (preceding > 0.0)]
        median = float(np.median(finite)) if finite.size else float("nan")
        values["f11_volume_ratio_20s"] = (
            float(current / median)
            if math.isfinite(current) and math.isfinite(median) and median > 0.0
            else float("nan")
        )
    else:
        values["f11_volume_ratio_20s"] = float("nan")

    stock_vol_20 = realised_vol(stock, 20)
    stock_vol_60 = realised_vol(stock, 60)
    values["f12_stock_daily_vol_20s"] = stock_vol_20
    values["f13_vol_ratio_20s_60s"] = (
        float(stock_vol_20 / stock_vol_60)
        if math.isfinite(stock_vol_20) and math.isfinite(stock_vol_60) and stock_vol_60 > 0.0
        else float("nan")
    )

    ema20 = _ema(stock.column("close"), span=20)
    last_close = stock.column("close")[-1] if len(stock) else float("nan")
    values["f14_distance_from_ema20_over_vol"] = (
        float((last_close / ema20 - 1.0) / stock_vol_20)
        if math.isfinite(ema20)
        and ema20 > 0.0
        and math.isfinite(last_close)
        and math.isfinite(stock_vol_20)
        and stock_vol_20 > 0.0
        else float("nan")
    )

    highs = stock.column("high") if "high" in stock.frame.columns else np.zeros(0)
    if highs.size >= 20 and math.isfinite(last_close):
        window_high = highs[-20:]
        finite_high = window_high[np.isfinite(window_high)]
        peak = float(np.max(finite_high)) if finite_high.size else float("nan")
        values["f15_drawdown_from_20s_high"] = (
            float(min(0.0, last_close / peak - 1.0))
            if math.isfinite(peak) and peak > 0.0
            else float("nan")
        )
    else:
        values["f15_drawdown_from_20s_high"] = float("nan")

    values["f16_beta_to_market_60s"] = _beta(stock, market, window=60)
    values["f17_market_return_5s"] = market_5
    values["f18_market_vol_20s"] = realised_vol(market, 20)
    values["f19_sector_minus_market_return_5s"] = (
        sector_5 - market_5 if math.isfinite(sector_5) and math.isfinite(market_5) else float("nan")
    )

    breadth, breadth_missing = _sector_breadth(peer_daily, origin_session)
    values[BREADTH_FEATURE_NAME] = breadth

    for name in NON_FORECAST_FEATURE_NAMES:
        if name == BREADTH_FEATURE_NAME:
            continue  # the one optional feature
        if not math.isfinite(values[name]):
            invalid.append(name)

    return DailyFeatures(
        symbol=symbol,
        origin_session=origin_session,
        values=values,
        sigma_2d=scale,
        breadth_missing=breadth_missing,
        invalid=tuple(invalid),
    )


def _ema(values: np.ndarray, *, span: int) -> float:
    """Final value of an exponential moving average with the given span."""
    finite = values[np.isfinite(values)]
    if finite.size < span:
        return float("nan")
    series = pd.Series(finite)
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _beta(stock: DailyPanel, market: DailyPanel, *, window: int) -> float:
    """Trailing daily beta to the market proxy, fitted with an intercept.

    Returns ``NaN`` when the market variance is degenerate, which masks the
    feature rather than producing a meaningless slope.
    """
    joined = _aligned_returns(stock, market)
    if joined is None:
        return float("nan")
    stock_returns, market_returns = joined
    if stock_returns.size < window:
        return float("nan")
    y = stock_returns[-window:]
    x = market_returns[-window:]
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < window // 2:
        return float("nan")
    y, x = y[mask], x[mask]
    variance = float(np.var(x, ddof=1))
    if not math.isfinite(variance) or variance <= 1e-12:
        return float("nan")
    covariance = float(np.cov(y, x, ddof=1)[0, 1])
    return covariance / variance


def _aligned_returns(
    left: DailyPanel, right: DailyPanel
) -> tuple[np.ndarray, np.ndarray] | None:
    """Log returns of two daily panels on their shared sessions."""
    right_sessions = set(right.sessions)
    shared = [session for session in left.sessions if session in right_sessions]
    if len(shared) < 3:
        return None
    left_closes = left.frame.loc[shared, "close"].to_numpy(dtype=float)
    right_closes = right.frame.loc[shared, "close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(left_closes)), np.diff(np.log(right_closes))


def _sector_breadth(
    peer_daily: Mapping[str, DailyPanel], origin_session: dt.date
) -> tuple[float, bool]:
    """Fraction of eligible sector peers trading above their 20-session EMA."""
    above = 0
    counted = 0
    for panel in peer_daily.values():
        history = panel.up_to(origin_session)
        closes = history.column("close") if len(history) else np.zeros(0)
        if closes.size < 21:
            continue
        ema = _ema(closes, span=20)
        last = closes[-1]
        if not (math.isfinite(ema) and math.isfinite(last)) or ema <= 0.0:
            continue
        counted += 1
        if last > ema:
            above += 1
    if counted < MIN_BREADTH_PEERS:
        return float("nan"), True
    return above / counted, False


def forecast_features(
    *,
    quantiles: Mapping[float, Sequence[float]],
    terminal_index_d1: int,
    terminal_index_d2: int,
    scale: float,
    task: ChronosTask,
) -> dict[str, float]:
    """Derive features 1-6 from one forecast.

    ``quantiles`` maps a level to its path over the horizon, in the rebased log
    target units the model was given. Everything is converted into simple
    return units first, because the registered definitions of the asymmetry and
    width features are stated in return units.

    Feature 6 is the D2-minus-D1 median difference. It describes the shape of
    the forecast curve only: the strategy never captures the move between two
    future points, and this value is not a tradable return.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise FeatureError(f"{task.symbol}: volatility scale must be positive and finite")
    for level in (0.10, 0.50, 0.90):
        if level not in quantiles:
            raise FeatureError(f"{task.symbol}: forecast is missing the {level} quantile")

    def value(level: float, index: int) -> float:
        path = quantiles[level]
        if index >= len(path):
            raise FeatureError(
                f"{task.symbol}: terminal index {index} outside a forecast of length "
                f"{len(path)}"
            )
        return task.return_from_target(float(path[index]))

    median_d1 = value(0.50, terminal_index_d1)
    median_d2 = value(0.50, terminal_index_d2)
    p10_d2 = value(0.10, terminal_index_d2)
    p90_d2 = value(0.90, terminal_index_d2)

    width = p90_d2 - p10_d2
    safe_width = max(width, 1e-4)

    return {
        "f01_median_return_d1_over_sigma": median_d1 / scale,
        "f02_median_return_d2_over_sigma": median_d2 / scale,
        "f03_d2_p90_p10_width_over_sigma": width / scale,
        "f04_d2_quantile_asymmetry": (p90_d2 + p10_d2 - 2.0 * median_d2) / safe_width,
        "f05_d2_median_over_width": median_d2 / safe_width,
        "f06_d2_minus_d1_median_over_sigma": (median_d2 - median_d1) / scale,
    }


def assemble_feature_row(
    *,
    daily: DailyFeatures,
    forecast: Mapping[str, float] | None,
) -> dict[str, float]:
    """Combine forecast and daily features into the preprocessing row.

    ``forecast=None`` builds the B0 row, in which the six forecast features are
    absent rather than zero-filled: B0 is a different feature set, not the same
    set with blanks.
    """
    row: dict[str, float] = {}
    if forecast is not None:
        missing = set(FORECAST_FEATURE_NAMES) - set(forecast)
        if missing:
            raise FeatureError(f"forecast features missing: {sorted(missing)}")
        row.update({name: float(forecast[name]) for name in FORECAST_FEATURE_NAMES})
    row.update({name: float(daily.values[name]) for name in NON_FORECAST_FEATURE_NAMES})
    row[BREADTH_MISSING_FLAG] = 1.0 if daily.breadth_missing else 0.0
    return row
