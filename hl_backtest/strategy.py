"""
Signal generation for the funding rate mean-reversion strategy.

Logic:
  - Compute rolling cumulative funding over 1h, 8h, 24h windows
  - Compute 30-day trailing percentile rank of 8h cumulative funding
  - LONG signal: percentile < 5th
  - SHORT signal: percentile > 95th
  - EXIT: percentile returns to 40-60, OR time stop 24h, OR SL -2%, TP +1.5%

All signals are computed on candle *close*, executed on *next* open (no look-ahead).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default strategy parameters (tuneable)
DEFAULT_PARAMS = {
    "entry_long_pct": 5.0,
    "entry_short_pct": 95.0,
    "exit_mean_low": 40.0,
    "exit_mean_high": 60.0,
    "funding_window_h": 8,
    "percentile_lookback_days": 30,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 1.5,
    "max_hold_hours": 24,
    "vol_lookback_h": 168,
    "bar_interval_h": 4,          # candle bar size in hours (1 or 4)
}


@dataclass
class StrategyParams:
    entry_long_pct: float = 5.0
    entry_short_pct: float = 95.0
    exit_mean_low: float = 40.0
    exit_mean_high: float = 60.0
    funding_window_h: int = 8
    percentile_lookback_days: int = 30
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 1.5
    max_hold_hours: int = 24
    vol_lookback_h: int = 168
    bar_interval_h: int = 4       # 4 for 4h candles, 1 for 1h candles


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    For each point, compute the percentile rank of that value within the
    trailing *window* observations (excluding current point → no look-ahead).
    Uses a rolling apply — reasonably fast for ~10k rows per asset.
    """
    def _rank(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 50.0
        val = arr[-1]
        pct = float(np.sum(arr[:-1] <= val) / (len(arr) - 1) * 100)
        return pct

    return series.rolling(window=window, min_periods=max(10, window // 4)).apply(_rank, raw=True)


def compute_signals(
    candles: pd.DataFrame,
    funding: pd.DataFrame,
    params: StrategyParams | None = None,
) -> pd.DataFrame:
    """
    Merge candle and funding data, compute all features and entry/exit signals.

    Works in native bar frequency (bar_interval_h=4 → 4h bars, =1 → 1h bars).
    All hour-based parameters are converted to bar counts automatically.

    Returns a DataFrame with columns:
        open, high, low, close, volume,
        funding_per_bar (sum of hourly rates within the bar),
        funding_8h, funding_24h,
        pct_rank_8h,
        signal_long, signal_short,
        realized_vol,
        next_open
    """
    if params is None:
        params = StrategyParams()

    interval = params.bar_interval_h
    freq = f"{interval}h"
    bars_per_day = 24 // interval

    # Resample candles to target interval
    candles_resampled = candles.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    # Aggregate hourly funding to bar frequency (sum over the bar)
    funding_per_bar = funding["fundingRate"].resample(freq).sum()
    funding_aligned = funding_per_bar.reindex(candles_resampled.index, method="ffill").fillna(0.0)

    df = candles_resampled.copy()
    df["funding_per_bar"] = funding_aligned

    # Cumulative funding in bars (funding_window_h / bar_interval_h = number of bars)
    w_bars = max(1, params.funding_window_h // interval)
    df["funding_8h"] = df["funding_per_bar"].rolling(w_bars, min_periods=1).sum()
    df["funding_24h"] = df["funding_per_bar"].rolling(max(1, 24 // interval), min_periods=1).sum()

    # Trailing percentile rank: 30-day lookback in bars
    lookback_bars = params.percentile_lookback_days * bars_per_day
    df["pct_rank_8h"] = _rolling_percentile_rank(df["funding_8h"], window=lookback_bars)

    # Raw threshold crossings
    in_long_zone = df["pct_rank_8h"] < params.entry_long_pct
    in_short_zone = df["pct_rank_8h"] > params.entry_short_pct

    # Only fire on fresh crossings: condition is True now AND was False one bar ago
    # This prevents re-entering on a persistent extreme funding regime
    df["signal_long"] = in_long_zone & (~in_long_zone.shift(1).fillna(False))
    df["signal_short"] = in_short_zone & (~in_short_zone.shift(1).fillna(False))

    df["exit_mean_rev"] = (df["pct_rank_8h"] >= params.exit_mean_low) & (
        df["pct_rank_8h"] <= params.exit_mean_high
    )

    # Realized vol: annualized from bar-frequency log returns
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_lookback_bars = max(24, params.vol_lookback_h // interval)
    df["realized_vol"] = log_ret.rolling(vol_lookback_bars, min_periods=bars_per_day).std() * np.sqrt(8760 // interval * interval)

    # Next open (execution price) — computed on prior bar close, fills at next open
    df["next_open"] = df["open"].shift(-1)

    return df


def build_signals_all(
    candles_map: dict[str, pd.DataFrame],
    funding_map: dict[str, pd.DataFrame],
    params: StrategyParams | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute signals for all coins in the universe."""
    signals: dict[str, pd.DataFrame] = {}
    for coin in candles_map:
        if coin not in funding_map:
            logger.warning("No funding data for %s — skipping", coin)
            continue
        try:
            sig = compute_signals(candles_map[coin], funding_map[coin], params)
            if len(sig) > 100:
                signals[coin] = sig
        except Exception as exc:
            logger.error("Signal computation failed for %s: %s", coin, exc)
    logger.info("Signals computed for %d assets", len(signals))
    return signals
