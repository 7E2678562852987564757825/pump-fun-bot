"""
Event-driven backtester for the funding rate mean-reversion strategy.

Key design decisions:
- Signals computed on bar CLOSE, fills at NEXT bar OPEN → no look-ahead
- Funding payments applied each hour on open positions at historical rate
- Costs: taker fee 0.035%, slippage 0.05% per side (both entry and exit)
- Maker rebate -0.001% (not assumed; we always model taker)
- Position sizing: 1% portfolio risk / stop distance (vol-targeted)
- Max 5 concurrent positions; no pyramiding
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from strategy import StrategyParams
from universe import get_universe_for_date, UniverseSnapshot

logger = logging.getLogger(__name__)

# Cost model
TAKER_FEE = 0.00035   # 0.035% per side
SLIPPAGE = 0.0005     # 0.05% per side
TOTAL_COST_PER_SIDE = TAKER_FEE + SLIPPAGE  # 0.085% per side

MAX_POSITIONS = 5
RISK_PER_TRADE = 0.01  # 1% of equity per trade


@dataclass
class Position:
    coin: str
    direction: Literal["long", "short"]
    entry_price: float
    entry_time: pd.Timestamp
    size_usd: float          # notional in USD
    stop_loss_price: float
    take_profit_price: float
    max_hold_bars: int       # number of 1h bars after entry
    bars_held: int = 0
    realized_pnl: float = 0.0
    funding_pnl: float = 0.0
    entry_fee: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.funding_pnl - self.entry_fee


@dataclass
class Trade:
    coin: str
    direction: Literal["long", "short"]
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size_usd: float
    gross_pnl: float
    fees: float          # total fees (entry + exit)
    funding_pnl: float
    net_pnl: float
    exit_reason: str
    hold_hours: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series          # DatetimeIndex → equity
    trades: list[Trade]
    daily_returns: pd.Series
    positions_over_time: pd.DataFrame  # n_positions at each bar


def _fill_price(price: float, direction: Literal["long", "short"]) -> float:
    """Apply slippage to a fill price."""
    if direction == "long":
        return price * (1 + SLIPPAGE)
    else:
        return price * (1 - SLIPPAGE)


def _exit_fill_price(price: float, direction: Literal["long", "short"]) -> float:
    """Apply slippage to exit fill (reverse direction)."""
    if direction == "long":
        return price * (1 - SLIPPAGE)
    else:
        return price * (1 + SLIPPAGE)


def _position_pnl(pos: Position, exit_price: float) -> float:
    """Raw price PnL (before fees) on closing a position."""
    if pos.direction == "long":
        return (exit_price - pos.entry_price) / pos.entry_price * pos.size_usd
    else:
        return (pos.entry_price - exit_price) / pos.entry_price * pos.size_usd


def _compute_size_usd(
    equity: float,
    entry_price: float,
    stop_price: float,
    params: StrategyParams,
) -> float:
    """
    Vol-targeted sizing: risk 1% of equity on the stop distance.
    size_usd = (risk_amount) / (stop_distance_pct)
    Capped at 20% of equity per position.
    """
    stop_dist_pct = abs(entry_price - stop_price) / entry_price
    if stop_dist_pct < 1e-6:
        return 0.0
    risk_amount = equity * RISK_PER_TRADE
    raw_size = risk_amount / stop_dist_pct
    # Cap at 20% of equity
    return min(raw_size, equity * 0.20)


def _infer_bar_interval(signals_map: dict[str, pd.DataFrame]) -> pd.Timedelta:
    """Infer bar size from the first signals DataFrame's index spacing."""
    for df in signals_map.values():
        if len(df) >= 2:
            diffs = df.index.to_series().diff().dropna()
            if not diffs.empty:
                return diffs.mode().iloc[0]
    return pd.Timedelta(hours=1)


def run_backtest(
    signals_map: dict[str, pd.DataFrame],
    funding_map: dict[str, pd.DataFrame],
    universe_snapshots: list[UniverseSnapshot],
    params: StrategyParams,
    initial_equity: float = 100_000.0,
    label: str = "backtest",
) -> BacktestResult:
    """
    Run the event-driven backtest.

    Iterates over a unified hourly timeline. At each bar:
    1. Update open positions (check SL/TP/time stop, accrue funding)
    2. Check mean-reversion exits
    3. Open new positions for coins with fresh signals
    """
    logger.info("[%s] Starting backtest, initial equity=%.0f", label, initial_equity)

    # Build unified hourly timeline
    all_times: set[pd.Timestamp] = set()
    for df in signals_map.values():
        all_times.update(df.index.tolist())
    timeline = sorted(all_times)

    if not timeline:
        logger.error("No data — empty timeline")
        return BacktestResult(
            equity_curve=pd.Series(dtype=float),
            trades=[],
            daily_returns=pd.Series(dtype=float),
            positions_over_time=pd.DataFrame(),
        )

    bar_interval = _infer_bar_interval(signals_map)
    logger.debug("[%s] Inferred bar interval: %s", label, bar_interval)

    cooldown_bars = getattr(params, "cooldown_bars", 3)
    cooldown_until: dict[str, pd.Timestamp] = {}

    equity = initial_equity
    open_positions: dict[str, Position] = {}  # coin → Position
    trades: list[Trade] = []
    equity_by_time: list[tuple[pd.Timestamp, float]] = []
    n_positions_by_time: list[tuple[pd.Timestamp, int]] = []

    for bar_time in timeline:
        # ---------------------------------------------------------------
        # 1. Process open positions
        # ---------------------------------------------------------------
        closed_this_bar: list[str] = []
        for coin, pos in list(open_positions.items()):
            if coin not in signals_map:
                continue
            sig_df = signals_map[coin]
            if bar_time not in sig_df.index:
                continue

            row = sig_df.loc[bar_time]
            current_price = row["close"]
            if pd.isna(current_price):
                continue

            pos.bars_held += 1
            exit_reason: str | None = None
            exit_price = current_price  # default: close price

            # Check stop loss and take profit using high/low of the bar
            bar_high = row.get("high", current_price)
            bar_low = row.get("low", current_price)

            if pos.direction == "long":
                if bar_low <= pos.stop_loss_price:
                    exit_price = pos.stop_loss_price
                    exit_reason = "stop_loss"
                elif bar_high >= pos.take_profit_price:
                    exit_price = pos.take_profit_price
                    exit_reason = "take_profit"
            else:  # short
                if bar_high >= pos.stop_loss_price:
                    exit_price = pos.stop_loss_price
                    exit_reason = "stop_loss"
                elif bar_low <= pos.take_profit_price:
                    exit_price = pos.take_profit_price
                    exit_reason = "take_profit"

            # Time stop
            if exit_reason is None and pos.bars_held >= pos.max_hold_bars:
                exit_reason = "time_stop"

            # Mean reversion exit (use close)
            if exit_reason is None and row.get("exit_mean_rev", False):
                exit_reason = "mean_reversion"

            # Apply funding payment for this bar.
            # funding_per_bar is already aggregated to bar frequency in signals.
            funding_rate = 0.0
            if "funding_per_bar" in row.index:
                funding_rate = float(row.get("funding_per_bar", 0.0))
            elif coin in funding_map:
                fd = funding_map[coin]
                if bar_time in fd.index:
                    funding_rate = float(fd.loc[bar_time, "fundingRate"])
            # Long pays positive funding, short receives it
            if pos.direction == "long":
                pos.funding_pnl -= funding_rate * pos.size_usd
            else:
                pos.funding_pnl += funding_rate * pos.size_usd

            if exit_reason is not None:
                # Close position
                actual_exit = _exit_fill_price(exit_price, pos.direction)
                gross_pnl = _position_pnl(pos, actual_exit)
                exit_fee = pos.size_usd * TAKER_FEE
                net_pnl = gross_pnl + pos.funding_pnl - pos.entry_fee - exit_fee

                equity += net_pnl

                trades.append(Trade(
                    coin=coin,
                    direction=pos.direction,
                    entry_time=pos.entry_time,
                    exit_time=bar_time,
                    entry_price=pos.entry_price,
                    exit_price=actual_exit,
                    size_usd=pos.size_usd,
                    gross_pnl=gross_pnl,
                    fees=pos.entry_fee + exit_fee,
                    funding_pnl=pos.funding_pnl,
                    net_pnl=net_pnl,
                    exit_reason=exit_reason,
                    hold_hours=pos.bars_held,
                ))
                closed_this_bar.append(coin)
                cooldown_until[coin] = bar_time + bar_interval * cooldown_bars
                logger.debug(
                    "[%s] %s %s EXIT %s @ %.4f, pnl=%.2f",
                    bar_time, coin, pos.direction, exit_reason, actual_exit, net_pnl,
                )

        for coin in closed_this_bar:
            del open_positions[coin]

        # ---------------------------------------------------------------
        # 2. Open new positions (if capacity available)
        # ---------------------------------------------------------------
        # Get current universe
        current_universe = get_universe_for_date(bar_time, universe_snapshots)

        if len(open_positions) < MAX_POSITIONS:
            # Collect candidate signals from coins in current universe
            candidates: list[tuple[str, Literal["long", "short"], pd.Series]] = []
            for coin in current_universe:
                if coin in open_positions:
                    continue
                if coin not in signals_map:
                    continue
                # Respect cooldown after last exit
                if cooldown_until.get(coin, pd.Timestamp.min.tz_localize("UTC")) > bar_time:
                    continue
                sig_df = signals_map[coin]
                # Signal computed on prior bar close; entry fills at current bar's open
                prev_bar = bar_time - bar_interval
                if prev_bar not in sig_df.index:
                    continue
                prev_row = sig_df.loc[prev_bar]
                # Entry price is the next_open from prev_bar = current bar's open
                entry_price_raw = prev_row.get("next_open", float("nan"))
                if pd.isna(entry_price_raw) or entry_price_raw <= 0:
                    continue
                if prev_row.get("signal_long", False):
                    candidates.append((coin, "long", prev_row))
                elif prev_row.get("signal_short", False):
                    candidates.append((coin, "short", prev_row))

            # Prioritize by distance from 50th percentile (more extreme = higher priority)
            candidates.sort(key=lambda x: abs(x[2].get("pct_rank_8h", 50) - 50), reverse=True)

            for coin, direction, prev_row in candidates:
                if len(open_positions) >= MAX_POSITIONS:
                    break
                entry_price_raw = prev_row["next_open"]
                entry_fill = _fill_price(entry_price_raw, direction)

                # Stop / TP prices
                sl_pct = params.stop_loss_pct / 100.0
                tp_pct = params.take_profit_pct / 100.0
                if direction == "long":
                    stop_price = entry_fill * (1 - sl_pct)
                    tp_price = entry_fill * (1 + tp_pct)
                else:
                    stop_price = entry_fill * (1 + sl_pct)
                    tp_price = entry_fill * (1 - tp_pct)

                size_usd = _compute_size_usd(equity, entry_fill, stop_price, params)
                if size_usd < 10:
                    continue

                entry_fee = size_usd * TAKER_FEE
                equity -= entry_fee  # deduct entry fee immediately

                pos = Position(
                    coin=coin,
                    direction=direction,
                    entry_price=entry_fill,
                    entry_time=bar_time,
                    size_usd=size_usd,
                    stop_loss_price=stop_price,
                    take_profit_price=tp_price,
                    max_hold_bars=max(1, params.max_hold_hours // params.bar_interval_h),
                    entry_fee=entry_fee,
                )
                open_positions[coin] = pos
                logger.debug(
                    "[%s] %s %s ENTRY @ %.4f size=%.0f",
                    bar_time, coin, direction, entry_fill, size_usd,
                )

        equity_by_time.append((bar_time, equity))
        n_positions_by_time.append((bar_time, len(open_positions)))

    # Close any remaining open positions at last available price
    last_time = timeline[-1] if timeline else None
    for coin, pos in list(open_positions.items()):
        if last_time is None:
            break
        sig_df = signals_map.get(coin)
        if sig_df is None or last_time not in sig_df.index:
            continue
        last_price = sig_df.loc[last_time, "close"]
        if pd.isna(last_price):
            continue
        actual_exit = _exit_fill_price(last_price, pos.direction)
        gross_pnl = _position_pnl(pos, actual_exit)
        exit_fee = pos.size_usd * TAKER_FEE
        net_pnl = gross_pnl + pos.funding_pnl - pos.entry_fee - exit_fee
        equity += net_pnl
        trades.append(Trade(
            coin=coin,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=last_time,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            size_usd=pos.size_usd,
            gross_pnl=gross_pnl,
            fees=pos.entry_fee + exit_fee,
            funding_pnl=pos.funding_pnl,
            net_pnl=net_pnl,
            exit_reason="end_of_data",
            hold_hours=pos.bars_held,
        ))

    equity_curve = pd.Series(
        dict(equity_by_time),
        name="equity",
    )
    equity_curve.index = pd.DatetimeIndex(equity_curve.index)

    positions_ts = pd.Series(
        {t: n for t, n in n_positions_by_time},
        name="n_positions",
    )

    daily_equity = equity_curve.resample("1D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()

    logger.info(
        "[%s] Done: %d trades, final equity=%.2f (%.1f%% return)",
        label,
        len(trades),
        equity,
        (equity / initial_equity - 1) * 100,
    )

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        daily_returns=daily_returns,
        positions_over_time=positions_ts.to_frame(),
    )
