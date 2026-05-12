"""
Main entry point for the Hyperliquid funding-rate mean-reversion backtest.

Usage:
    python main.py                          # run full backtest with defaults
    python main.py --show-universe-only     # fetch data and print universe, then stop
    python main.py --entry-long 10 --entry-short 90
    python main.py --force-refresh          # ignore cache and re-fetch from API
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Set up logging before any imports that might log
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "backtest.log"),
    ],
)
logger = logging.getLogger(__name__)

import numpy as np

from data_loader import get_meta, get_candles_chunked, get_funding_chunked, get_asset_contexts
from universe import get_all_alt_perps, build_monthly_universes, print_universe_summary
from strategy import StrategyParams, build_signals_all
from backtest import run_backtest
from analytics import (
    compute_metrics,
    per_asset_breakdown,
    monthly_returns,
    bootstrap_sharpe_ci,
    walk_forward_analysis,
    plot_equity_curve,
    plot_monthly_heatmap,
    plot_walk_forward,
    print_metrics_table,
)

SEED = 42
np.random.seed(SEED)

# HL API candle availability (empirically determined):
#   1h candles: Oct 2025 onwards (~7 months)
#   4h candles: Jan 2024 onwards (~28 months) — used for signals/backtest
#   1d candles: 2022 onwards (~4 years) — used for universe volume selection
DAILY_START_MS = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
SIGNAL_START_MS = int(datetime(2024, 1, 15, tzinfo=timezone.utc).timestamp() * 1000)  # 4h candle start
DATA_END_MS = int(datetime.now(timezone.utc).timestamp() * 1000)
SIGNAL_INTERVAL = "4h"
BAR_INTERVAL_H = 4

OUTPUT_DIR = Path(__file__).parent / "output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HL Funding Rate Mean-Reversion Backtest")
    p.add_argument("--show-universe-only", action="store_true",
                   help="Fetch data, print universe selection, then exit")
    p.add_argument("--force-refresh", action="store_true",
                   help="Ignore disk cache and re-fetch all data from API")
    p.add_argument("--entry-long", type=float, default=5.0,
                   help="Percentile threshold for long entry (default: 5)")
    p.add_argument("--entry-short", type=float, default=95.0,
                   help="Percentile threshold for short entry (default: 95)")
    p.add_argument("--stop-loss", type=float, default=2.0,
                   help="Stop loss %% (default: 2.0)")
    p.add_argument("--take-profit", type=float, default=4.0,
                   help="Take profit %% (default: 4.0)")
    p.add_argument("--min-abs-funding", type=float, default=0.0003,
                   help="Min |8h cumulative funding| to trigger signal (default: 0.0003)")
    p.add_argument("--max-hold-hours", type=int, default=48,
                   help="Maximum hold duration in hours (default: 48)")
    p.add_argument("--direction", type=str, default="both",
                   choices=["both", "long_only", "short_only"],
                   help="Trade direction filter (default: both)")
    p.add_argument("--funding-window", type=int, default=8,
                   help="Cumulative funding window in hours (default: 8)")
    p.add_argument("--max-coins", type=int, default=None,
                   help="Limit coins fetched (useful for quick test runs)")
    p.add_argument("--initial-equity", type=float, default=100_000.0)
    p.add_argument("--train-fraction", type=float, default=0.70,
                   help="Fraction of data for in-sample (default: 0.70)")
    p.add_argument("--sensitivity", action="store_true",
                   help="Run sensitivity analysis across threshold pairs")
    return p.parse_args()


def fetch_all_data(
    coins: list[str],
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Fetch data for all coins.
    Returns (daily_candles_map, signal_candles_map, funding_map).
      - daily_candles_map: 1d OHLCV from 2022, used for universe volume selection
      - signal_candles_map: 4h OHLCV from Jan 2024, used for signals/backtest
      - funding_map: hourly funding from Jan 2024
    """
    daily_candles_map: dict[str, pd.DataFrame] = {}
    signal_candles_map: dict[str, pd.DataFrame] = {}
    funding_map: dict[str, pd.DataFrame] = {}
    total = len(coins)
    for i, coin in enumerate(coins, 1):
        logger.info("Fetching %s (%d/%d)", coin, i, total)
        try:
            # 1d candles for universe (long history, few API calls)
            daily = get_candles_chunked(
                coin, "1d", DAILY_START_MS, DATA_END_MS,
                chunk_days=365, force_refresh=force_refresh,
            )
            if not daily.empty and len(daily) > 30:
                daily_candles_map[coin] = daily

            # 4h candles for signal/backtest
            signal = get_candles_chunked(
                coin, SIGNAL_INTERVAL, SIGNAL_START_MS, DATA_END_MS,
                chunk_days=90, force_refresh=force_refresh,
            )
            if not signal.empty and len(signal) > 200:
                signal_candles_map[coin] = signal
            else:
                logger.warning("%s: insufficient signal candle data (%d rows), skipping", coin, len(signal))
                continue

            # Funding history aligned with signal period
            funding = get_funding_chunked(
                coin, SIGNAL_START_MS, DATA_END_MS,
                chunk_days=90, force_refresh=force_refresh,
            )
            if not funding.empty:
                funding_map[coin] = funding
            else:
                logger.warning("%s: no funding data", coin)
        except Exception as exc:
            logger.error("Error fetching %s: %s", coin, exc)

    logger.info(
        "Data loaded: %d daily, %d signal candles, %d with funding",
        len(daily_candles_map), len(signal_candles_map), len(funding_map),
    )
    return daily_candles_map, signal_candles_map, funding_map


def build_bnh_curve(
    candles_map: dict[str, pd.DataFrame],
    coins: list[str],
    initial_equity: float,
) -> pd.Series:
    """Equal-weight buy-and-hold benchmark."""
    valid = [c for c in coins if c in candles_map]
    if not valid:
        return pd.Series(dtype=float)
    returns = []
    for coin in valid:
        df = candles_map[coin]["close"].resample("1D").last().ffill().pct_change()
        returns.append(df)
    eq_ret = pd.concat(returns, axis=1).mean(axis=1).fillna(0)
    eq_curve = (1 + eq_ret).cumprod() * initial_equity
    return eq_curve


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Get all alt perp names, pre-filtered by current 24h volume
    # -----------------------------------------------------------------------
    logger.info("=== Step 1: Fetching available perps ===")
    all_coins = get_all_alt_perps()
    logger.info("Total alt perps in HL universe: %d", len(all_coins))

    # Use current asset contexts to pre-select by 24h volume; avoids fetching
    # history for dead/illiquid coins that will never appear in the top-30 universe.
    try:
        meta = get_meta()
        contexts = get_asset_contexts()
        if contexts:
            coin_names = [m["name"] for m in meta]
            vol_map: dict[str, float] = {}
            for i, ctx in enumerate(contexts):
                if i < len(coin_names):
                    coin = coin_names[i]
                    if coin not in {"BTC", "ETH"}:
                        try:
                            vol_map[coin] = float(ctx.get("dayNtlVlm", 0))
                        except (ValueError, TypeError):
                            pass
            # Keep top N*2 by current 24h volume so we have headroom for monthly changes
            n_prefetch = (args.max_coins or 80) * 2
            sorted_by_vol = sorted(vol_map, key=lambda c: -vol_map[c])
            prefetch_coins = sorted_by_vol[:n_prefetch]
            logger.info("Pre-filtered to %d coins by current 24h volume", len(prefetch_coins))
            all_coins = [c for c in all_coins if c in set(prefetch_coins)]
    except Exception as exc:
        logger.warning("Volume pre-filter failed (%s) — using all coins", exc)

    if args.max_coins:
        all_coins = all_coins[: args.max_coins]
        logger.info("Further limited to %d coins by --max-coins", args.max_coins)

    # -----------------------------------------------------------------------
    # Step 2: Fetch data
    # -----------------------------------------------------------------------
    logger.info("=== Step 2: Fetching market data (%d coins) ===", len(all_coins))
    daily_candles_map, signal_candles_map, funding_map = fetch_all_data(
        all_coins, force_refresh=args.force_refresh
    )

    if not signal_candles_map:
        logger.error("No signal candle data fetched — cannot continue")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Universe selection (uses daily candles for long volume history)
    # -----------------------------------------------------------------------
    logger.info("=== Step 3: Building monthly universes ===")
    # Start from 2022-02-01 (first month after daily candle start + 30d lookback)
    start_dt = datetime(2022, 2, 1, tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc)

    universe_candles = daily_candles_map if daily_candles_map else signal_candles_map
    universe_snapshots = build_monthly_universes(start_dt, end_dt, universe_candles)
    print_universe_summary(universe_snapshots)

    if args.show_universe_only:
        logger.info("--show-universe-only flag set — stopping after universe display")
        return

    # -----------------------------------------------------------------------
    # Step 4: Compute signals (on 4h candles)
    # -----------------------------------------------------------------------
    logger.info("=== Step 4: Computing signals (bar_interval=%dh) ===", BAR_INTERVAL_H)
    params = StrategyParams(
        entry_long_pct=args.entry_long,
        entry_short_pct=args.entry_short,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_hold_hours=args.max_hold_hours,
        funding_window_h=args.funding_window,
        min_abs_funding_8h=args.min_abs_funding,
        direction=args.direction,
        bar_interval_h=BAR_INTERVAL_H,
    )

    # Only compute signals for coins that appear in any universe snapshot
    universe_coins: set[str] = set()
    for snap in universe_snapshots:
        universe_coins.update(snap.coins)

    filtered_signal = {k: v for k, v in signal_candles_map.items() if k in universe_coins}
    filtered_funding = {k: v for k, v in funding_map.items() if k in universe_coins}

    signals_map = build_signals_all(filtered_signal, filtered_funding, params)

    if not signals_map:
        logger.error("No signals computed — check data")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 5: Train/test split
    # -----------------------------------------------------------------------
    logger.info("=== Step 5: Train/test split (%.0f%%/%.0f%%) ===",
                args.train_fraction * 100, (1 - args.train_fraction) * 100)

    # Find overall timeline
    all_times_set: set[pd.Timestamp] = set()
    for df in signals_map.values():
        all_times_set.update(df.index)
    all_times = sorted(all_times_set)
    split_idx = int(len(all_times) * args.train_fraction)
    split_time = all_times[split_idx]
    logger.info("IS period: %s → %s", all_times[0], split_time)
    logger.info("OOS period: %s → %s", split_time, all_times[-1])

    def split_signals(signals: dict[str, pd.DataFrame], end: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return {k: v.loc[v.index <= end] for k, v in signals.items() if not v.loc[v.index <= end].empty}

    def split_signals_from(signals: dict[str, pd.DataFrame], start: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return {k: v.loc[v.index >= start] for k, v in signals.items() if not v.loc[v.index >= start].empty}

    def split_funding(funding: dict[str, pd.DataFrame], lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> dict[str, pd.DataFrame]:
        result = {}
        for k, v in funding.items():
            mask = pd.Series(True, index=v.index)
            if lo is not None:
                mask &= v.index >= lo
            if hi is not None:
                mask &= v.index <= hi
            sliced = v[mask]
            if not sliced.empty:
                result[k] = sliced
        return result

    is_snapshots = [s for s in universe_snapshots if pd.Timestamp(s.date) <= split_time]
    oos_snapshots = [s for s in universe_snapshots if pd.Timestamp(s.date) > split_time]

    is_signals = split_signals(signals_map, split_time)
    oos_signals = split_signals_from(signals_map, split_time)

    is_funding = split_funding(filtered_funding, None, split_time)
    oos_funding = split_funding(filtered_funding, split_time, None)

    # -----------------------------------------------------------------------
    # Step 6: Run backtests
    # -----------------------------------------------------------------------
    logger.info("=== Step 6: Running in-sample backtest ===")
    is_result = run_backtest(
        is_signals, is_funding, is_snapshots, params,
        initial_equity=args.initial_equity, label="IS",
    )

    logger.info("=== Step 6b: Running out-of-sample backtest ===")
    oos_initial = is_result.equity_curve.iloc[-1] if not is_result.equity_curve.empty else args.initial_equity
    oos_result = run_backtest(
        oos_signals, oos_funding, oos_snapshots, params,
        initial_equity=float(oos_initial), label="OOS",
    )

    # -----------------------------------------------------------------------
    # Step 7: Metrics
    # -----------------------------------------------------------------------
    logger.info("=== Step 7: Computing metrics ===")
    is_metrics = compute_metrics(is_result, initial_equity=args.initial_equity, label="In-Sample")
    oos_metrics = compute_metrics(oos_result, initial_equity=float(oos_initial), label="Out-of-Sample")

    print_metrics_table(is_metrics, "IN-SAMPLE RESULTS")
    print_metrics_table(oos_metrics, "OUT-OF-SAMPLE RESULTS")

    # Per-asset breakdown
    print("\n--- IN-SAMPLE Per-Asset Breakdown (top 10) ---")
    is_asset_df = per_asset_breakdown(is_result.trades)
    if not is_asset_df.empty:
        print(is_asset_df.head(10).to_string(index=False))

    print("\n--- OUT-OF-SAMPLE Per-Asset Breakdown (top 10) ---")
    oos_asset_df = per_asset_breakdown(oos_result.trades)
    if not oos_asset_df.empty:
        print(oos_asset_df.head(10).to_string(index=False))

    # Bootstrap CI
    logger.info("=== Robustness: Bootstrap Sharpe CI ===")
    for label, result in [("IS", is_result), ("OOS", oos_result)]:
        lo, pt, hi = bootstrap_sharpe_ci(result.daily_returns)
        print(f"\nBootstrap Sharpe 95% CI [{label}]: ({lo:.3f}, {pt:.3f}, {hi:.3f})")

    # Walk-forward
    logger.info("=== Robustness: Walk-forward analysis ===")
    wf_df = walk_forward_analysis(is_result, args.initial_equity)
    if not wf_df.empty:
        print(f"\nWalk-forward ({len(wf_df)} windows, IS only):")
        print(f"  Sharpe: mean={wf_df['sharpe'].mean():.3f}, "
              f"min={wf_df['sharpe'].min():.3f}, max={wf_df['sharpe'].max():.3f}")
        print(f"  Positive windows: {(wf_df['sharpe'] > 0).sum()}/{len(wf_df)}")

    # -----------------------------------------------------------------------
    # Step 8: Plots
    # -----------------------------------------------------------------------
    logger.info("=== Step 8: Generating plots ===")

    all_universe_coins = list(universe_coins & set(signal_candles_map.keys()))
    bnh_curve = build_bnh_curve(signal_candles_map, all_universe_coins, args.initial_equity)

    plot_equity_curve(
        is_result, oos_result, bnh_curve,
        output_path=OUTPUT_DIR / "equity_curve.png",
        initial_equity=args.initial_equity,
    )

    is_monthly = monthly_returns(is_result.equity_curve)
    oos_monthly = monthly_returns(oos_result.equity_curve)
    plot_monthly_heatmap(is_monthly, oos_monthly, OUTPUT_DIR / "monthly_heatmap.png")

    if not wf_df.empty:
        plot_walk_forward(wf_df, OUTPUT_DIR / "walk_forward.png")

    # -----------------------------------------------------------------------
    # Step 9: Sensitivity (optional)
    # -----------------------------------------------------------------------
    if args.sensitivity:
        logger.info("=== Step 9: Sensitivity analysis ===")
        threshold_pairs = [(5, 95), (10, 90), (1, 99)]
        sens_rows = []
        for lo_pct, hi_pct in threshold_pairs:
            s_params = StrategyParams(
                entry_long_pct=lo_pct,
                entry_short_pct=hi_pct,
                stop_loss_pct=args.stop_loss,
                take_profit_pct=args.take_profit,
                funding_window_h=args.funding_window,
                bar_interval_h=BAR_INTERVAL_H,
            )
            s_signals = build_signals_all(filtered_signal, filtered_funding, s_params)
            s_is_signals = split_signals(s_signals, split_time)
            s_result = run_backtest(
                s_is_signals, is_funding, is_snapshots, s_params,
                initial_equity=args.initial_equity, label=f"sens_{lo_pct}/{hi_pct}",
            )
            m = compute_metrics(s_result, initial_equity=args.initial_equity)
            m["thresholds"] = f"{lo_pct}/{hi_pct}"
            sens_rows.append(m)
        if sens_rows:
            print("\n--- SENSITIVITY ANALYSIS (IS only) ---")
            sens_df = pd.DataFrame(sens_rows)[["thresholds", "total_return_pct", "sharpe", "max_drawdown_pct", "n_trades", "win_rate_pct"]]
            print(sens_df.to_string(index=False))

    logger.info("=== All done. Outputs in %s ===", OUTPUT_DIR)
    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
