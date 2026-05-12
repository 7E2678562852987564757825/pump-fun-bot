"""
Grid search over strategy parameters using cached data.
IS: 2024-01-15 → 2024-10-31 (70%)
OOS: 2024-11-01 → 2025-04-30 (30%)
"""
from __future__ import annotations

import logging
import sys
import os
import re
from pathlib import Path
from dataclasses import asdict

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))

from universe import build_monthly_universes
from strategy import StrategyParams, build_signals_all
from backtest import run_backtest
from analytics import compute_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "data" / "cache"

IS_END  = pd.Timestamp("2024-10-31", tz="UTC")
OOS_END = pd.Timestamp("2025-04-30", tz="UTC")
SIGNAL_START = pd.Timestamp("2024-01-15", tz="UTC")
DAILY_START  = pd.Timestamp("2022-01-01", tz="UTC")

INITIAL_EQUITY = 100_000.0
BAR_H = 4


# ── Load merged parquets from cache ──────────────────────────────────────────

def _load_best_parquet(pattern: str) -> dict[str, pd.DataFrame]:
    """
    Scan CACHE_DIR for files matching `pattern` (regex with one capture group = coin).
    For coins with multiple files, pick the one with the most rows.
    Returns {coin: DataFrame}.
    """
    result: dict[str, list[pd.DataFrame]] = {}
    for f in CACHE_DIR.iterdir():
        m = re.match(pattern, f.name)
        if m:
            coin = m.group(1)
            try:
                df = pq.read_table(f).to_pandas()
                if not df.empty:
                    result.setdefault(coin, []).append(df)
            except Exception:
                pass
    # Pick largest file per coin, ensure DatetimeIndex
    out = {}
    for coin, dfs in result.items():
        df = max(dfs, key=len)
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to set index from a 'time' column
            if "time" in df.columns:
                df = df.set_index("time")
            elif "__index_level_0__" in df.columns:
                df = df.set_index("__index_level_0__")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        out[coin] = df.sort_index()
    return out


def load_all_data():
    print("Loading cached data from disk…")
    # Pattern: {hash}_{COIN}_4h_merged.parquet
    signal_candles = _load_best_parquet(r"[0-9a-f]+_([A-Za-z0-9k]+)_4h_merged\.parquet")
    funding_map    = _load_best_parquet(r"[0-9a-f]+_([A-Za-z0-9k]+)_funding_merged\.parquet")
    daily_candles  = _load_best_parquet(r"[0-9a-f]+_([A-Za-z0-9k]+)_1d_merged\.parquet")

    # Filter: signal candles must overlap with backtest period
    signal_candles = {
        c: df for c, df in signal_candles.items()
        if df.index.max() >= SIGNAL_START and len(df) >= 100
    }
    funding_map = {
        c: df for c, df in funding_map.items()
        if c in signal_candles and len(df) >= 50
    }
    daily_candles = {
        c: df for c, df in daily_candles.items()
        if len(df) >= 30
    }

    print(f"Loaded: {len(daily_candles)} daily, {len(signal_candles)} signal, {len(funding_map)} funding")
    print(f"Coins: {sorted(signal_candles.keys())}")
    return daily_candles, signal_candles, funding_map


def split_by_date(candles_map, funding_map, cutoff: pd.Timestamp):
    """Split candles/funding at cutoff into IS/OOS."""
    is_c, oos_c, is_f, oos_f = {}, {}, {}, {}
    for coin, df in candles_map.items():
        is_df = df[df.index <= cutoff]
        oos_df = df[df.index > cutoff]
        if len(is_df) >= 50:
            is_c[coin] = is_df
        if len(oos_df) >= 20:
            oos_c[coin] = oos_df
    for coin, df in funding_map.items():
        is_df = df[df.index <= cutoff]
        oos_df = df[df.index > cutoff]
        if len(is_df) >= 20:
            is_f[coin] = is_df
        if len(oos_df) >= 10:
            oos_f[coin] = oos_df
    return is_c, is_f, oos_c, oos_f


def run_variant(
    name: str,
    params: StrategyParams,
    signal_candles: dict,
    funding_map: dict,
    universe_snapshots,
    label: str = "IS",
) -> dict:
    signals = build_signals_all(signal_candles, funding_map, params)
    if not signals:
        return {"sharpe": float("nan"), "n_trades": 0}
    result = run_backtest(
        signals, funding_map, universe_snapshots, params,
        initial_equity=INITIAL_EQUITY, label=f"{name}_{label}",
    )
    if not result.trades:
        return {
            "sharpe": float("nan"), "sortino": float("nan"),
            "calmar": float("nan"), "max_drawdown_pct": float("nan"),
            "total_return_pct": float("nan"), "win_rate_pct": float("nan"),
            "profit_factor": float("nan"), "n_trades": 0,
        }
    return compute_metrics(result, initial_equity=INITIAL_EQUITY, label=label)


# ── Parameter variants ────────────────────────────────────────────────────────

VARIANTS = [
    # ── Round 2 top candidates (reference) ───────────────────────────────────
    ("both_floor_wide",          # best OOS from round 2
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=4.0,
         max_hold_hours=48, direction="both",
         exit_mean_low=35.0, exit_mean_high=65.0,
         min_abs_funding_8h=0.0003, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
    ("short_floor_tp4",          # best IS from round 2
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=4.0,
         max_hold_hours=48, direction="short_only",
         exit_mean_low=35.0, exit_mean_high=65.0,
         min_abs_funding_8h=0.0003, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
    # ── Round 3: fine-tune around both_floor_wide ────────────────────────────
    ("bfw_floor2",               # lower funding floor
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=4.0,
         max_hold_hours=48, direction="both",
         exit_mean_low=35.0, exit_mean_high=65.0,
         min_abs_funding_8h=0.0002, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
    ("bfw_floor4",               # higher funding floor
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=4.0,
         max_hold_hours=48, direction="both",
         exit_mean_low=35.0, exit_mean_high=65.0,
         min_abs_funding_8h=0.0004, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
    ("bfw_tp35",                 # slightly lower TP
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=3.5,
         max_hold_hours=48, direction="both",
         exit_mean_low=35.0, exit_mean_high=65.0,
         min_abs_funding_8h=0.0003, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
    ("bfw_tp5",                  # higher TP — let winners run more
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=5.0,
         max_hold_hours=72, direction="both",
         exit_mean_low=30.0, exit_mean_high=70.0,
         min_abs_funding_8h=0.0003, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
    ("bfw_hold72",               # longer hold
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=4.0,
         max_hold_hours=72, direction="both",
         exit_mean_low=35.0, exit_mean_high=65.0,
         min_abs_funding_8h=0.0003, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
    ("bfw_sl15",                 # tighter stop — let carry absorb smaller moves
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=1.5, take_profit_pct=4.0,
         max_hold_hours=48, direction="both",
         exit_mean_low=35.0, exit_mean_high=65.0,
         min_abs_funding_8h=0.0003, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
    ("bfw_cd5",                  # longer cooldown — avoid whipsawing
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=4.0,
         max_hold_hours=48, direction="both",
         exit_mean_low=35.0, exit_mean_high=65.0,
         min_abs_funding_8h=0.0003, cooldown_bars=5,
         bar_interval_h=BAR_H,
     )),
    ("bfw_exit3070",             # narrower exit zone — exit sooner
     StrategyParams(
         entry_long_pct=5.0, entry_short_pct=95.0,
         stop_loss_pct=2.0, take_profit_pct=4.0,
         max_hold_hours=48, direction="both",
         exit_mean_low=30.0, exit_mean_high=70.0,
         min_abs_funding_8h=0.0003, cooldown_bars=3,
         bar_interval_h=BAR_H,
     )),
]


def fmt(m: dict) -> str:
    sharpe = m.get("sharpe", float("nan"))
    ret    = m.get("total_return_pct", float("nan"))
    wr_pct = m.get("win_rate_pct", float("nan"))
    dd     = m.get("max_drawdown_pct", float("nan"))
    pf     = m.get("profit_factor", float("nan"))
    n      = m.get("n_trades", 0)
    wr_str = f"{wr_pct:.1f}%" if not (isinstance(wr_pct, float) and np.isnan(wr_pct)) else "nan%"
    return (f"Sharpe={sharpe:+.2f}  Ret={ret:+.1f}%  WR={wr_str}"
            f"  MaxDD={dd:.1f}%  PF={pf:.2f}  N={n}")


def main():
    daily_candles, signal_candles, funding_map = load_all_data()

    if not signal_candles:
        print("ERROR: No cached data found. Run main.py first to populate the cache.")
        sys.exit(1)

    # Build universe snapshots using whatever range the data covers
    data_end = max(df.index.max() for df in signal_candles.values())
    print(f"Data spans: {SIGNAL_START.date()} → {data_end.date()}")

    print("Building universe snapshots…")
    universe_all = build_monthly_universes(
        SIGNAL_START.to_pydatetime(), data_end.to_pydatetime(), daily_candles
    )

    # IS/OOS split
    effective_is_end = min(IS_END, data_end)
    is_c, is_f, oos_c, oos_f = split_by_date(signal_candles, funding_map, effective_is_end)

    is_snapshots  = [s for s in universe_all if pd.Timestamp(s.date) <= effective_is_end]
    oos_snapshots = universe_all

    print(f"IS coins: {sorted(is_c.keys())}")
    print(f"OOS coins: {sorted(oos_c.keys())}")

    # ── IS grid search ──────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("IN-SAMPLE GRID SEARCH  (up to 2024-10-31)")
    print("="*72)

    is_results = {}
    for name, params in VARIANTS:
        print(f"  [{name:<26}] …", end="", flush=True)
        m = run_variant(name, params, is_c, is_f, is_snapshots, "IS")
        is_results[name] = m
        print(f" {fmt(m)}")

    ranked = sorted(
        is_results.items(),
        key=lambda x: x[1].get("sharpe", -999),
        reverse=True,
    )

    print("\n" + "-"*72)
    print("IS Ranking (by Sharpe):")
    for rank, (name, m) in enumerate(ranked, 1):
        s = m.get("sharpe", float("nan"))
        print(f"  {rank}. {name:<30} Sharpe={s:+.2f}")

    # ── OOS evaluation of top-3 IS candidates ───────────────────────────────
    variants_dict = {n: p for n, p in VARIANTS}
    top3 = ranked[:3]

    print("\n" + "="*72)
    print(f"OOS EVALUATION  (2024-11-01 → {data_end.date()})")
    print("="*72)
    oos_results = {}
    for name, is_m in top3:
        params = variants_dict[name]
        oos_m = run_variant(name, params, oos_c, oos_f, oos_snapshots, "OOS")
        oos_results[name] = oos_m
        print(f"\n  [{name}]")
        print(f"    IS  → {fmt(is_m)}")
        print(f"    OOS → {fmt(oos_m)}")

    # Pick best OOS by Sharpe
    best_oos_name = max(oos_results, key=lambda n: oos_results[n].get("sharpe", -999))
    best_name = best_oos_name
    best_params = variants_dict[best_name]

    # ── Full-period eval of best OOS candidate ───────────────────────────────
    print("\n" + "="*72)
    print(f"FULL PERIOD  (IS + OOS)  —  [{best_name}]")
    print("="*72)
    full_m = run_variant(best_name, best_params, signal_candles, funding_map, universe_all, "FULL")
    print(f"  {fmt(full_m)}")

    print("\n" + "="*72)
    print(f"Best variant (IS+OOS combined): [{best_name}]")
    for k, v in asdict(best_params).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
