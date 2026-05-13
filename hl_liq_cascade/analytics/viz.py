"""Visualization: matplotlib and plotly charts for the liquidation cascade strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_curve: pl.DataFrame,
    metrics: dict[str, Any],
    out_path: str | None = None,
    oos_split_ts: int | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if len(equity_curve) == 0:
        return

    df = equity_curve.sort("ts")
    ts = df["ts"].to_numpy()
    eq = df["equity"].to_numpy()

    running_max = np.maximum.accumulate(eq)
    dd = (running_max - eq) / np.where(running_max > 0, running_max, 1.0)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(ts, eq, color="steelblue", linewidth=1.5, label="Portfolio Equity")
    if oos_split_ts is not None:
        ax1.axvline(x=oos_split_ts, color="orange", linestyle="--", alpha=0.8, label="IS/OOS boundary")
    ax1.set_ylabel("Equity (USD)")
    ax1.set_title("HL Liquidation Cascade Strategy — Equity Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    sharpe = metrics.get("sharpe", 0)
    max_dd = metrics.get("max_drawdown", 0)
    total_ret = metrics.get("total_return", 0)
    n_trades = metrics.get("total_trades", 0)
    ax1.text(
        0.02, 0.97,
        f"Return: {total_ret:.1%}  |  Sharpe: {sharpe:.2f}  |  MaxDD: {max_dd:.1%}  |  Trades: {n_trades}",
        transform=ax1.transAxes, verticalalignment="top",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(ts, -dd * 100, 0, color="red", alpha=0.4, label="Drawdown %")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Liquidation map
# ---------------------------------------------------------------------------

def plot_liq_map(
    liq_map: Any,  # LiqMap
    price_history: list[tuple[int, float]],
    coin: str,
    out_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    buckets = liq_map.buckets
    if not buckets:
        return

    prices = [b.price for b in buckets]
    long_notional = [-b.long_notional / 1e6 for b in buckets]   # negative = left side
    short_notional = [b.short_notional / 1e6 for b in buckets]  # positive = right side

    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

    ax_main = axes[0]
    bar_h = (prices[1] - prices[0]) * 0.8 if len(prices) > 1 else 0.5
    ax_main.barh(prices, short_notional, height=bar_h, color="royalblue", alpha=0.7, label="Short liq (above)")
    ax_main.barh(prices, long_notional, height=bar_h, color="crimson", alpha=0.7, label="Long liq (below)")
    ax_main.axhline(y=liq_map.current_px, color="gold", linewidth=2, linestyle="--", label=f"Current ${liq_map.current_px:,.2f}")
    ax_main.set_xlabel("Notional at Liquidation (USD M)")
    ax_main.set_ylabel("Price")
    ax_main.set_title(f"{coin} Liquidation Map")
    ax_main.legend(fontsize=8)
    ax_main.grid(alpha=0.2)

    ax_price = axes[1]
    if price_history:
        ts_arr = [p[0] for p in price_history]
        px_arr = [p[1] for p in price_history]
        ax_price.plot(ts_arr, px_arr, color="steelblue", linewidth=1.5)
        ax_price.axhline(y=liq_map.current_px, color="gold", linewidth=2, linestyle="--")
        ax_price.set_xlabel("Time (ms)")
        ax_price.set_title(f"{coin} Price (168h)")
        ax_price.yaxis.set_label_position("right")
        ax_price.yaxis.tick_right()
        ax_price.grid(alpha=0.2)

    plt.suptitle(f"{coin} Liq Map + Price — {_fmt_ts(liq_map.ts)}", fontsize=12)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def interactive_liq_map(liq_map: Any, out_path: str | None = None) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    buckets = liq_map.buckets
    prices = [b.price for b in buckets]
    long_n = [b.long_notional / 1e6 for b in buckets]
    short_n = [b.short_notional / 1e6 for b in buckets]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=prices, x=[-n for n in long_n], name="Long Liq", orientation="h",
        marker_color="crimson", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        y=prices, x=short_n, name="Short Liq", orientation="h",
        marker_color="royalblue", opacity=0.7,
    ))
    fig.add_hline(y=liq_map.current_px, line_color="gold", line_width=2,
                  annotation_text=f"${liq_map.current_px:,.2f}")
    fig.update_layout(
        title=f"{liq_map.coin} Interactive Liquidation Map",
        xaxis_title="Notional (USD M)",
        yaxis_title="Price",
        barmode="overlay",
        height=800,
    )
    if out_path:
        fig.write_html(out_path)
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Signal scores
# ---------------------------------------------------------------------------

def plot_signal_scores(
    scores: dict[str, list[tuple[int, float]]],
    coin: str,
    out_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    n = len(scores)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"cascade_frontrun": "steelblue", "postcascade_fade": "darkorange", "squeeze": "purple"}
    for ax, (name, data) in zip(axes, scores.items()):
        if not data:
            continue
        ts_arr = [d[0] for d in data]
        sc_arr = [d[1] for d in data]
        ax.plot(ts_arr, sc_arr, color=colors.get(name, "gray"), linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.fill_between(ts_arr, sc_arr, 0,
                         where=[s > 0 for s in sc_arr], color="green", alpha=0.2)
        ax.fill_between(ts_arr, sc_arr, 0,
                         where=[s < 0 for s in sc_arr], color="red", alpha=0.2)
        ax.set_ylabel(name)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.2)

    plt.suptitle(f"{coin} Signal Scores Over Time", fontsize=12)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Cascade events overlay
# ---------------------------------------------------------------------------

def plot_cascade_events(
    liq_events: list[Any],
    price_history: list[tuple[int, float]],
    coin: str,
    out_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    if price_history:
        ts_arr = [p[0] for p in price_history]
        px_arr = [p[1] for p in price_history]
        ax.plot(ts_arr, px_arr, color="steelblue", linewidth=1.2, zorder=2, label="Price")

    for ev in liq_events:
        color = "red" if ev.side == "long" else "blue"
        height = ev.notional / 1e6
        ax.axvline(x=ev.ts, color=color, alpha=min(height / 5, 0.8), linewidth=max(height * 0.5, 0.5))

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Price")
    ax.set_title(f"{coin} Price + Cascade Events (red=long liq, blue=short liq)")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Trade analysis grid
# ---------------------------------------------------------------------------

def plot_trade_analysis(
    trades: pl.DataFrame,
    out_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if len(trades) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pnl = trades["pnl"].to_numpy()

    # Duration histogram
    ax = axes[0, 0]
    if "entry_ts" in trades.columns and "exit_ts" in trades.columns:
        dur_h = ((trades["exit_ts"] - trades["entry_ts"]) / 3_600_000).to_numpy()
        ax.hist(dur_h, bins=40, color="steelblue", alpha=0.7)
        ax.axvline(np.median(dur_h), color="orange", linewidth=1.5, linestyle="--",
                   label=f"Median: {np.median(dur_h):.1f}h")
        ax.legend()
    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Count")
    ax.set_title("Trade Duration Distribution")
    ax.grid(alpha=0.2)

    # PnL distribution
    ax = axes[0, 1]
    ax.hist(pnl, bins=50, color="mediumseagreen", alpha=0.7)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(np.mean(pnl), color="orange", linewidth=1.5, linestyle="--",
               label=f"Mean: ${np.mean(pnl):.0f}")
    ax.legend()
    ax.set_xlabel("PnL (USD)")
    ax.set_ylabel("Count")
    ax.set_title("PnL Distribution")
    ax.grid(alpha=0.2)

    # PnL by signal source
    ax = axes[1, 0]
    if "signal_source" in trades.columns:
        sources = trades["signal_source"].unique().to_list()
        pnls_by_src = [
            float(trades.filter(pl.col("signal_source") == s)["pnl"].sum())
            for s in sources
        ]
        colors = ["steelblue" if p >= 0 else "crimson" for p in pnls_by_src]
        ax.bar(sources, pnls_by_src, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Signal")
    ax.set_ylabel("Total PnL (USD)")
    ax.set_title("PnL by Signal Source")
    ax.grid(alpha=0.2)

    # PnL by coin (top 10)
    ax = axes[1, 1]
    if "coin" in trades.columns:
        grp = (
            trades.group_by("coin")
            .agg(pl.col("pnl").sum())
            .sort("pnl", descending=True)
            .head(10)
        )
        coins = grp["coin"].to_list()
        pnls = grp["pnl"].to_numpy()
        colors = ["steelblue" if p >= 0 else "crimson" for p in pnls]
        ax.barh(coins[::-1], pnls[::-1], color=colors[::-1], alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Total PnL (USD)")
    ax.set_title("PnL by Coin (Top 10)")
    ax.grid(alpha=0.2)

    plt.suptitle("Trade Analysis", fontsize=14)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_oi_coverage(
    coverage_by_coin: dict[str, float],
    out_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if not coverage_by_coin:
        return

    coins = list(coverage_by_coin.keys())
    vals = [coverage_by_coin[c] for c in coins]
    colors = ["steelblue" if v >= 0.5 else "orange" for v in vals]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(coins, vals, color=colors, alpha=0.8)
    ax.axvline(0.5, color="red", linewidth=1.5, linestyle="--", label="50% coverage")
    ax.set_xlabel("OI Coverage Fraction")
    ax.set_title("OI Coverage by Coin (from top 500 addresses)")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _fmt_ts(ts_ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
