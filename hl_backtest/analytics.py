"""
Performance analytics and visualization for the backtest results.

Computes: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor,
per-asset breakdown, monthly heatmap, equity curve, walk-forward, bootstrap CI.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from backtest import BacktestResult, Trade

logger = logging.getLogger(__name__)

TRADING_HOURS_PER_YEAR = 8760
RISK_FREE_RATE = 0.05  # 5% annualized


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _annualize(returns: pd.Series, freq: str = "1h") -> float:
    """Annualization factor based on return frequency."""
    periods = {"1h": 8760, "1D": 365, "1min": 525_600}
    return periods.get(freq, 8760)


def sharpe_ratio(daily_returns: pd.Series, rfr_annual: float = RISK_FREE_RATE) -> float:
    if daily_returns.std() == 0 or len(daily_returns) < 5:
        return 0.0
    rfr_daily = (1 + rfr_annual) ** (1 / 365) - 1
    excess = daily_returns - rfr_daily
    return float(excess.mean() / excess.std() * np.sqrt(365))


def sortino_ratio(daily_returns: pd.Series, rfr_annual: float = RISK_FREE_RATE) -> float:
    if len(daily_returns) < 5:
        return 0.0
    rfr_daily = (1 + rfr_annual) ** (1 / 365) - 1
    excess = daily_returns - rfr_daily
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf")
    return float(excess.mean() / downside.std() * np.sqrt(365))


def max_drawdown(equity_curve: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Returns (max_drawdown_pct, peak_time, trough_time)."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    min_idx = drawdown.idxmin()
    mdd = float(drawdown.min())
    if mdd == 0:
        return 0.0, None, None
    peak_time = equity_curve[:min_idx].idxmax()
    return abs(mdd), peak_time, min_idx


def longest_drawdown_duration(equity_curve: pd.Series) -> pd.Timedelta:
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max
    # Find consecutive runs of True
    if not underwater.any():
        return pd.Timedelta(0)
    groups = underwater.astype(int).groupby((underwater != underwater.shift()).cumsum())
    max_len = 0
    for _, grp in groups:
        if grp.iloc[0] == 1:
            duration = (grp.index[-1] - grp.index[0]).total_seconds() / 3600
            max_len = max(max_len, duration)
    return pd.Timedelta(hours=max_len)


def calmar_ratio(equity_curve: pd.Series, daily_returns: pd.Series) -> float:
    mdd, _, _ = max_drawdown(equity_curve)
    if mdd == 0:
        return float("inf")
    ann_return = (1 + daily_returns.mean()) ** 365 - 1
    return float(ann_return / mdd)


def compute_metrics(result: BacktestResult, initial_equity: float = 100_000.0, label: str = "") -> dict[str, Any]:
    """Compute all performance metrics for a BacktestResult."""
    eq = result.equity_curve
    dr = result.daily_returns
    trades = result.trades

    if eq.empty or len(trades) == 0:
        return {"label": label, "error": "No data"}

    final_equity = float(eq.iloc[-1])
    total_return = (final_equity / initial_equity) - 1

    # Annualized return
    n_days = max((eq.index[-1] - eq.index[0]).days, 1)
    ann_return = (1 + total_return) ** (365 / n_days) - 1

    mdd, _, _ = max_drawdown(eq)
    dd_dur = longest_drawdown_duration(eq)

    # Trade stats
    winning = [t for t in trades if t.net_pnl > 0]
    losing = [t for t in trades if t.net_pnl <= 0]
    win_rate = len(winning) / len(trades) if trades else 0.0
    avg_win = np.mean([t.net_pnl for t in winning]) if winning else 0.0
    avg_loss = np.mean([t.net_pnl for t in losing]) if losing else 0.0
    gross_win = sum(t.net_pnl for t in winning)
    gross_loss = abs(sum(t.net_pnl for t in losing))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    avg_hold = np.mean([t.hold_hours for t in trades]) if trades else 0.0
    total_fees = sum(t.fees for t in trades)
    total_funding = sum(t.funding_pnl for t in trades)

    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    return {
        "label": label,
        "total_return_pct": total_return * 100,
        "ann_return_pct": ann_return * 100,
        "sharpe": sharpe_ratio(dr),
        "sortino": sortino_ratio(dr),
        "calmar": calmar_ratio(eq, dr),
        "max_drawdown_pct": mdd * 100,
        "dd_duration_days": dd_dur.total_seconds() / 86400,
        "n_trades": len(trades),
        "win_rate_pct": win_rate * 100,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "profit_factor": profit_factor,
        "avg_hold_hours": avg_hold,
        "total_fees_usd": total_fees,
        "total_funding_pnl_usd": total_funding,
        "final_equity": final_equity,
        "exit_reasons": exit_counts,
    }


def per_asset_breakdown(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    coins = set(t.coin for t in trades)
    for coin in sorted(coins):
        ctrades = [t for t in trades if t.coin == coin]
        wins = [t for t in ctrades if t.net_pnl > 0]
        rows.append({
            "coin": coin,
            "n_trades": len(ctrades),
            "net_pnl": sum(t.net_pnl for t in ctrades),
            "win_rate": len(wins) / len(ctrades) * 100,
            "avg_hold_h": np.mean([t.hold_hours for t in ctrades]),
            "total_funding": sum(t.funding_pnl for t in ctrades),
        })
    df = pd.DataFrame(rows).sort_values("net_pnl", ascending=False)
    return df


def monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """Build a year × month grid of monthly returns."""
    monthly_eq = equity_curve.resample("ME").last().ffill()
    monthly_ret = monthly_eq.pct_change().dropna() * 100
    df = monthly_ret.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
    return pivot


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------

def bootstrap_sharpe_ci(
    daily_returns: pd.Series,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for Sharpe. Returns (lower, point_est, upper)."""
    rng = np.random.default_rng(seed)
    n = len(daily_returns)
    if n < 10:
        s = sharpe_ratio(daily_returns)
        return s, s, s
    arr = daily_returns.values
    boot_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        s = sharpe_ratio(pd.Series(sample))
        boot_sharpes.append(s)
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_sharpes, alpha * 100))
    hi = float(np.percentile(boot_sharpes, (1 - alpha) * 100))
    return lo, sharpe_ratio(daily_returns), hi


def walk_forward_analysis(
    result: BacktestResult,
    initial_equity: float,
    window_days: int = 182,
    step_days: int = 30,
) -> pd.DataFrame:
    """Slide a window over the equity curve and compute Sharpe per window."""
    eq = result.equity_curve
    if eq.empty:
        return pd.DataFrame()

    rows = []
    start = eq.index[0]
    end = eq.index[-1]
    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    cursor = start
    while cursor + window <= end:
        w_eq = eq.loc[cursor: cursor + window]
        if len(w_eq) < 10:
            cursor += step
            continue
        daily_eq = w_eq.resample("1D").last().ffill()
        dr = daily_eq.pct_change().dropna()
        mdd, _, _ = max_drawdown(w_eq)
        rows.append({
            "window_start": cursor,
            "window_end": cursor + window,
            "sharpe": sharpe_ratio(dr),
            "return_pct": (w_eq.iloc[-1] / w_eq.iloc[0] - 1) * 100,
            "max_dd_pct": mdd * 100,
        })
        cursor += step
    return pd.DataFrame(rows)


def sensitivity_analysis(
    run_fn,  # callable(entry_long_pct, entry_short_pct) -> BacktestResult
    thresholds: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Run strategy with different threshold pairs, return metrics table."""
    if thresholds is None:
        thresholds = [(5, 95), (10, 90), (1, 99), (5, 95)]
    rows = []
    for lo, hi in thresholds:
        result = run_fn(lo, hi)
        m = compute_metrics(result, label=f"pct_{lo}/{hi}")
        m["entry_long_pct"] = lo
        m["entry_short_pct"] = hi
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_equity_curve(
    is_result: BacktestResult,
    oos_result: BacktestResult,
    bnh_curve: pd.Series,
    output_path: Path,
    initial_equity: float = 100_000.0,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("Funding Rate Mean-Reversion Strategy — Hyperliquid Alt Perps", fontsize=13, fontweight="bold")

    ax_eq, ax_dd, ax_pos = axes

    # Combine equity curves
    is_eq = is_result.equity_curve
    oos_eq = oos_result.equity_curve

    # Normalize to 100
    is_norm = is_eq / initial_equity * 100
    oos_norm = oos_eq / is_eq.iloc[-1] * (is_eq.iloc[-1] / initial_equity * 100)
    bnh_norm = bnh_curve / bnh_curve.iloc[0] * 100 if not bnh_curve.empty else None

    split_time = is_eq.index[-1]

    ax_eq.plot(is_norm.index, is_norm.values, color="#2196F3", lw=1.5, label="Strategy (IS)")
    ax_eq.plot(oos_norm.index, oos_norm.values, color="#FF9800", lw=1.5, label="Strategy (OOS)")
    ax_eq.axvline(split_time, color="gray", ls="--", lw=1, alpha=0.7, label="IS/OOS split")
    if bnh_norm is not None:
        ax_eq.plot(bnh_norm.index, bnh_norm.values, color="#9E9E9E", lw=1, ls=":", label="Buy & Hold EW")
    ax_eq.set_ylabel("Equity (rebased 100)")
    ax_eq.legend(fontsize=8)
    ax_eq.grid(alpha=0.3)

    # Drawdown
    full_eq = pd.concat([is_eq, oos_eq]).sort_index()
    full_eq = full_eq[~full_eq.index.duplicated(keep="last")]
    running_max = full_eq.cummax()
    dd_pct = (full_eq - running_max) / running_max * 100
    ax_dd.fill_between(dd_pct.index, dd_pct.values, 0, color="#F44336", alpha=0.5, label="Drawdown %")
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.grid(alpha=0.3)

    # Number of positions
    is_pos = is_result.positions_over_time
    oos_pos = oos_result.positions_over_time
    all_pos = pd.concat([is_pos, oos_pos]).sort_index()
    ax_pos.step(all_pos.index, all_pos["n_positions"].values, color="#4CAF50", lw=1)
    ax_pos.set_ylabel("# Positions")
    ax_pos.set_ylim(0, 6)
    ax_pos.grid(alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity curve saved → %s", output_path)


def plot_monthly_heatmap(
    pivot_is: pd.DataFrame,
    pivot_oos: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, pivot, title in zip(axes, [pivot_is, pivot_oos], ["In-Sample", "Out-of-Sample"]):
        if pivot.empty:
            ax.set_title(f"Monthly Returns — {title} (no data)")
            continue
        vmax = max(abs(pivot.values[np.isfinite(pivot.values)]).max(), 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", norm=norm)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f"Monthly Returns (%) — {title}", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.02)
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6,
                            color="black" if abs(val) < vmax * 0.6 else "white")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Monthly heatmap saved → %s", output_path)


def plot_walk_forward(wf_df: pd.DataFrame, output_path: Path) -> None:
    if wf_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    ax1, ax2 = axes
    ax1.bar(range(len(wf_df)), wf_df["sharpe"], color=["#4CAF50" if s > 0 else "#F44336" for s in wf_df["sharpe"]])
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_title("Walk-Forward Sharpe Ratio (6-month windows)", fontsize=11)
    ax1.set_ylabel("Sharpe")
    ax1.set_xlabel("Window #")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(range(len(wf_df)), wf_df["return_pct"], color=["#4CAF50" if r > 0 else "#F44336" for r in wf_df["return_pct"]])
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_title("Walk-Forward Return per Window (%)", fontsize=11)
    ax2.set_ylabel("Return %")
    ax2.set_xlabel("Window #")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Walk-forward plot saved → %s", output_path)


def print_metrics_table(metrics: dict[str, Any], label: str = "") -> None:
    """Pretty-print metrics to console."""
    title = label or metrics.get("label", "Results")
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)
    fmt = [
        ("Total Return", "total_return_pct", "{:.2f}%"),
        ("Annualized Return", "ann_return_pct", "{:.2f}%"),
        ("Sharpe Ratio", "sharpe", "{:.3f}"),
        ("Sortino Ratio", "sortino", "{:.3f}"),
        ("Calmar Ratio", "calmar", "{:.3f}"),
        ("Max Drawdown", "max_drawdown_pct", "{:.2f}%"),
        ("Longest DD (days)", "dd_duration_days", "{:.1f}"),
        ("# Trades", "n_trades", "{:d}"),
        ("Win Rate", "win_rate_pct", "{:.1f}%"),
        ("Avg Win (USD)", "avg_win_usd", "{:.2f}"),
        ("Avg Loss (USD)", "avg_loss_usd", "{:.2f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Avg Hold (hours)", "avg_hold_hours", "{:.1f}"),
        ("Total Fees (USD)", "total_fees_usd", "{:.2f}"),
        ("Funding PnL (USD)", "total_funding_pnl_usd", "{:.2f}"),
    ]
    for name, key, fmt_str in fmt:
        val = metrics.get(key, "N/A")
        if val == "N/A":
            print(f"  {name:<25} {'N/A':>15}")
        else:
            try:
                if key == "n_trades":
                    formatted = fmt_str.format(int(val))
                else:
                    formatted = fmt_str.format(float(val))
                print(f"  {name:<25} {formatted:>15}")
            except (ValueError, TypeError):
                print(f"  {name:<25} {str(val):>15}")
    if "exit_reasons" in metrics:
        print(f"\n  Exit reasons:")
        for reason, count in sorted(metrics["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<20} {count:>5}")
    print("=" * 55)
