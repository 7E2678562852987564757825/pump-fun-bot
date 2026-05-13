"""Performance metrics for the liquidation cascade backtest."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_ANNUALIZE_DAILY = 252.0


class PerformanceMetrics:
    def compute_all(
        self,
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame,
        initial_capital: float,
        oos_split_ts: int | None = None,
    ) -> dict[str, Any]:
        if len(equity_curve) == 0:
            return {"error": "no trades"}

        metrics = self._compute_subset(trades, equity_curve, initial_capital)
        if oos_split_ts is not None:
            is_trades = trades.filter(pl.col("entry_ts") < oos_split_ts)
            oos_trades = trades.filter(pl.col("entry_ts") >= oos_split_ts)
            is_eq = equity_curve.filter(pl.col("ts") < oos_split_ts)
            oos_eq = equity_curve.filter(pl.col("ts") >= oos_split_ts)
            metrics["is_metrics"] = self._compute_subset(is_trades, is_eq, initial_capital)
            if len(oos_eq) > 0:
                oos_start_cap = oos_eq["equity"][0]
                metrics["oos_metrics"] = self._compute_subset(oos_trades, oos_eq, oos_start_cap)
            else:
                metrics["oos_metrics"] = {}
        return metrics

    def _compute_subset(
        self,
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame,
        initial_capital: float,
    ) -> dict[str, Any]:
        if len(equity_curve) == 0:
            return {}

        final_equity = float(equity_curve["equity"][-1])
        total_return = (final_equity - initial_capital) / initial_capital

        daily = self._daily_returns(equity_curve)
        max_dd, max_dd_days = self._max_drawdown(equity_curve)

        sharpe = self._sharpe(daily)
        sortino = self._sortino(daily)
        calmar = total_return / max_dd if max_dd > 0 else float("inf")

        n = len(trades)
        if n == 0:
            return {
                "total_return": total_return,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_drawdown": max_dd,
                "max_drawdown_duration_days": max_dd_days,
                "total_trades": 0,
            }

        pnl_col = trades["pnl"]
        wins = pnl_col.filter(pnl_col > 0)
        losses = pnl_col.filter(pnl_col <= 0)
        win_rate = len(wins) / n
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win = float(wins.mean() or 0) if len(wins) > 0 else 0.0  # type: ignore[arg-type]
        avg_loss = float(losses.mean() or 0) if len(losses) > 0 else 0.0  # type: ignore[arg-type]

        durations_h: list[float] = []
        if "entry_ts" in trades.columns and "exit_ts" in trades.columns:
            dur_ms = trades["exit_ts"] - trades["entry_ts"]
            durations_h = (dur_ms / 3_600_000).to_list()
        avg_duration_h = float(np.mean(durations_h)) if durations_h else 0.0

        duration_percentiles: dict[str, float] = {}
        if durations_h:
            arr = np.array(durations_h)
            duration_percentiles = {
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

        # Attribution by signal
        attribution: dict[str, float] = {}
        if "signal_source" in trades.columns:
            grp = trades.group_by("signal_source").agg(pl.col("pnl").sum())
            total_pnl = float(pnl_col.sum())
            for row in grp.to_dicts():
                src = row["signal_source"]
                pnl_frac = row["pnl"] / total_pnl if total_pnl != 0 else 0.0
                attribution[src] = pnl_frac

        per_coin: dict[str, float] = {}
        if "coin" in trades.columns:
            grp_coin = trades.group_by("coin").agg(pl.col("pnl").sum())
            for row in grp_coin.to_dicts():
                per_coin[row["coin"]] = row["pnl"]

        avg_cascade_notional: float = 0.0
        if "cascade_notional" in trades.columns:
            avg_cascade_notional = float(trades["cascade_notional"].drop_nulls().mean() or 0)  # type: ignore[arg-type]

        slippage_total = float(trades["slippage"].sum() or 0) if "slippage" in trades.columns else 0.0

        return {
            "total_return": total_return,
            "final_equity": final_equity,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": max_dd_days,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": n,
            "avg_duration_h": avg_duration_h,
            "trade_duration_percentiles": duration_percentiles,
            "signal_attribution": attribution,
            "per_coin_pnl": per_coin,
            "avg_cascade_notional": avg_cascade_notional,
            "slippage_total": slippage_total,
        }

    def _daily_returns(self, equity_curve: pl.DataFrame) -> pl.Series:
        if "ts" not in equity_curve.columns or len(equity_curve) < 2:
            return pl.Series([], dtype=pl.Float64)
        df = equity_curve.sort("ts")
        df = df.with_columns(pl.col("ts").cast(pl.Datetime("ms")))
        df = df.group_by_dynamic("ts", every="1d").agg(pl.col("equity").last())
        df = df.sort("ts")
        eq = df["equity"]
        returns = (eq[1:] - eq[:-1]) / eq[:-1]
        return returns

    def _max_drawdown(self, equity_curve: pl.DataFrame) -> tuple[float, int]:
        if len(equity_curve) < 2:
            return 0.0, 0
        eq = equity_curve.sort("ts")["equity"].to_numpy()
        running_max = np.maximum.accumulate(eq)
        drawdowns = (running_max - eq) / np.where(running_max > 0, running_max, 1.0)
        max_dd = float(drawdowns.max())

        # Duration: find longest consecutive drawdown period
        in_dd = drawdowns > 1e-6
        max_duration = 0
        current = 0
        for v in in_dd:
            if v:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0

        # Convert bar-count to days (each bar ≈ 1h candle → /24)
        days = max_duration / 24
        return max_dd, int(days)

    def _sharpe(self, daily_returns: pl.Series, annualization: float = _ANNUALIZE_DAILY) -> float:
        if len(daily_returns) < 2:
            return 0.0
        arr = daily_returns.to_numpy()
        std = float(np.std(arr, ddof=1))
        if std < 1e-12:
            return 0.0
        return float(np.mean(arr)) / std * math.sqrt(annualization)

    def _sortino(self, daily_returns: pl.Series, annualization: float = _ANNUALIZE_DAILY) -> float:
        if len(daily_returns) < 2:
            return 0.0
        arr = daily_returns.to_numpy()
        downside = arr[arr < 0]
        if len(downside) < 2:
            return float("inf") if float(np.mean(arr)) > 0 else 0.0
        downside_std = float(np.std(downside, ddof=1))
        if downside_std < 1e-12:
            return 0.0
        return float(np.mean(arr)) / downside_std * math.sqrt(annualization)

    def print_report(self, metrics: dict[str, Any]) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title="Backtest Performance Report", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            _add_rows(table, metrics)
            console.print(table)
        except ImportError:
            _print_plain(metrics)


def _add_rows(table: Any, metrics: dict[str, Any], prefix: str = "") -> None:
    skip = {"is_metrics", "oos_metrics", "signal_attribution", "per_coin_pnl",
            "trade_duration_percentiles"}
    for k, v in metrics.items():
        if k in skip:
            continue
        label = (prefix + k).replace("_", " ").title()
        if isinstance(v, float):
            if "return" in k or "rate" in k or "drawdown" in k:
                table.add_row(label, f"{v:.2%}")
            else:
                table.add_row(label, f"{v:.4f}")
        elif isinstance(v, int):
            table.add_row(label, str(v))
        else:
            table.add_row(label, str(v))


def _print_plain(metrics: dict[str, Any]) -> None:
    print("\n=== Backtest Performance Report ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue
        if isinstance(v, float):
            if any(x in k for x in ("return", "rate", "drawdown", "win")):
                print(f"  {k:40s}: {v:.2%}")
            else:
                print(f"  {k:40s}: {v:.4f}")
        else:
            print(f"  {k:40s}: {v}")
    if "is_metrics" in metrics:
        print("\n--- In-Sample ---")
        _print_plain(metrics["is_metrics"])
    if "oos_metrics" in metrics:
        print("\n--- OUT-OF-SAMPLE (held out, not iterated on) ---")
        _print_plain(metrics["oos_metrics"])
