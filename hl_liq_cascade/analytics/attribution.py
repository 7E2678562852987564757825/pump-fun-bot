"""Signal and regime attribution analysis."""

from __future__ import annotations

from typing import Any

import polars as pl


class SignalAttributor:
    def compute_attribution(self, trades: pl.DataFrame) -> dict[str, dict[str, Any]]:
        if "signal_source" not in trades.columns or len(trades) == 0:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for src in trades["signal_source"].unique().to_list():
            sub = trades.filter(pl.col("signal_source") == src)
            result[src] = self._stats(sub)
        return result

    def regime_attribution(self, trades: pl.DataFrame) -> dict[str, dict[str, Any]]:
        if "regime" not in trades.columns or len(trades) == 0:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for regime in trades["regime"].unique().to_list():
            sub = trades.filter(pl.col("regime") == regime)
            result[regime] = self._stats(sub)
        return result

    def coin_attribution(self, trades: pl.DataFrame) -> dict[str, dict[str, Any]]:
        if "coin" not in trades.columns or len(trades) == 0:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for coin in trades["coin"].unique().to_list():
            sub = trades.filter(pl.col("coin") == coin)
            result[coin] = self._stats(sub)
        return result

    def _stats(self, sub: pl.DataFrame) -> dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {"trade_count": 0}

        pnl = sub["pnl"]
        wins = pnl.filter(pnl > 0)
        losses = pnl.filter(pnl <= 0)
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0

        durations_h: list[float] = []
        if "entry_ts" in sub.columns and "exit_ts" in sub.columns:
            durations_h = ((sub["exit_ts"] - sub["entry_ts"]) / 3_600_000).to_list()

        return {
            "trade_count": n,
            "total_pnl": float(pnl.sum()),
            "win_rate": len(wins) / n,
            "profit_factor": float(wins.sum() or 0) / gross_loss if gross_loss > 0 else float("inf"),  # type: ignore[arg-type]
            "avg_pnl": float(pnl.mean() or 0),  # type: ignore[arg-type]
            "avg_duration_h": float(sum(durations_h) / len(durations_h)) if durations_h else 0.0,
        }
