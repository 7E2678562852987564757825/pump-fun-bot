"""Vol-targeted position sizing."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    def __init__(self, cfg: dict) -> None:
        self._risk_per_trade: float = cfg.get("portfolio_risk_per_trade", 0.005)
        self._max_coin_exposure: float = cfg.get("max_coin_exposure", 0.30)

    def size(
        self,
        equity: float,
        signal_score: float,
        atr: float,
        entry_px: float,
        stop_px: float,
        coin: str,
    ) -> float:
        """Return position size in USD, scaled by signal strength."""
        stop_distance = abs(entry_px - stop_px)
        if stop_distance < 1e-12 or entry_px < 1e-12:
            return 0.0

        risk_usd = equity * self._risk_per_trade
        size_coins = risk_usd / stop_distance
        size_usd = size_coins * entry_px

        # Cap at max_coin_exposure
        size_usd = min(size_usd, equity * self._max_coin_exposure)

        # Scale by signal strength (absolute value)
        size_usd *= min(abs(signal_score), 1.0)

        if size_usd < 10.0:
            return 0.0

        return size_usd

    def compute_stop(
        self,
        side: str,
        entry_px: float,
        hard_stop_pct: float = 0.03,
    ) -> float:
        if side == "long":
            return entry_px * (1.0 - hard_stop_pct)
        return entry_px * (1.0 + hard_stop_pct)
