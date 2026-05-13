"""Portfolio exposure and correlation management."""

from __future__ import annotations

import logging

from hl_liq_cascade.types import BacktestPosition

logger = logging.getLogger(__name__)


class ExposureManager:
    def __init__(self, cfg: dict) -> None:
        self._max_concurrent: int = cfg.get("max_concurrent_positions", 3)
        self._max_coin_exposure: float = cfg.get("max_coin_exposure", 0.30)
        self._correlation_cap: bool = cfg.get("correlation_cap", True)
        self._groups: list[list[str]] = cfg.get("correlated_coins", [])

    def can_open(
        self,
        coin: str,
        side: str,
        size_usd: float,
        equity: float,
        open_positions: list[BacktestPosition],
    ) -> tuple[bool, str]:
        if len(open_positions) >= self._max_concurrent:
            return False, "max_concurrent"

        coin_exposure = self.current_exposure(coin, open_positions, equity)
        if coin_exposure + size_usd / equity > self._max_coin_exposure:
            return False, "max_coin_exposure"

        if self._correlation_cap and not self._passes_correlation_check(coin, side, open_positions):
            return False, "correlation_cap"

        return True, "ok"

    def current_exposure(
        self,
        coin: str,
        open_positions: list[BacktestPosition],
        equity: float,
    ) -> float:
        coin_usd = sum(p.size_usd for p in open_positions if p.coin == coin and p.is_open)
        return coin_usd / equity if equity > 0 else 0.0

    def _passes_correlation_check(
        self,
        coin: str,
        side: str,
        open_positions: list[BacktestPosition],
    ) -> bool:
        for group in self._groups:
            if coin not in group:
                continue
            same_side_in_group = sum(
                1 for p in open_positions
                if p.coin in group and p.side == side and p.is_open
            )
            if same_side_in_group >= 2:
                logger.debug(
                    "Correlation cap: %d %s positions already in group %s, blocking %s %s",
                    same_side_in_group, side, group, side, coin,
                )
                return False
        return True
