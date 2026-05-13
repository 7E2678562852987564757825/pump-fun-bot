"""Stop management: hard stops and time stops."""

from __future__ import annotations

from hl_liq_cascade.types import BacktestPosition


class StopManager:
    def __init__(self, cfg: dict) -> None:
        self._hard_stop_pct: float = cfg.get("hard_stop_pct", 0.03)
        self._time_stop_h: float = cfg.get("time_stop_h", 1.0)

    def check_hard_stop(self, pos: BacktestPosition, current_px: float) -> bool:
        if pos.stop_px is None:
            return False
        if pos.side == "long":
            return current_px <= pos.stop_px
        return current_px >= pos.stop_px

    def check_time_stop(self, pos: BacktestPosition, current_ts: int) -> bool:
        fill_ts = pos.fill_ts if pos.fill_ts is not None else pos.entry_ts
        age_ms = current_ts - fill_ts
        return age_ms >= int(self._time_stop_h * 3_600_000)

    def stop_fill_px(
        self,
        pos: BacktestPosition,
        current_px: float,
        bid: float,
        ask: float,
    ) -> float:
        if pos.side == "long":
            return min(ask, pos.stop_px) if pos.stop_px else bid  # type: ignore[return-value]
        return max(bid, pos.stop_px) if pos.stop_px else ask  # type: ignore[return-value]
