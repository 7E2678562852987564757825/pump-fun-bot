"""Real-time data feed for live/paper trading.

NOT wired for real order execution — paper trading only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from hl_liq_cascade.data.ws_client import HLWebSocketClient
from hl_liq_cascade.data.api_client import HLApiClient
from hl_liq_cascade.liq_map.builder import LiqMapBuilder
from hl_liq_cascade.types import Event, EventKind, LiqMap, MarketSnapshot, Position

logger = logging.getLogger(__name__)

_LIQ_MAP_REFRESH_S = 60
_SNAPSHOT_REFRESH_S = 300


class LiveFeed:
    def __init__(
        self,
        cfg: dict,
        on_event: Callable[[Event], Awaitable[None]],
    ) -> None:
        self._cfg = cfg
        self._on_event = on_event
        self._api = HLApiClient(cfg["api"])
        self._ws = HLWebSocketClient(cfg["api"], on_event=on_event)
        self._builder = LiqMapBuilder(cfg["liq_map"])
        self._positions: list[Position] = []
        self._mids: dict[str, float] = {}
        self._running = False

    async def start(self, coins: list[str]) -> None:
        self._running = True
        self._coins = coins

        # Initial snapshot of mids and a lightweight position list
        try:
            self._mids = await self._api.get_all_mids()
        except Exception as exc:
            logger.warning("Failed to get initial mids: %s", exc)

        asyncio.create_task(self._refresh_liq_maps())
        asyncio.create_task(self._ws.run(coins))
        logger.info("LiveFeed started for coins: %s", coins)

    async def stop(self) -> None:
        self._running = False
        await self._ws.disconnect()
        await self._api.close()

    async def _refresh_liq_maps(self) -> None:
        while self._running:
            try:
                ts = int(time.time() * 1000)
                if self._positions and self._mids:
                    liq_maps = self._builder.build(self._positions, self._mids, ts)
                    for coin, lm in liq_maps.items():
                        if coin in self._coins:
                            event = Event(ts=ts, kind=EventKind.LIQ_MAP, payload=lm)
                            await self._on_event(event)
            except Exception as exc:
                logger.error("Error refreshing liq maps: %s", exc)
            await asyncio.sleep(_LIQ_MAP_REFRESH_S)
