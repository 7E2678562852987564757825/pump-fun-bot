"""Async WebSocket client for real-time Hyperliquid data streams."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

import msgspec
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from hl_liq_cascade.types import Event, EventKind, Trade

logger = logging.getLogger(__name__)

_WS_URL = "wss://api.hyperliquid.xyz/ws"
_RECONNECT_BASE_DELAY = 5.0   # seconds
_RECONNECT_MAX_DELAY = 120.0  # seconds


class HLWebSocketClient:
    """Real-time WebSocket client for Hyperliquid market data.

    Emits typed Event objects to the caller-provided async callback.
    Handles reconnection with exponential backoff.
    """

    def __init__(
        self,
        cfg: dict,
        on_event: Callable[[Event], Awaitable[None]],
    ) -> None:
        self._cfg = cfg
        self._on_event = on_event
        self._ws_url: str = cfg.get("ws_url", _WS_URL)

        # websockets connection handle
        self._ws: websockets.ClientConnection | None = None  # type: ignore[attr-defined]

        # Track active subscriptions so we can re-subscribe after reconnect
        self._subscriptions: list[dict] = []
        self._running = False

        # msgspec decoder for incoming Trade lists
        self._trade_dec = msgspec.json.Decoder(list[Trade])

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish the WebSocket connection."""
        logger.info("Connecting to %s", self._ws_url)
        self._ws = await websockets.connect(
            self._ws_url,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
        )
        logger.info("WebSocket connected")

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.warning("Error during disconnect: %s", exc)
            finally:
                self._ws = None
        logger.info("WebSocket disconnected")

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    async def _send(self, message: dict) -> None:
        """Serialize and send a message over the WebSocket."""
        if self._ws is None:
            raise RuntimeError("WebSocket is not connected")
        payload = json.dumps(message)
        await self._ws.send(payload)
        logger.debug("WS → %s", payload[:120])

    async def subscribe_all_mids(self) -> None:
        """Subscribe to all mid-price updates."""
        sub = {"type": "allMids"}
        msg = {"method": "subscribe", "subscription": sub}
        await self._send(msg)
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)

    async def subscribe_trades(self, coin: str) -> None:
        """Subscribe to trade updates for a specific coin."""
        sub = {"type": "trades", "coin": coin}
        msg = {"method": "subscribe", "subscription": sub}
        await self._send(msg)
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)

    async def subscribe_l2book(self, coin: str) -> None:
        """Subscribe to L2 order book updates for a specific coin."""
        sub = {"type": "l2Book", "coin": coin, "nSigFigs": 5}
        msg = {"method": "subscribe", "subscription": sub}
        await self._send(msg)
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)

    async def _resubscribe_all(self) -> None:
        """Re-send all subscriptions after a reconnect."""
        for sub in self._subscriptions:
            msg = {"method": "subscribe", "subscription": sub}
            await self._send(msg)
        logger.info("Re-subscribed %d channels after reconnect", len(self._subscriptions))

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    async def _handle_message(self, raw: str) -> None:
        """Parse a raw WebSocket message and emit the appropriate Event(s)."""
        try:
            msg: dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode WS message: %s", exc)
            return

        channel = msg.get("channel", "")
        data = msg.get("data")

        if channel == "allMids":
            await self._handle_all_mids(data)
        elif channel == "trades":
            await self._handle_trades(data)
        elif channel == "l2Book":
            await self._handle_l2book(data)
        elif channel == "subscriptionResponse":
            logger.debug("Subscription confirmed: %s", msg.get("data"))
        elif channel == "error":
            logger.error("WS error message: %s", msg)
        else:
            logger.debug("Unhandled WS channel '%s'", channel)

    async def _handle_all_mids(self, data: dict | None) -> None:
        """Handle allMids channel: emit a MARKET_TICK event."""
        if not isinstance(data, dict):
            return
        # data shape: {"mids": {"BTC": "65000.0", ...}} or directly coin→price
        mids_raw: dict = data.get("mids", data)
        try:
            mids = {coin: float(px) for coin, px in mids_raw.items()}
        except (ValueError, AttributeError) as exc:
            logger.warning("allMids parse error: %s", exc)
            return

        event = Event(
            ts=self._now_ms(),
            kind=EventKind.MARKET_TICK,
            payload={"mids": mids},
        )
        await self._on_event(event)

    async def _handle_trades(self, data: list | None) -> None:
        """Handle trades channel: parse trades and emit FILL or LIQ_EVENT."""
        if not isinstance(data, list):
            return

        for raw_trade in data:
            if not isinstance(raw_trade, dict):
                continue
            try:
                trade = Trade(
                    coin=raw_trade["coin"],
                    side=raw_trade["side"],
                    px=str(raw_trade.get("px", "0")),
                    sz=str(raw_trade.get("sz", "0")),
                    time=int(raw_trade.get("time", self._now_ms())),
                    hash=raw_trade.get("hash", ""),
                    tid=int(raw_trade.get("tid", 0)),
                    users=raw_trade.get("users", []),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Trade parse error: %s | raw=%s", exc, raw_trade)
                continue

            # Liquidation detection:
            # HL marks liquidations via:
            #   1. non-empty `users` list in the trade dict
            #   2. a `liq` field on the trade
            is_liq = (
                bool(trade.users)
                or raw_trade.get("liq") is not None
            )

            if is_liq:
                # Determine which side was liquidated
                # If side=="B" (buy), the liquidated position was short (forced to close short)
                # If side=="A" (sell), the liquidated position was long (forced to close long)
                liq_side = "short" if trade.side == "B" else "long"
                liq_address: str | None = trade.users[0] if trade.users else None

                event = Event(
                    ts=trade.time,
                    kind=EventKind.LIQ_EVENT,
                    payload={
                        "trade": trade,
                        "liq_side": liq_side,
                        "address": liq_address,
                        "raw": raw_trade,
                    },
                )
            else:
                event = Event(
                    ts=trade.time,
                    kind=EventKind.FILL,
                    payload={"trade": trade, "raw": raw_trade},
                )

            await self._on_event(event)

    async def _handle_l2book(self, data: dict | None) -> None:
        """Handle l2Book channel: emit a raw l2book event."""
        if data is None:
            return
        event = Event(
            ts=self._now_ms(),
            kind="l2book",
            payload=data,
        )
        await self._on_event(event)

    # ------------------------------------------------------------------
    # Main run loop with reconnection
    # ------------------------------------------------------------------

    async def run(self, coins: list[str]) -> None:
        """Connect, subscribe to all_mids + trades for each coin, receive indefinitely.

        Automatically reconnects with exponential backoff on disconnection.
        """
        self._running = True
        reconnect_delay = _RECONNECT_BASE_DELAY

        while self._running:
            try:
                await self.connect()
                # Subscribe
                await self.subscribe_all_mids()
                for coin in coins:
                    await self.subscribe_trades(coin)
                logger.info(
                    "Subscribed: allMids + trades for %d coins", len(coins)
                )
                # Reset reconnect delay on successful connection
                reconnect_delay = _RECONNECT_BASE_DELAY

                # Receive loop
                assert self._ws is not None
                async for raw_message in self._ws:
                    if not self._running:
                        break
                    if isinstance(raw_message, bytes):
                        raw_message = raw_message.decode("utf-8")
                    await self._handle_message(raw_message)

            except ConnectionClosed as exc:
                if not self._running:
                    break
                logger.warning(
                    "WS connection closed (code=%s reason=%s), reconnecting in %.1fs...",
                    exc.code, exc.reason, reconnect_delay,
                )
            except WebSocketException as exc:
                if not self._running:
                    break
                logger.warning(
                    "WS error: %s, reconnecting in %.1fs...", exc, reconnect_delay
                )
            except asyncio.CancelledError:
                logger.info("WS run loop cancelled")
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.error(
                    "Unexpected WS error: %s, reconnecting in %.1fs...",
                    exc, reconnect_delay, exc_info=True,
                )

            if not self._running:
                break

            await asyncio.sleep(reconnect_delay)
            # Exponential backoff capped at max delay
            reconnect_delay = min(reconnect_delay * 2, _RECONNECT_MAX_DELAY)

            # Re-subscribe after reconnect (subscriptions list was populated above)
            logger.info("Attempting reconnect...")

        logger.info("WS run loop exited")
