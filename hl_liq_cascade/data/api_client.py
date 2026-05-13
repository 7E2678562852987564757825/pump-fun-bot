"""Rate-limited async HTTP client for the Hyperliquid REST API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
import msgspec
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from hl_liq_cascade.types import (
    AssetMeta,
    AssetCtx,
    ClearinghouseState,
    Fill,
    FundingPayment,
    Candle,
    Trade,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.hyperliquid.xyz"
_INFO_ENDPOINT = "/info"
_RATE_LIMIT_RPS = 10  # requests per second


class HLApiClient:
    """Async HTTP client for the Hyperliquid REST API with rate limiting and retries."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._base_url = cfg.get("base_url", _BASE_URL)
        self._timeout = cfg.get("timeout", 30.0)
        self._rps = cfg.get("rate_limit_rps", cfg.get("requests_per_second", _RATE_LIMIT_RPS))

        # Token bucket state — lock makes the sliding-window check+append atomic
        self._rate_lock = asyncio.Lock()
        self._last_request_times: list[float] = []

        # httpx client — created lazily so the event loop is available
        self._client: httpx.AsyncClient | None = None

        # msgspec decoders for each response type
        self._dec_meta = msgspec.json.Decoder(list)
        self._dec_clearing = msgspec.json.Decoder(ClearinghouseState)
        self._dec_fills = msgspec.json.Decoder(list[Fill])
        self._dec_funding = msgspec.json.Decoder(list[FundingPayment])
        self._dec_candles = msgspec.json.Decoder(list[Candle])
        self._dec_trades = msgspec.json.Decoder(list[Trade])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def close(self) -> None:
        """Alias for aclose() for ergonomic use."""
        await self.aclose()

    async def __aenter__(self) -> "HLApiClient":
        await self._get_client()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Rate limiting (sliding-window token bucket)
    # ------------------------------------------------------------------

    async def _rate_limit(self) -> None:
        """Enforce at most _rps requests per second using a sliding window.

        The Lock makes the check-then-append atomic so concurrent coroutines
        cannot all slip through the window check simultaneously.
        """
        async with self._rate_lock:
            now = time.monotonic()
            self._last_request_times = [t for t in self._last_request_times if now - t < 1.0]
            if len(self._last_request_times) >= self._rps:
                sleep_s = 1.0 - (now - self._last_request_times[0])
                if sleep_s > 0:
                    logger.debug("Rate limit reached, sleeping %.3fs", sleep_s)
                    await asyncio.sleep(sleep_s)
                    # Re-prune after sleep
                    now = time.monotonic()
                    self._last_request_times = [t for t in self._last_request_times if now - t < 1.0]
            self._last_request_times.append(time.monotonic())

    # ------------------------------------------------------------------
    # Core POST helper with retry
    # ------------------------------------------------------------------

    async def _post(self, body: dict) -> bytes:
        """POST to /info with retries, rate limiting, and logging."""
        await self._rate_limit()
        return await self._post_with_retry(body)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _post_with_retry(self, body: dict) -> bytes:
        client = await self._get_client()
        payload = msgspec.json.encode(body)
        logger.debug("POST /info body=%s", body.get("type", body))
        try:
            resp = await client.post(_INFO_ENDPOINT, content=payload)
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP %s error for %s: %s",
                exc.response.status_code,
                body.get("type"),
                exc.response.text[:200],
            )
            raise
        except httpx.HTTPError as exc:
            logger.error("HTTP error for %s: %s", body.get("type"), exc)
            raise

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def get_meta(self) -> tuple[list[AssetMeta], list[AssetCtx]]:
        """Fetch metadata and asset contexts for all perpetuals.

        Response shape: [{"universe": [AssetMeta...]}, [AssetCtx...]]
        """
        raw = await self._post({"type": "metaAndAssetCtxs"})
        data: list = msgspec.json.decode(raw)
        meta_block: dict = data[0]
        ctx_block: list = data[1]

        metas = msgspec.convert(meta_block["universe"], list[AssetMeta])
        ctxs = msgspec.convert(ctx_block, list[AssetCtx])
        return metas, ctxs

    async def get_clearinghouse_state(self, address: str) -> ClearinghouseState:
        """Fetch clearinghouse state for one user address."""
        raw = await self._post({"type": "clearinghouseState", "user": address})
        return self._dec_clearing.decode(raw)

    async def get_user_fills(
        self,
        address: str,
        start_ms: int,
        end_ms: int | None = None,
    ) -> list[Fill]:
        """Fetch fills for a user in [start_ms, end_ms), sorted ascending by time."""
        body: dict = {
            "type": "userFills",
            "user": address,
            "startTime": start_ms,
        }
        if end_ms is not None:
            body["endTime"] = end_ms
        raw = await self._post(body)
        fills: list[Fill] = self._dec_fills.decode(raw)
        fills.sort(key=lambda f: f.time)
        return fills

    async def get_user_funding(
        self,
        address: str,
        start_ms: int,
        end_ms: int | None = None,
    ) -> list[FundingPayment]:
        """Fetch funding payment history for a user."""
        body: dict = {
            "type": "userFunding",
            "user": address,
            "startTime": start_ms,
        }
        if end_ms is not None:
            body["endTime"] = end_ms
        raw = await self._post(body)
        payments: list[FundingPayment] = self._dec_funding.decode(raw)
        payments.sort(key=lambda p: p.time)
        return payments

    async def get_all_mids(self) -> dict[str, float]:
        """Fetch mid prices for all coins. Returns coin → mid_price."""
        raw = await self._post({"type": "allMids"})
        raw_dict: dict[str, str] = msgspec.json.decode(raw)
        return {coin: float(px) for coin, px in raw_dict.items()}

    async def get_candles(
        self,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[Candle]:
        """Fetch OHLCV candles for a coin and interval."""
        body = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
        raw = await self._post(body)
        candles: list[Candle] = self._dec_candles.decode(raw)
        candles.sort(key=lambda c: c.t)
        return candles

    async def get_leaderboard(self) -> list[dict]:
        """Fetch the leaderboard. Returns raw list of dicts.

        Falls back gracefully to an empty list if the endpoint is unavailable.
        """
        try:
            raw = await self._post({"type": "leaderboard"})
            data: Any = msgspec.json.decode(raw)
            if isinstance(data, dict):
                return data.get("leaderboardRows", [])
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("get_leaderboard failed (%s), use get_active_addresses() instead", exc)
        return []

    async def get_active_addresses(self, coins: list[str], per_coin: int = 50) -> list[str]:
        """Collect unique active trader addresses via recentTrades across multiple coins.

        Used as a fallback when the leaderboard endpoint is unavailable.
        Returns up to per_coin * len(coins) unique addresses.
        """
        seen: set[str] = set()
        for coin in coins:
            try:
                trades = await self.get_recent_trades(coin)
                for t in trades:
                    for addr in t.users:
                        if addr and addr.startswith("0x"):
                            seen.add(addr)
                logger.debug("get_active_addresses: %s contributed %d addrs", coin, len(trades))
            except Exception as exc:
                logger.warning("recentTrades failed for %s: %s", coin, exc)
        addrs = sorted(seen)
        logger.info("get_active_addresses: %d unique addresses collected from %d coins", len(addrs), len(coins))
        return addrs

    async def get_recent_trades(self, coin: str) -> list[Trade]:
        """Fetch recent trades for a coin."""
        body = {"type": "recentTrades", "coin": coin}
        raw = await self._post(body)
        trades: list[Trade] = self._dec_trades.decode(raw)
        trades.sort(key=lambda t: t.time)
        return trades
