"""Loads cached historical data and populates the backtest EventBus."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import numpy as np

from hl_liq_cascade.types import Event, EventKind, LiqEvent, MarketSnapshot
from hl_liq_cascade.backtest.event_bus import EventBus
from hl_liq_cascade.data.cache import Cache
from hl_liq_cascade.liq_map.builder import LiqMapBuilder

logger = logging.getLogger(__name__)


class BacktestDataLoader:
    def __init__(self, cache: Cache, liq_map_builder: LiqMapBuilder, cfg: dict) -> None:
        self._cache = cache
        self._builder = liq_map_builder
        self._cfg = cfg

    def load_events(
        self,
        coins: list[str],
        start_ts: int,
        end_ts: int,
    ) -> EventBus:
        bus = EventBus()
        events: list[Event] = []

        for coin in coins:
            # Load 1h candles
            candles = self._cache.load_candles(coin, "1h")
            if candles is None or len(candles) == 0:
                logger.warning("No candles cached for %s, skipping", coin)
                continue

            candles = candles.filter(
                (pl.col("t") >= start_ts) & (pl.col("t") <= end_ts)
            ).sort("t")

            if len(candles) == 0:
                logger.warning("No candles in range for %s", coin)
                continue

            # Compute ATR column
            candles = self._compute_atr(candles)

            logger.info("Loading %d candles for %s", len(candles), coin)

            for row in candles.to_dicts():
                mid = float(row["c"])
                atr = float(row.get("atr", mid * 0.01))
                slip_bps = 2 if coin in self._cfg.get("execution", {}).get("major_coins", ["BTC", "ETH"]) else 5
                slip = slip_bps / 10_000
                spread = mid * slip / 2

                event = Event(
                    ts=int(row["t"]),
                    kind=EventKind.CANDLE,
                    payload={
                        "coin": coin,
                        "o": float(row["o"]),
                        "h": float(row["h"]),
                        "l": float(row["l"]),
                        "c": mid,
                        "v": float(row["v"]),
                        "atr": atr,
                        "funding_rate": 0.0,  # would be populated from funding history
                        "oi_long": 0.0,
                        "oi_short": 0.0,
                        "vol_24h": float(row["v"]) * mid * 24,  # rough proxy
                    },
                )
                events.append(event)

                # Also emit a market tick at each candle open
                tick_event = Event(
                    ts=int(row["t"]),
                    kind=EventKind.MARKET_TICK,
                    payload={"mids": {coin: mid}},
                )
                events.append(tick_event)

        # Load position snapshots and build liq maps
        liq_map_events = self._load_liq_map_events(coins, start_ts, end_ts)
        events.extend(liq_map_events)

        # Load liquidation events from fills marked as liquidations
        liq_events = self._load_liq_events(coins, start_ts, end_ts)
        events.extend(liq_events)

        # Sort and push all events — EventBus enforces ordering at pop time
        events.sort(key=lambda e: e.ts)
        for event in events:
            try:
                bus.push(event)
            except ValueError as exc:
                logger.warning("Skipping event: %s", exc)

        logger.info("EventBus populated with %d events", bus.size())
        return bus

    def _load_liq_map_events(
        self,
        coins: list[str],
        start_ts: int,
        end_ts: int,
    ) -> list[Event]:
        events: list[Event] = []
        snapshots = self._cache.load_position_snapshots(start_ts, end_ts)
        if snapshots is None or len(snapshots) == 0:
            logger.warning("No position snapshots found for liq map events")
            return events

        # Group snapshots by ts bucket (hourly)
        if "snapshot_ts" not in snapshots.columns:
            return events

        unique_ts = snapshots["snapshot_ts"].unique().sort().to_list()
        logger.info("Building liq maps from %d snapshot timestamps", len(unique_ts))

        for snap_ts in unique_ts:
            if snap_ts < start_ts or snap_ts > end_ts:
                continue

            snap_df = snapshots.filter(pl.col("snapshot_ts") == snap_ts)
            if len(snap_df) == 0:
                continue

            # Build mock mids from entry prices as proxy (real would use mark prices)
            mids: dict[str, float] = {}
            for coin in coins:
                coin_rows = snap_df.filter(pl.col("coin") == coin)
                if len(coin_rows) > 0:
                    mids[coin] = float(coin_rows["entry_px"].median() or 0)  # type: ignore[arg-type]

            if not mids:
                continue

            # Convert DataFrame rows to Position objects for the builder
            from hl_liq_cascade.types import Position
            positions = []
            for row in snap_df.to_dicts():
                if row.get("liq_px") is None:
                    continue
                positions.append(Position(
                    address=row.get("address", ""),
                    coin=row["coin"],
                    size=row["size"],
                    entry_px=row["entry_px"],
                    liq_px=row["liq_px"],
                    margin_used=0.0,
                    unrealized_pnl=0.0,
                    leverage=10,
                    leverage_type="isolated",
                    snapshot_ts=snap_ts,
                ))

            try:
                liq_maps = self._builder.build(positions, mids, snap_ts)
            except Exception as exc:
                logger.warning("Failed to build liq map at ts=%d: %s", snap_ts, exc)
                continue

            for coin, lm in liq_maps.items():
                events.append(Event(ts=snap_ts, kind=EventKind.LIQ_MAP, payload=lm))

        return events

    def _load_liq_events(
        self,
        coins: list[str],
        start_ts: int,
        end_ts: int,
    ) -> list[Event]:
        events: list[Event] = []
        liq_map_path = Path(self._cache.cache_dir) / "liq_events"
        if not liq_map_path.exists():
            return events

        for coin in coins:
            coin_path = liq_map_path / f"{coin}.parquet"
            if not coin_path.exists():
                continue
            try:
                df = pl.read_parquet(coin_path)
                df = df.filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts))
                for row in df.to_dicts():
                    ev = LiqEvent(
                        coin=coin,
                        side=row.get("side", "long"),
                        size=row.get("size", 0.0),
                        px=row.get("px", 0.0),
                        notional=row.get("notional", 0.0),
                        ts=row["ts"],
                    )
                    events.append(Event(ts=ev.ts, kind=EventKind.LIQ_EVENT, payload=ev))
            except Exception as exc:
                logger.warning("Failed to load liq events for %s: %s", coin, exc)

        return events

    def _compute_atr(self, candles: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Add exponential ATR column to candles."""
        if "h" not in candles.columns or "l" not in candles.columns:
            return candles.with_columns(pl.lit(0.0).alias("atr"))

        high = candles["h"].cast(pl.Float64).to_numpy()
        low = candles["l"].cast(pl.Float64).to_numpy()
        tr = high - low

        alpha = 2.0 / (period + 1)
        atr = np.zeros(len(tr))
        if len(tr) > 0:
            atr[0] = tr[0]
            for i in range(1, len(tr)):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        return candles.with_columns(pl.Series("atr", atr))

    def _candle_to_snapshot(
        self,
        candle_row: dict,
        coin: str,
        funding_rate: float = 0.0,
        long_oi: float = 0.0,
        short_oi: float = 0.0,
        vol_24h: float = 0.0,
    ) -> MarketSnapshot:
        mid = float(candle_row["c"])
        atr = float(candle_row.get("atr", mid * 0.01))
        slip_half = mid * 0.0005
        return MarketSnapshot(
            coin=coin,
            ts=int(candle_row["t"]),
            mid_px=mid,
            bid_px=mid - slip_half,
            ask_px=mid + slip_half,
            mark_px=mid,
            funding_rate=funding_rate,
            open_interest_long=long_oi,
            open_interest_short=short_oi,
            volume_24h=vol_24h,
            atr_1h=atr,
        )
