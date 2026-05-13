"""Position reconstruction engine: replays fills+funding to build position history."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import polars as pl

from hl_liq_cascade.types import Fill, FundingPayment, Position
from hl_liq_cascade.data.cache import Cache

logger = logging.getLogger(__name__)

# Maintenance margin fraction (0.5%)
MAINTENANCE_MARGIN: float = 0.005

# How many ms in 30 days (pagination window for fill fetches)
_THIRTY_DAYS_MS: int = 30 * 24 * 60 * 60 * 1_000

# Maximum concurrent address reconstructions
_RECONSTRUCT_CONCURRENCY = 20

# Hyperliquid mainnet launched 2022-11-01; no fills can exist before this
_HL_GENESIS_MS: int = 1_667_260_800_000


# ---------------------------------------------------------------------------
# Internal position state used during fill replay
# ---------------------------------------------------------------------------

@dataclass
class _CoinPosition:
    """Mutable per-coin position state used during fill replay."""
    size: float = 0.0          # + long, - short
    entry_px: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0


def _parse_dir(dir_str: str) -> tuple[str, str]:
    """Parse the `dir` field of a Fill into (action, side).

    Known values:
      "Open Long", "Close Long", "Open Short", "Close Short"
      "Buy"  (older format, treat as Open Long)
      "Sell" (older format, treat as Open Short)

    Returns:
        action: "open" | "close"
        side:   "long" | "short"
    """
    dl = dir_str.lower().strip()
    if "open" in dl and "long" in dl:
        return "open", "long"
    if "close" in dl and "long" in dl:
        return "close", "long"
    if "open" in dl and "short" in dl:
        return "open", "short"
    if "close" in dl and "short" in dl:
        return "close", "short"
    # Fallback: use side character
    if dl in ("buy", "b"):
        return "open", "long"
    if dl in ("sell", "a"):
        return "open", "short"
    # Default — treat as open long for unknown formats
    logger.warning("Unknown fill dir '%s', defaulting to open/long", dir_str)
    return "open", "long"


def _apply_fill(pos: _CoinPosition, fill: Fill) -> list[dict]:
    """Apply one fill to a position and return snapshot row(s).

    Returns a list of snapshot dicts (usually one, two on a position flip).
    """
    px = float(fill.px)
    sz = float(fill.sz)
    fee = float(fill.fee)
    closed_pnl = float(fill.closedPnl)

    action, side = _parse_dir(fill.dir)

    # Determine signed size delta
    # "Open Long" or "Close Short" → positive delta
    # "Open Short" or "Close Long" → negative delta
    if side == "long":
        if action == "open":
            size_delta = +sz
        else:  # close long
            size_delta = -sz
    else:  # short
        if action == "open":
            size_delta = -sz
        else:  # close short
            size_delta = +sz

    pos.total_fees += fee
    pos.realized_pnl += closed_pnl

    snapshots: list[dict] = []

    old_size = pos.size
    new_size = old_size + size_delta

    # ---------------------------------------------------------------
    # Case 1: Position goes through zero (flip)
    # ---------------------------------------------------------------
    if old_size != 0.0 and (
        (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0)
    ):
        # Close the old position entirely
        pos.realized_pnl += (px - pos.entry_px) * old_size  # gross PnL on close
        # Emit a snapshot for the closed position
        snapshots.append({
            "size": 0.0,
            "entry_px": pos.entry_px,
            "realized_pnl": pos.realized_pnl,
            "fees": pos.total_fees,
        })
        # Open a new position in the opposite direction
        remaining_sz = abs(new_size)
        pos.size = new_size
        pos.entry_px = px if remaining_sz > 0 else 0.0
        # Emit another snapshot for the new position
        snapshots.append({
            "size": pos.size,
            "entry_px": pos.entry_px,
            "realized_pnl": pos.realized_pnl,
            "fees": pos.total_fees,
        })
        return snapshots

    # ---------------------------------------------------------------
    # Case 2: Adding to position (same direction or new position from flat)
    # ---------------------------------------------------------------
    if (old_size >= 0 and size_delta > 0) or (old_size <= 0 and size_delta < 0):
        # Weighted average entry price
        total_abs = abs(old_size) + abs(size_delta)
        if total_abs > 0:
            pos.entry_px = (abs(old_size) * pos.entry_px + abs(size_delta) * px) / total_abs
        pos.size = new_size
    else:
        # ---------------------------------------------------------------
        # Case 3: Reducing position (partial close)
        # ---------------------------------------------------------------
        close_sz = abs(size_delta)
        pnl_this_fill = (px - pos.entry_px) * (close_sz if old_size > 0 else -close_sz)
        pos.realized_pnl += pnl_this_fill
        pos.size = new_size
        if abs(new_size) < 1e-12:
            pos.entry_px = 0.0

    snapshots.append({
        "size": pos.size,
        "entry_px": pos.entry_px,
        "realized_pnl": pos.realized_pnl,
        "fees": pos.total_fees,
    })
    return snapshots


class PositionStore:
    """Reconstructs per-user and aggregate position histories from fill data."""

    def __init__(self, cache: Cache, cfg: dict) -> None:
        self._cache = cache
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Paginated fill fetching
    # ------------------------------------------------------------------

    async def fetch_fills_paginated(
        self,
        address: str,
        client: "HLApiClient",  # type: ignore[name-defined]  # avoid circular import
        start_ms: int = 0,
    ) -> list[Fill]:
        """Fetch all fills for an address in 30-day windows from start_ms to now.

        HL's userFills endpoint caps at ~2000 results per call; windowing ensures
        we retrieve the complete history.
        """
        now_ms = int(time.time() * 1000)
        all_fills: list[Fill] = []
        seen_tids: set[int] = set()

        # Never scan before HL went live — avoids hundreds of empty API calls
        window_start = max(start_ms, _HL_GENESIS_MS)
        while window_start < now_ms:
            window_end = min(window_start + _THIRTY_DAYS_MS, now_ms)
            logger.debug(
                "Fetching fills for %s window [%d, %d]",
                address[:8], window_start, window_end,
            )
            batch = await client.get_user_fills(address, window_start, window_end)
            new_fills = [f for f in batch if f.tid not in seen_tids]
            for f in new_fills:
                seen_tids.add(f.tid)
            all_fills.extend(new_fills)
            logger.debug(
                "  → %d new fills (total so far: %d)", len(new_fills), len(all_fills)
            )

            # Stop once we reach the current epoch with no fills (normal end of history)
            if len(batch) == 0 and window_end >= now_ms - _THIRTY_DAYS_MS:
                break
            window_start = window_end

        all_fills.sort(key=lambda f: f.time)
        return all_fills

    # ------------------------------------------------------------------
    # Reconstruction for one address
    # ------------------------------------------------------------------

    async def reconstruct_one(
        self,
        address: str,
        client: "HLApiClient",  # type: ignore[name-defined]
    ) -> pl.DataFrame:
        """Replay fills for one address and return the position history as a DataFrame.

        Columns: [address, coin, size, entry_px, realized_pnl, fees, ts]
        """
        # 1. Load cached fills; determine start for incremental fetch
        cached_df = self._cache.load_fills(address)
        if cached_df is not None and len(cached_df) > 0:
            # Fetch only from the latest cached fill time onward
            start_ms = int(cached_df["time"].max() or 0) + 1  # type: ignore[arg-type]
            logger.debug(
                "reconstruct_one %s: incremental fetch from %d", address[:8], start_ms
            )
        else:
            start_ms = _HL_GENESIS_MS
            logger.debug(
                "reconstruct_one %s: full fetch from HL genesis", address[:8]
            )

        # 2. Fetch any new fills
        new_fills = await self.fetch_fills_paginated(address, client, start_ms)

        # 3. Save to cache
        if new_fills:
            self._cache.save_fills(address, new_fills)

        # 4. Reload full cached fills for replay
        fills_df = self._cache.load_fills(address)
        if fills_df is None or len(fills_df) == 0:
            logger.debug("reconstruct_one %s: no fills found", address[:8])
            return pl.DataFrame(schema={
                "address": pl.Utf8,
                "coin": pl.Utf8,
                "size": pl.Float64,
                "entry_px": pl.Float64,
                "realized_pnl": pl.Float64,
                "fees": pl.Float64,
                "ts": pl.Int64,
            })

        # 5. Convert to list of Fill-like dicts and replay chronologically
        #    Sort by time (should already be sorted, but ensure it)
        fills_df = fills_df.sort("time")

        # Build rows from fill replay
        positions: dict[str, _CoinPosition] = {}
        rows: list[dict] = []

        for row in fills_df.iter_rows(named=True):
            # Reconstruct a Fill-like object from the row dict
            fill = Fill(
                coin=row["coin"],
                px=row["px"],
                sz=row["sz"],
                side=row["side"],
                time=row["time"],
                startPosition=row["startPosition"],
                dir=row["dir"],
                closedPnl=row["closedPnl"],
                hash=row["hash"],
                oid=row["oid"],
                crossed=row["crossed"],
                fee=row["fee"],
                tid=row["tid"],
                feeToken=row.get("feeToken", "USDC"),
                liquidation=row.get("liquidation"),
            )

            coin = fill.coin
            if coin not in positions:
                positions[coin] = _CoinPosition()

            snapshots = _apply_fill(positions[coin], fill)
            for snap in snapshots:
                rows.append({
                    "address": address,
                    "coin": coin,
                    "size": snap["size"],
                    "entry_px": snap["entry_px"],
                    "realized_pnl": snap["realized_pnl"],
                    "fees": snap["fees"],
                    "ts": fill.time,
                })

        if not rows:
            return pl.DataFrame(schema={
                "address": pl.Utf8,
                "coin": pl.Utf8,
                "size": pl.Float64,
                "entry_px": pl.Float64,
                "realized_pnl": pl.Float64,
                "fees": pl.Float64,
                "ts": pl.Int64,
            })

        result = pl.DataFrame(rows)
        logger.debug(
            "reconstruct_one %s: %d snapshot rows across %d coins",
            address[:8], len(result), result["coin"].n_unique(),
        )
        return result

    # ------------------------------------------------------------------
    # Liquidation price calculation
    # ------------------------------------------------------------------

    def compute_liq_px(
        self,
        size: float,
        entry_px: float,
        leverage: int,
        lev_type: str,
    ) -> float | None:
        """Compute the isolated-margin liquidation price.

        Returns None for cross-margin positions (liq_px depends on portfolio).

        Formulas:
          Isolated long:  liq_px = entry_px * (1 - 1/leverage + MAINTENANCE_MARGIN)
          Isolated short: liq_px = entry_px * (1 + 1/leverage - MAINTENANCE_MARGIN)
        """
        if lev_type == "cross":
            return None

        if size == 0.0 or entry_px == 0.0 or leverage <= 0:
            return None

        if size > 0:
            # Long position
            liq_px = entry_px * (1.0 - 1.0 / leverage + MAINTENANCE_MARGIN)
        else:
            # Short position
            liq_px = entry_px * (1.0 + 1.0 / leverage - MAINTENANCE_MARGIN)

        return liq_px

    # ------------------------------------------------------------------
    # Reconstruct top-N addresses from leaderboard
    # ------------------------------------------------------------------

    async def reconstruct_top_n(
        self,
        client: "HLApiClient",  # type: ignore[name-defined]
        n: int = 500,
    ) -> pl.DataFrame:
        """Fetch leaderboard, reconstruct fills for top-N addresses, return combined DataFrame.

        Uses a bounded asyncio.Semaphore to cap concurrency at _RECONSTRUCT_CONCURRENCY.
        """
        logger.info("Fetching top addresses (target=%d)...", n)
        # Try leaderboard first; fall back to active-address collection via recentTrades
        addresses: list[str] = []
        raw_rows = await client.get_leaderboard()
        for entry in raw_rows:
            addr = (
                entry.get("ethAddress")
                or entry.get("user")
                or entry.get("address")
                or ""
            )
            if addr and addr.startswith("0x"):
                addresses.append(addr)
            if len(addresses) >= n:
                break

        if not addresses:
            logger.info("Leaderboard unavailable — collecting addresses via recentTrades")
            _ALL_COINS = [
                "BTC", "ETH", "SOL", "AVAX", "DOGE", "ARB", "OP",
                "SUI", "APT", "WIF", "PEPE", "BNB", "MATIC", "LINK",
                "ATOM", "FTM", "INJ", "SEI", "TIA", "JUP",
            ]
            addresses = await client.get_active_addresses(_ALL_COINS, per_coin=50)
            addresses = addresses[:n]

        logger.info(
            "reconstruct_top_n: got %d addresses (requested %d)",
            len(addresses), n,
        )

        sem = asyncio.Semaphore(_RECONSTRUCT_CONCURRENCY)

        async def _reconstruct_with_sem(addr: str) -> pl.DataFrame | None:
            async with sem:
                try:
                    return await self.reconstruct_one(addr, client)
                except Exception as exc:
                    logger.error("reconstruct_one failed for %s: %s", addr[:8], exc)
                    return None

        tasks = [_reconstruct_with_sem(addr) for addr in addresses]
        results = await asyncio.gather(*tasks)

        frames = [df for df in results if df is not None and len(df) > 0]
        successes = len(frames)
        failures = len(addresses) - successes

        logger.info(
            "reconstruct_top_n: %d/%d succeeded, %d failed",
            successes, len(addresses), failures,
        )

        if not frames:
            return pl.DataFrame(schema={
                "address": pl.Utf8,
                "coin": pl.Utf8,
                "size": pl.Float64,
                "entry_px": pl.Float64,
                "realized_pnl": pl.Float64,
                "fees": pl.Float64,
                "ts": pl.Int64,
            })

        combined = pl.concat(frames, how="diagonal")
        logger.info(
            "reconstruct_top_n: combined DataFrame has %d rows, %d unique addresses",
            len(combined),
            combined["address"].n_unique() if "address" in combined.columns else 0,
        )
        return combined

    # ------------------------------------------------------------------
    # Aggregate snapshot
    # ------------------------------------------------------------------

    def snapshot_aggregate(
        self,
        positions_df: pl.DataFrame,
        mids: dict[str, float],
        ts: int,
    ) -> list[Position]:
        """Compute current unrealized PnL and liq_px for all open positions.

        Takes the most recent position row per (address, coin) from positions_df,
        enriches with current prices and returns a list of Position objects.
        """
        if positions_df is None or len(positions_df) == 0:
            return []

        # Keep only the most recent snapshot per (address, coin)
        latest = (
            positions_df
            .sort("ts")
            .group_by(["address", "coin"])
            .last()
        )

        # Filter to non-zero positions only
        if "size" in latest.columns:
            latest = latest.filter(pl.col("size").abs() > 1e-12)

        result: list[Position] = []

        for row in latest.iter_rows(named=True):
            address = row["address"]
            coin = row["coin"]
            size = float(row["size"])
            entry_px = float(row.get("entry_px", 0.0) or 0.0)

            if size == 0.0 or coin not in mids:
                continue

            mid_px = mids[coin]
            if size > 0:
                unrealized_pnl = (mid_px - entry_px) * size
            else:
                unrealized_pnl = (entry_px - mid_px) * abs(size)

            # We don't have leverage or lev_type in the replay df;
            # use API-provided leverage if available in the row, else default
            leverage = int(row.get("leverage", 10) or 10)
            leverage_type = str(row.get("leverage_type", "cross") or "cross")
            margin_used = abs(size) * entry_px / leverage if leverage > 0 else 0.0

            liq_px = self.compute_liq_px(size, entry_px, leverage, leverage_type)

            result.append(
                Position(
                    address=address,
                    coin=coin,
                    size=size,
                    entry_px=entry_px,
                    liq_px=liq_px,
                    margin_used=margin_used,
                    unrealized_pnl=unrealized_pnl,
                    leverage=leverage,
                    leverage_type=leverage_type,
                    snapshot_ts=ts,
                )
            )

        logger.debug("snapshot_aggregate: %d open positions at ts=%d", len(result), ts)
        return result
