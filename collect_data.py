"""
Standalone data collection script — fills the cache for phases 1-3.

Runs sequentially:
  1. Fetch candles for major coins (12 months, 1h)
  2. Collect active addresses via recentTrades
  3. Reconstruct positions for each address (fill replay)
  4. Save hourly liq map snapshots
  5. Brief WebSocket capture for live liq events

Usage:
  python collect_data.py [--coins BTC ETH SOL AVAX] [--max-addresses 200]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from hl_liq_cascade.config import load_config, setup_logging
from hl_liq_cascade.data.api_client import HLApiClient
from hl_liq_cascade.data.cache import Cache
from hl_liq_cascade.data.position_store import PositionStore
from hl_liq_cascade.liq_map.builder import LiqMapBuilder
from hl_liq_cascade.liq_map.cluster import ClusterDetector
from hl_liq_cascade.types import Position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 — Candle fetch (prerequisite for liq map + backtest)
# ---------------------------------------------------------------------------

async def fetch_candles(
    client: HLApiClient,
    cache: Cache,
    coins: list[str],
    months: int = 12,
    interval: str = "1h",
) -> None:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - months * 30 * 24 * 3_600_000
    chunk_ms = 30 * 24 * 3_600_000  # 30-day chunks

    for coin in coins:
        existing = cache.load_candles(coin, interval)
        if existing is not None and len(existing) > 0:
            last_t = int(existing["t"].max() or 0)  # type: ignore[arg-type]
            if now_ms - last_t < 2 * 3_600_000:  # cached within 2h → skip
                print(f"  {coin}: candles already up to date ({len(existing)} rows)")
                continue
            fetch_from = last_t + 1
        else:
            fetch_from = start_ms

        all_candles = []
        t = fetch_from
        while t < now_ms:
            t_end = min(t + chunk_ms, now_ms)
            try:
                batch = await client.get_candles(coin, interval, t, t_end)
                all_candles.extend(batch)
                print(f"  {coin}: [{_fmt(t)}→{_fmt(t_end)}] {len(batch)} candles")
            except Exception as exc:
                logger.warning("Candle fetch failed for %s: %s", coin, exc)
            t = t_end

        if all_candles:
            rows = [
                {
                    "t": c.t, "T": c.T, "s": c.s, "i": c.i,
                    "o": c.o, "h": c.h, "l": c.l, "c": c.c,
                    "v": c.v, "n": c.n,
                }
                for c in all_candles
            ]
            df = pl.DataFrame(rows)
            cache.save_candles(coin, interval, df)
            print(f"  {coin}: saved {len(df)} candles total")


# ---------------------------------------------------------------------------
# Phase 2 — Address collection + position reconstruction
# ---------------------------------------------------------------------------

async def fetch_current_positions(
    client: HLApiClient,
    cache: Cache,
    coins: list[str],
    max_addresses: int = 200,
    ts: int | None = None,
) -> list[Position]:
    """Phase 2 (fast path): snapshot current positions via clearinghouseState.

    1 API call per address → no fill-replay needed.
    Returns actual liq_px from the API (including cross-margin account state).
    """
    if ts is None:
        ts = int(time.time() * 1000)

    print(f"\n--- Collecting active addresses from recentTrades ({len(coins)} coins) ---")
    addresses = await client.get_active_addresses(coins, per_coin=50)
    addresses = addresses[:max_addresses]
    print(f"  → {len(addresses)} unique addresses")

    mids = await client.get_all_mids()

    print(f"\n--- Fetching current positions via clearinghouseState ({len(addresses)} addresses) ---")
    positions: list[Position] = []

    for i, addr in enumerate(addresses):
        try:
            state = await client.get_clearinghouse_state(addr)
            n_before = len(positions)
            for ap in state.assetPositions:
                pd_data = ap.position
                coin = pd_data.coin
                size = float(pd_data.szi)
                if abs(size) < 1e-12 or coin not in mids:
                    continue
                entry_px = float(pd_data.entryPx or 0)
                if entry_px == 0:
                    continue
                liq_px = float(pd_data.liquidationPx) if pd_data.liquidationPx else None
                leverage = int(pd_data.leverage.get("value", 10)) if isinstance(pd_data.leverage, dict) else 10
                lev_type = str(pd_data.leverage.get("type", "cross")) if isinstance(pd_data.leverage, dict) else "cross"
                positions.append(Position(
                    address=addr,
                    coin=coin,
                    size=size,
                    entry_px=entry_px,
                    liq_px=liq_px,
                    margin_used=float(pd_data.marginUsed),
                    unrealized_pnl=float(pd_data.unrealizedPnl),
                    leverage=leverage,
                    leverage_type=lev_type,
                    snapshot_ts=ts,
                ))
            n_added = len(positions) - n_before
            if n_added > 0:
                sys.stdout.write(".")
                sys.stdout.flush()
        except Exception as exc:
            logger.debug("clearinghouseState %s: %s", addr[:8], exc)

    print(f"\n  → {len(positions)} open positions from {len(addresses)} addresses")
    # Cache snapshot
    if positions:
        cache.save_position_snapshot(ts, positions)
    return positions


async def reconstruct_positions(
    client: HLApiClient,
    cache: Cache,
    store: PositionStore,
    coins: list[str],
    max_addresses: int = 200,
) -> pl.DataFrame:
    """Slow path: full fill-replay for historical position reconstruction.

    Makes ~42 API calls per address (one 30-day window each).
    Use only when historical liq map snapshots are needed.
    """
    print(f"\n--- Collecting active addresses from recentTrades ({len(coins)} coins) ---")
    addresses = await client.get_active_addresses(coins, per_coin=50)
    addresses = addresses[:max_addresses]
    print(f"  → {len(addresses)} unique addresses")

    print(f"\n--- Reconstructing positions for {len(addresses)} addresses ---")
    sem = asyncio.Semaphore(1)  # sequential: one address at a time, no burst risk

    async def _one(addr: str) -> pl.DataFrame | None:
        async with sem:
            try:
                df = await store.reconstruct_one(addr, client)
                if len(df) > 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                return df
            except Exception as exc:
                logger.debug("reconstruct_one %s: %s", addr[:8], exc)
                return None

    results = await asyncio.gather(*[_one(a) for a in addresses])
    print()

    frames = [df for df in results if df is not None and len(df) > 0]
    if not frames:
        print("  No positions reconstructed")
        return pl.DataFrame()

    combined = pl.concat(frames, how="diagonal")
    print(f"  → {len(combined)} fill-replay rows, {combined['address'].n_unique()} addresses")
    return combined


# ---------------------------------------------------------------------------
# Phase 3 — Snapshot aggregate + liq maps
# ---------------------------------------------------------------------------

async def build_liq_maps(
    client: HLApiClient,
    cache: Cache,
    positions: list[Position],
    builder: LiqMapBuilder,
    coins: list[str],
) -> None:
    print("\n--- Building liq map snapshot ---")
    mids = await client.get_all_mids()
    ts = int(time.time() * 1000)

    if not positions:
        print("  No positions available for liq map")
        return

    # Filter to positions with valid liq_px
    valid = [p for p in positions if p.liq_px is not None and p.liq_px > 0]
    print(f"  {len(valid)} positions with liq_px across {len(set(p.coin for p in valid))} coins")

    # Build and display liq maps
    liq_maps = builder.build(valid, mids, ts)
    detector = ClusterDetector({"cluster_min_notional_usd": 100_000})

    Path("cache/viz").mkdir(parents=True, exist_ok=True)

    for coin in coins:
        if coin not in liq_maps:
            continue
        lm = liq_maps[coin]
        density = builder.compute_density(lm)
        clusters = detector.detect(lm)

        print(f"\n  {coin} @ ${lm.current_px:,.4f}")
        print(f"    Long liq near (±0.5%): ${density['long_density_near']:>12,.0f}")
        print(f"    Short liq near (±0.5%): ${density['short_density_near']:>11,.0f}")
        print(f"    Imbalance: {density['liq_imbalance_direction']} ({density['imbalance_ratio']:.1f}x)")
        print(f"    Clusters detected: {len(clusters)}")
        for c in clusters[:3]:
            print(f"      [{c.side}] ${c.price:,.4f} ({c.distance_pct:+.2%}) ${c.notional:,.0f}")

        # Save liq map as parquet
        lm_df = builder.to_dataframe(lm)
        lm_df = lm_df.with_columns([
            pl.lit(coin).alias("coin"),
            pl.lit(ts).alias("ts"),
            pl.lit(lm.current_px).alias("current_px"),
        ])
        cache.save_liq_map(coin, ts, lm_df)

        # Save chart
        try:
            from hl_liq_cascade.analytics.viz import plot_liq_map
            candles = cache.load_candles(coin, "1h")
            price_history: list[tuple[int, float]] = []
            if candles is not None and len(candles) > 0:
                price_history = [
                    (int(r["t"]), float(r["c"]))
                    for r in candles.tail(168).select(["t", "c"]).to_dicts()
                ]
            out = f"cache/viz/liq_map_{coin}.png"
            plot_liq_map(lm, price_history, coin, out_path=out)
            print(f"    Chart → {out}")
        except Exception as exc:
            logger.warning("Chart failed for %s: %s", coin, exc)


# ---------------------------------------------------------------------------
# OI coverage report
# ---------------------------------------------------------------------------

async def oi_coverage_report(
    client: HLApiClient,
    positions: list[Position],
) -> None:
    print("\n--- OI Coverage Report ---")
    meta_list, ctx_list = await client.get_meta()
    mids = await client.get_all_mids()

    total_captured = 0.0
    total_on_chain = 0.0

    for meta, ctx in zip(meta_list, ctx_list):
        coin = meta.name
        if coin not in mids:
            continue
        try:
            total_oi = float(ctx.openInterest) * float(mids[coin])
        except Exception:
            continue
        if total_oi < 1_000_000:  # skip tiny markets
            continue

        our_long = sum(abs(p.size) * p.entry_px for p in positions if p.coin == coin and p.size > 0)
        our_short = sum(abs(p.size) * p.entry_px for p in positions if p.coin == coin and p.size < 0)
        captured = our_long + our_short
        pct = captured / total_oi if total_oi > 0 else 0

        total_captured += captured
        total_on_chain += total_oi

        if pct > 0.001:
            bar = "█" * int(pct * 50) + "░" * max(0, 50 - int(pct * 50))
            print(f"  {coin:8s} {bar} {pct:6.1%}  (${captured/1e6:.1f}M / ${total_oi/1e6:.1f}M)")

    overall = total_captured / total_on_chain if total_on_chain > 0 else 0
    print(f"\n  Overall capture: {overall:.1%} of total on-chain OI (${total_captured/1e6:.1f}M / ${total_on_chain/1e6:.1f}M)")
    print(f"  Note: cross-margin liq prices excluded → underestimates true coverage")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    cfg = load_config()
    setup_logging(cfg)

    coins = args.coins
    max_addr = args.max_addresses

    print(f"\n{'='*60}")
    print(f"HL Liquidation Cascade — Data Collection")
    print(f"Coins: {coins}")
    print(f"Max addresses: {max_addr}")
    print(f"{'='*60}\n")

    cache = Cache(cfg["data"]["cache_dir"])
    client = HLApiClient(cfg["api"])
    store = PositionStore(cache, cfg)
    builder = LiqMapBuilder(cfg["liq_map"])

    # ── Phase 0: verify connectivity ───────────────────────────────────────
    print("--- Connectivity check ---")
    mids = await client.get_all_mids()
    print(f"  ✓ API reachable — {len(mids)} coins. BTC mid: ${mids.get('BTC', 0):,.0f}")

    # ── Phase 1: candles ────────────────────────────────────────────────────
    print(f"\n--- Phase 1: Fetching {args.months}m of 1h candles for {len(coins)} coins ---")
    await fetch_candles(client, cache, coins, months=args.months)

    # ── Phase 2: current positions via clearinghouseState (1 call/address) ──
    ts_now = int(time.time() * 1000)
    positions = await fetch_current_positions(client, cache, coins, max_addr, ts_now)

    # ── Phase 3: liq maps ───────────────────────────────────────────────────
    await build_liq_maps(client, cache, positions, builder, coins)

    # ── Coverage report ─────────────────────────────────────────────────────
    if positions:
        await oi_coverage_report(client, positions)

    await client.close()
    print(f"\n{'='*60}")
    print("Data collection complete. Cache at: cache/")
    print("Next: python main.py --phase backtest")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill the HL liq cascade cache")
    p.add_argument("--coins", nargs="+",
                   default=["BTC", "ETH", "SOL", "AVAX", "DOGE", "ARB", "SUI", "WIF", "PEPE", "INJ"])
    p.add_argument("--months", type=int, default=3,
                   help="Months of candle history to fetch (default 3, max 12)")
    p.add_argument("--max-addresses", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
