"""
Hyperliquid Liquidation Cascade Strategy — CLI Orchestrator.

Usage:
  python main.py --phase layer1 --address 0xabc...          # PoC: reconstruct one address
  python main.py --phase layer1_scale                        # Scale to top 500 addresses
  python main.py --phase layer2 --coins SOL AVAX            # Build and visualize liq maps
  python main.py --phase layer3 --coins SOL                 # Liq event stream + validation
  python main.py --phase backtest                            # Full backtest
  python main.py --phase live --coins BTC ETH SOL           # Paper trading (no real orders)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from hl_liq_cascade.config import load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

async def run_layer1_poc(cfg: dict, address: str) -> None:
    """Phase 0 Step 1: reconstruct one address and sanity-check vs live state."""
    from hl_liq_cascade.data.api_client import HLApiClient
    from hl_liq_cascade.data.cache import Cache
    from hl_liq_cascade.data.position_store import PositionStore

    client = HLApiClient(cfg["api"])
    cache = Cache(cfg["data"]["cache_dir"])
    store = PositionStore(cache, cfg)

    print(f"\n{'='*60}")
    print(f"Layer 1 PoC — Reconstructing: {address}")
    print(f"{'='*60}")

    # Reconstruct positions via fill replay
    positions_df = await store.reconstruct_one(address, client)
    print(f"\nReconstructed {len(positions_df)} position snapshots")
    print(positions_df.tail(5))

    # Compare against live clearinghouse state
    print("\n--- Live clearinghouse state ---")
    live_state = await client.get_clearinghouse_state(address)
    mids = await client.get_all_mids()

    print(f"Live account value: ${live_state.crossMarginSummary.accountValue}")
    for ap in live_state.assetPositions:
        p = ap.position
        coin = None
        # coin name is the asset name — need to look up from meta
        print(f"  Position: szi={p.szi}, entry={p.entryPx}, liq={p.liquidationPx}")

    # Snapshot aggregate at current prices
    final_positions = store.snapshot_aggregate(positions_df, mids, int(asyncio.get_event_loop().time() * 1000))
    print(f"\nReconstructed {len(final_positions)} open positions")
    for pos in final_positions:
        print(f"  {pos.coin}: size={pos.size:.4f}, entry={pos.entry_px:.4f}, liq={pos.liq_px}")

    await client.close()


async def run_layer1_scale(cfg: dict) -> None:
    """Phase 0 Step 2: scale to top 500 addresses, show OI coverage."""
    import polars as pl
    from hl_liq_cascade.data.api_client import HLApiClient
    from hl_liq_cascade.data.cache import Cache
    from hl_liq_cascade.data.position_store import PositionStore
    from hl_liq_cascade.analytics.viz import plot_oi_coverage

    client = HLApiClient(cfg["api"])
    cache = Cache(cfg["data"]["cache_dir"])
    store = PositionStore(cache, cfg)

    n = cfg["data"]["top_n_addresses"]
    print(f"\n{'='*60}")
    print(f"Layer 1 Scale — Top {n} addresses")
    print(f"{'='*60}")

    positions_df = await store.reconstruct_top_n(client, n=n)
    print(f"\nTotal reconstructed rows: {len(positions_df)}")

    mids = await client.get_all_mids()
    ts = int(asyncio.get_event_loop().time() * 1000)

    all_positions = store.snapshot_aggregate(positions_df, mids, ts)
    print(f"Total open positions: {len(all_positions)}")

    # OI coverage analysis
    meta_list, ctx_list = await client.get_meta()
    for i, (meta, ctx) in enumerate(zip(meta_list, ctx_list)):
        coin = meta.name
        total_oi = float(ctx.openInterest) * float(ctx.markPx)
        our_long = sum(abs(p.size) * p.entry_px for p in all_positions if p.coin == coin and p.size > 0)
        our_short = sum(abs(p.size) * p.entry_px for p in all_positions if p.coin == coin and p.size < 0)
        if total_oi > 0:
            coverage = (our_long + our_short) / total_oi
            if coverage > 0.01:
                print(f"  {coin}: total_OI=${total_oi:,.0f}, captured={coverage:.1%}")

    await client.close()


async def run_layer2(cfg: dict, coins: list[str]) -> None:
    """Phase 0 Step 3: build and visualize liq maps."""
    import polars as pl
    from hl_liq_cascade.data.api_client import HLApiClient
    from hl_liq_cascade.data.cache import Cache
    from hl_liq_cascade.data.position_store import PositionStore
    from hl_liq_cascade.liq_map.builder import LiqMapBuilder
    from hl_liq_cascade.liq_map.cluster import ClusterDetector
    from hl_liq_cascade.analytics.viz import plot_liq_map

    client = HLApiClient(cfg["api"])
    cache = Cache(cfg["data"]["cache_dir"])
    store = PositionStore(cache, cfg)
    builder = LiqMapBuilder(cfg["liq_map"])
    detector = ClusterDetector(cfg["liq_map"])

    print(f"\n{'='*60}")
    print(f"Layer 2 — Liquidation Maps for: {coins}")
    print(f"{'='*60}")

    mids = await client.get_all_mids()
    ts = int(asyncio.get_event_loop().time() * 1000)

    # Load cached positions
    positions_df = cache.load_position_snapshots(ts - 86_400_000, ts)
    if positions_df is None or len(positions_df) == 0:
        print("No cached positions. Run layer1_scale first.")
        await client.close()
        return

    latest = positions_df.filter(pl.col("snapshot_ts") == positions_df["snapshot_ts"].max())
    all_positions = store.snapshot_aggregate(latest, mids, ts)

    for coin in coins:
        if coin not in mids:
            print(f"  {coin}: not in mids, skipping")
            continue

        liq_maps = builder.build(all_positions, mids, ts)
        if coin not in liq_maps:
            print(f"  {coin}: no positions found, skipping")
            continue

        lm = liq_maps[coin]
        density = builder.compute_density(lm)
        clusters = detector.detect(lm)

        print(f"\n{coin}:")
        print(f"  Current price: ${lm.current_px:,.4f}")
        print(f"  Total long notional at risk (±5%): ${density['long_density_near']:,.0f}")
        print(f"  Total short notional at risk (±5%): ${density['short_density_near']:,.0f}")
        print(f"  Imbalance: {density['liq_imbalance_direction']} ({density['imbalance_ratio']:.2f}x)")
        print(f"  Top clusters: {len(clusters)}")
        for c in clusters[:3]:
            print(f"    {c.side} ${c.price:,.4f} ({c.distance_pct:+.2%}) — ${c.notional:,.0f}")

        # Load price history for chart
        candles = cache.load_candles(coin, "1h")
        price_history: list[tuple[int, float]] = []
        if candles is not None:
            rows = candles.select(["t", "c"]).tail(168).to_dicts()
            price_history = [(r["t"], float(r["c"])) for r in rows]

        out_path = f"cache/viz/liq_map_{coin}_{ts}.png"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plot_liq_map(lm, price_history, coin, out_path=out_path)
        print(f"  Chart saved: {out_path}")

    await client.close()


async def run_layer3(cfg: dict, coins: list[str]) -> None:
    """Phase 0 Step 4: capture liq event stream and validate vs liq map clusters."""
    import json
    from hl_liq_cascade.data.api_client import HLApiClient
    from hl_liq_cascade.data.ws_client import HLWebSocketClient
    from hl_liq_cascade.types import Event, EventKind, LiqEvent

    print(f"\n{'='*60}")
    print(f"Layer 3 — Liquidation Event Stream for: {coins}")
    print(f"{'='*60}")
    print("Listening for 60 seconds... (Ctrl+C to stop early)\n")

    liq_events: list[LiqEvent] = []

    async def on_event(event: Event) -> None:
        if event.kind == EventKind.LIQ_EVENT:
            ev: LiqEvent = event.payload  # type: ignore
            liq_events.append(ev)
            print(f"  [{ev.coin}] {ev.side.upper()} LIQ: {ev.size:.4f} @ ${ev.px:,.4f} = ${ev.notional:,.0f}")

    client = HLApiClient(cfg["api"])
    ws = HLWebSocketClient(cfg["api"], on_event=on_event)
    await ws.run(coins)

    await client.close()


async def run_backtest(cfg: dict) -> None:
    """Phase 5-6: full historical backtest."""
    import polars as pl
    from datetime import datetime
    from hl_liq_cascade.data.api_client import HLApiClient
    from hl_liq_cascade.data.cache import Cache
    from hl_liq_cascade.liq_map.builder import LiqMapBuilder
    from hl_liq_cascade.backtest.event_bus import EventBus
    from hl_liq_cascade.backtest.data_loader import BacktestDataLoader
    from hl_liq_cascade.backtest.engine import BacktestEngine
    from hl_liq_cascade.analytics.metrics import PerformanceMetrics
    from hl_liq_cascade.analytics.attribution import SignalAttributor
    from hl_liq_cascade.analytics.viz import plot_equity_curve, plot_trade_analysis

    print(f"\n{'='*60}")
    print("Full Backtest")
    print(f"{'='*60}")

    bt_cfg = cfg["backtest"]
    start_dt = datetime.strptime(bt_cfg["start_date"], "%Y-%m-%d")
    end_dt = datetime.strptime(bt_cfg["end_date"], "%Y-%m-%d")
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    oos_ts = int(start_ts + (end_ts - start_ts) * bt_cfg["oos_split"])

    print(f"Period: {bt_cfg['start_date']} → {bt_cfg['end_date']}")
    print(f"IS: {bt_cfg['start_date']} → OOS boundary")
    print(f"OOS: {bt_cfg['oos_split']*100:.0f}% of period (last 25%)")
    print(f"Initial capital: ${bt_cfg['initial_capital']:,.0f}")

    cache = Cache(cfg["data"]["cache_dir"])
    builder = LiqMapBuilder(cfg["liq_map"])
    loader = BacktestDataLoader(cache, builder, cfg)

    # Get available coins from cached candles
    coins = ["BTC", "ETH", "SOL", "AVAX", "DOGE", "ARB", "OP", "SUI", "APT", "WIF"]
    print(f"\nCoins: {coins}")

    print("\nLoading events into bus...")
    bus = loader.load_events(coins, start_ts, end_ts)
    print(f"Bus loaded: {bus.size()} events")

    engine = BacktestEngine(cfg, initial_capital=bt_cfg["initial_capital"])
    print("\nRunning simulation...")
    trades = engine.run(bus)
    equity = engine.equity_curve()

    print(f"\nCompleted. {len(trades)} trades over {len(equity)} equity events.")

    metrics_calc = PerformanceMetrics()
    metrics = metrics_calc.compute_all(trades, equity, bt_cfg["initial_capital"])
    metrics_calc.print_report(metrics)

    attributor = SignalAttributor()
    attribution = attributor.compute_attribution(trades)
    coin_attr = attributor.coin_attribution(trades)

    print("\n--- Signal Attribution ---")
    for sig, stats in attribution.items():
        print(f"  {sig}: PnL=${stats['total_pnl']:,.0f}, WR={stats['win_rate']:.1%}, trades={stats['trade_count']}")

    print("\n--- Per-Coin PnL (top 5) ---")
    sorted_coins = sorted(coin_attr.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
    for coin, stats in sorted_coins[:5]:
        print(f"  {coin}: ${stats['total_pnl']:,.0f}")

    # Save outputs
    Path("cache/results").mkdir(parents=True, exist_ok=True)
    trades.write_parquet("cache/results/trades.parquet")
    equity.write_parquet("cache/results/equity.parquet")

    plot_equity_curve(equity, metrics, out_path="cache/results/equity_curve.png")
    plot_trade_analysis(trades, out_path="cache/results/trade_analysis.png")
    print("\nCharts saved to cache/results/")


async def run_live(cfg: dict, coins: list[str]) -> None:
    """Paper trading mode — no real orders."""
    from hl_liq_cascade.live.paper_trade import PaperTrader

    print(f"\n{'='*60}")
    print(f"PAPER TRADING — {coins}")
    print("No real orders will be placed.")
    print(f"{'='*60}\n")

    trader = PaperTrader(cfg)
    await trader.run(coins)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hyperliquid Liquidation Cascade Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--phase", required=True, choices=[
        "layer1", "layer1_scale", "layer2", "layer3", "backtest", "live"
    ])
    p.add_argument("--config", default=None, help="Path to YAML config (default: built-in)")
    p.add_argument("--address", default=None, help="HL address for layer1 PoC")
    p.add_argument("--coins", nargs="+", default=["SOL", "AVAX"], help="Coins to process")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    phase = args.phase

    if phase == "layer1":
        if not args.address:
            print("--address required for layer1 phase")
            sys.exit(1)
        asyncio.run(run_layer1_poc(cfg, args.address))

    elif phase == "layer1_scale":
        asyncio.run(run_layer1_scale(cfg))

    elif phase == "layer2":
        asyncio.run(run_layer2(cfg, args.coins))

    elif phase == "layer3":
        asyncio.run(run_layer3(cfg, args.coins))

    elif phase == "backtest":
        asyncio.run(run_backtest(cfg))

    elif phase == "live":
        asyncio.run(run_live(cfg, args.coins))


if __name__ == "__main__":
    main()
