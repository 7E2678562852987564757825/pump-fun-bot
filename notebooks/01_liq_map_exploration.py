"""
Layer 2: Liquidation Map Exploration.

Run after layer1_scale to have position snapshots cached.

Visualizes liq maps for SOL and AVAX, validates against price action.
"""

# %% [markdown]
# # Layer 2: Liquidation Map Exploration
#
# Validates the liq map computation and cluster detection.
# Shows liq density vs price action for SOL and AVAX.

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

sys.path.insert(0, str(Path("..").resolve()))

from hl_liq_cascade.config import load_config
from hl_liq_cascade.data.cache import Cache
from hl_liq_cascade.data.position_store import PositionStore
from hl_liq_cascade.liq_map.builder import LiqMapBuilder
from hl_liq_cascade.liq_map.cluster import ClusterDetector
from hl_liq_cascade.analytics.viz import plot_liq_map, interactive_liq_map

cfg = load_config()
cache = Cache(cfg["data"]["cache_dir"])
builder = LiqMapBuilder(cfg["liq_map"])
detector = ClusterDetector(cfg["liq_map"])

# %% [markdown]
# ## Load Latest Position Snapshot

# %%
import time
now_ts = int(time.time() * 1000)
snapshots = cache.load_position_snapshots(now_ts - 86_400_000, now_ts)

if snapshots is None or len(snapshots) == 0:
    print("No snapshots. Run: python main.py --phase layer1_scale")
else:
    print(f"Loaded {len(snapshots)} position rows")
    print(f"Unique addresses: {snapshots['address'].n_unique()}")
    print(f"Unique coins: {snapshots['coin'].n_unique()}")
    print(snapshots.head(5))

# %% [markdown]
# ## Build Liq Maps for SOL and AVAX

# %%
coins_to_analyze = ["SOL", "AVAX"]
# Mock mids for demo (replace with API call)
mids = {"SOL": 140.0, "AVAX": 38.0}

if snapshots is not None and len(snapshots) > 0:
    # Get latest snapshot
    latest_ts = snapshots["snapshot_ts"].max()
    latest = snapshots.filter(pl.col("snapshot_ts") == latest_ts)

    from hl_liq_cascade.types import Position
    positions = []
    for row in latest.to_dicts():
        if row.get("liq_px") and abs(row["size"]) > 0:
            positions.append(Position(
                address=row["address"],
                coin=row["coin"],
                size=row["size"],
                entry_px=row["entry_px"],
                liq_px=row["liq_px"],
                margin_used=0.0,
                unrealized_pnl=0.0,
                leverage=10,
                leverage_type="isolated",
                snapshot_ts=int(latest_ts),
            ))

    liq_maps = builder.build(positions, mids, int(latest_ts))
    print(f"Built liq maps for: {list(liq_maps.keys())}")
else:
    print("No positions to build liq maps from")
    liq_maps = {}

# %% [markdown]
# ## Visualize: SOL Liq Map

# %%
if "SOL" in liq_maps:
    lm = liq_maps["SOL"]
    density = builder.compute_density(lm)
    clusters = detector.detect(lm)

    print(f"SOL @ ${lm.current_px:.2f}")
    print(f"  Long liq near (±0.5%): ${density['long_density_near']:,.0f}")
    print(f"  Short liq near (±0.5%): ${density['short_density_near']:,.0f}")
    print(f"  Imbalance: {density['liq_imbalance_direction']} ({density['imbalance_ratio']:.2f}x)")
    print(f"  Clusters: {len(clusters)}")
    for c in clusters[:5]:
        print(f"    {c.side} @ ${c.price:.2f} ({c.distance_pct:+.2%}) ${c.notional:,.0f} score={c.density_score:.3f}")

    candles = cache.load_candles("SOL", "1h")
    price_history = []
    if candles is not None:
        rows = candles.tail(168).select(["t", "c"]).to_dicts()
        price_history = [(r["t"], float(r["c"])) for r in rows]

    Path("../cache/viz").mkdir(parents=True, exist_ok=True)
    plot_liq_map(lm, price_history, "SOL", out_path="../cache/viz/liq_map_SOL.png")
    print("Chart saved: ../cache/viz/liq_map_SOL.png")

# %% [markdown]
# ## Cluster Detection Sensitivity Analysis
#
# How does the number of detected clusters change with different
# minimum notional thresholds?

# %%
if "SOL" in liq_maps:
    lm = liq_maps["SOL"]
    thresholds = [50_000, 100_000, 250_000, 500_000, 1_000_000]
    cluster_counts = []
    for thresh in thresholds:
        from hl_liq_cascade.liq_map.cluster import ClusterDetector
        det = ClusterDetector({"cluster_min_notional_usd": thresh})
        clusters = det.detect(lm)
        cluster_counts.append(len(clusters))
        print(f"  Min notional ${thresh:,.0f}: {len(clusters)} clusters")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([t / 1e6 for t in thresholds], cluster_counts, "o-", color="steelblue")
    ax.set_xlabel("Min Notional Threshold (USD M)")
    ax.set_ylabel("Detected Clusters")
    ax.set_title("SOL: Cluster Count vs Detection Threshold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../cache/viz/cluster_sensitivity.png", dpi=150)
    plt.show()

# %% [markdown]
# ## Asymmetry Analysis: Are liq maps skewed?

# %%
if liq_maps:
    print("=== Liq Map Asymmetry by Coin ===")
    for coin, lm in liq_maps.items():
        ratio, dominant = detector.score_asymmetry(lm, depth_pct=0.02)
        print(f"  {coin}: ratio={ratio:.2f}x dominant={dominant}")
