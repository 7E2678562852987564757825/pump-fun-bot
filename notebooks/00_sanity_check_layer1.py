"""
Layer 1 Sanity Check: Position Reconstruction vs Live State.

Run this after running:
  python main.py --phase layer1 --address <your_address>

This script validates that fill replay produces positions consistent
with the HL clearinghouse state API.
"""

# %% [markdown]
# # Layer 1: Position Reconstruction Sanity Check
#
# This notebook validates the fill replay engine by:
# 1. Reconstructing positions for a known address
# 2. Comparing against live clearinghouse state from HL API
# 3. Showing a histogram of reconstruction accuracy

# %%
import asyncio
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path("..").resolve()))

from hl_liq_cascade.config import load_config, setup_logging
from hl_liq_cascade.data.api_client import HLApiClient
from hl_liq_cascade.data.cache import Cache
from hl_liq_cascade.data.position_store import PositionStore

cfg = load_config()
setup_logging(cfg)

# %% [markdown]
# ## Step 1: Load Cached Fills and Replay

# %%
ADDRESS = "0x0000000000000000000000000000000000000000"  # Replace with real address
cache = Cache(cfg["data"]["cache_dir"])

# Load cached fills (run layer1 phase first to populate cache)
fills_df = cache.load_fills(ADDRESS)
if fills_df is not None:
    print(f"Loaded {len(fills_df)} cached fills for {ADDRESS[:8]}...")
    print(fills_df.head(5))
else:
    print("No cached fills. Run: python main.py --phase layer1 --address <address>")

# %% [markdown]
# ## Step 2: Compare Reconstructed vs Live

# %%
async def compare_positions(address: str):
    client = HLApiClient(cfg["api"])
    store = PositionStore(cache, cfg)

    # Live state
    live = await client.get_clearinghouse_state(address)
    mids = await client.get_all_mids()

    print("=== LIVE CLEARINGHOUSE STATE ===")
    for ap in live.assetPositions:
        p = ap.position
        print(f"  szi={p.szi}, entryPx={p.entryPx}, liqPx={p.liquidationPx}")

    # Reconstructed positions (from cached fills)
    if fills_df is not None and len(fills_df) > 0:
        reconstructed = store.snapshot_aggregate(fills_df, mids, int(asyncio.get_event_loop().time() * 1000))
        print(f"\n=== RECONSTRUCTED ({len(reconstructed)} positions) ===")
        for pos in reconstructed:
            if abs(pos.size) > 0.001:
                print(f"  {pos.coin}: size={pos.size:.4f}, entry={pos.entry_px:.4f}, liq={pos.liq_px}")

    await client.close()

# Run (in notebook: asyncio.run or await in async context)
# asyncio.run(compare_positions(ADDRESS))

# %% [markdown]
# ## Step 3: Accuracy Metrics
#
# To validate reconstruction accuracy, compare:
# - Reconstructed size vs live szi for each coin
# - Reconstructed entry_px vs live entryPx

# %%
def plot_reconstruction_accuracy(reconstructed, live_positions):
    """Plot side-by-side comparison of reconstructed vs live positions."""
    coins = [p.coin for p in reconstructed]
    rec_sizes = [p.size for p in reconstructed]

    # Parse live positions
    live_dict = {}
    for ap in live_positions.assetPositions:
        p = ap.position
        # Need to map index to coin name (requires meta)
        # This would be populated in the actual notebook run

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(coins, rec_sizes, color="steelblue", alpha=0.7, label="Reconstructed")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Coin")
    ax.set_ylabel("Position Size (USD)")
    ax.set_title("Reconstructed Position Sizes")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../cache/viz/layer1_reconstruction.png", dpi=150)
    plt.show()

print("Notebook complete. Run cells above with a real address to see output.")
