"""
Signal Isolation Analysis.

For each of the three signals (A, B, C), analyze in isolation:
- Score distribution over time
- Win rate when signal fires above threshold
- Correlation between signal and subsequent returns

Run after building liq maps and having some event data cached.
"""

# %% [markdown]
# # Signal Isolation Analysis
#
# Analyze each signal independently before combining.
# Critical: if a signal doesn't show edge here, it won't in the combined backtest.

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import sys

sys.path.insert(0, str(Path("..").resolve()))

from hl_liq_cascade.config import load_config
from hl_liq_cascade.data.cache import Cache

cfg = load_config()
cache = Cache(cfg["data"]["cache_dir"])

# %% [markdown]
# ## Signal A: Cascade Front-Run
#
# Tests: Does momentum toward a liq cluster predict price continuation?

# %%
from hl_liq_cascade.signals.cascade_frontrun import CascadeFrontrunSignal
from hl_liq_cascade.liq_map.builder import LiqMapBuilder

sig_a = CascadeFrontrunSignal(cfg["signals"]["cascade_frontrun"])
builder = LiqMapBuilder(cfg["liq_map"])

# Load candles for backtesting signal A in isolation
coins = ["SOL", "AVAX", "BTC", "ETH"]
signal_scores_a: dict[str, list[tuple[int, float]]] = {c: [] for c in coins}

# For each hourly candle, compute signal A score using the cached liq maps
import time
now_ts = int(time.time() * 1000)
start_ts = now_ts - 30 * 86_400_000  # 30 days

for coin in coins:
    candles = cache.load_candles(coin, "1h")
    if candles is None or len(candles) == 0:
        print(f"  {coin}: no candles")
        continue

    candles = candles.filter(pl.col("t") >= start_ts).sort("t")
    print(f"  {coin}: {len(candles)} candles")

    # Load liq maps
    liq_maps_df = cache.load_liq_maps(coin, start_ts, now_ts)
    if liq_maps_df is None:
        print(f"    No liq maps for {coin}")

# %% [markdown]
# ## Signal B: Post-Cascade Fade
#
# Tests: After large liq events, does price reverse?
# Key metric: Average return in next N minutes after cascade spike

# %%
from hl_liq_cascade.signals.postcascade_fade import PostCascadeFadeSignal
sig_b = PostCascadeFadeSignal(cfg["signals"]["postcascade_fade"])

# Analyze historical cascade events
# Load liq events from cache
liq_event_path = Path("../cache/liq_events")
if liq_event_path.exists():
    for path in liq_event_path.glob("*.parquet"):
        coin = path.stem
        df = pl.read_parquet(path)
        print(f"{coin}: {len(df)} liq events")

        # Find 5-min cascade windows
        if len(df) == 0:
            continue

        window_ms = 5 * 60_000
        cascade_windows = []
        for i in range(0, len(df), 10):
            row = df[i]
            ts_start = int(row["ts"][0])
            ts_end = ts_start + window_ms
            window_events = df.filter((pl.col("ts") >= ts_start) & (pl.col("ts") <= ts_end))
            total_notional = float(window_events["notional"].sum())
            cascade_windows.append((ts_start, total_notional))

        if cascade_windows:
            notionals = [w[1] for w in cascade_windows]
            p95 = np.percentile(notionals, 95)
            print(f"  95th percentile cascade (5min): ${p95:,.0f}")
            print(f"  Max cascade: ${max(notionals):,.0f}")
else:
    print("No liq event data. Run layer3 phase to capture liq events.")

# %% [markdown]
# ## Signal C: Squeeze Setup
#
# Tests: Does asymmetric OI + extreme funding predict mean reversion?

# %%
from hl_liq_cascade.signals.squeeze import SqueezeSignal
sig_c = SqueezeSignal(cfg["signals"]["squeeze"])

# Analyze funding rate history (available from HL funding endpoint)
# funding_path = cache.cache_dir / "funding"
# For now, show example score distribution

import random
random.seed(42)

# Synthetic demonstration of squeeze signal scoring
print("=== Squeeze Signal: Score Distribution ===")
print("(Using synthetic data — run layer1_scale first for real analysis)")

scores = []
for _ in range(1000):
    asymmetry = max(1.0, random.lognormvariate(1.0, 1.5))
    funding = random.gauss(0, 0.005)
    oi_growth = random.gauss(0.02, 0.05)
    # Signal triggers when asymmetry > 5, funding extreme, OI growing
    triggered = (asymmetry > 5 and abs(funding) > 0.01 and oi_growth > 0.05)
    if triggered:
        score = min(asymmetry / 15, 1.0) * (-1 if funding > 0 else 1)
        scores.append(score)

if scores:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=20, color="purple", alpha=0.7)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Signal C (Squeeze) — Score Distribution (synthetic)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../cache/viz/signal_c_scores.png", dpi=150)
    plt.show()

# %% [markdown]
# ## Correlation Between Signals
#
# Do the three signals fire together (bad — correlated alpha)
# or independently (good — diversified)?

# %%
# This would be populated with real signal scores from backtesting
# For now, show expected correlation structure:
print("""
Expected signal correlations (ideally uncorrelated):
  A vs B: moderate negative correlation
         (A fires going INTO cascade, B fires AFTER cascade)
  A vs C: low correlation (different triggers)
  B vs C: low correlation (B = event-driven, C = structural)

Run full backtest to compute empirical correlations.
""")
