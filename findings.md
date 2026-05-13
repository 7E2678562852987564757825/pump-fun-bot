# Liquidation Cascade Strategy — Findings

**Status**: Framework complete. Findings template below — to be filled in after running full backtest.

---

## Framework Architecture

Built a strict event-bus driven backtest with three independent signal modules:
- **Signal A** (cascade_frontrun): Front-run dense liq clusters by momentum direction
- **Signal B** (postcascade_fade): Mean-revert after cascade exhaustion
- **Signal C** (squeeze): Position against crowded/extreme one-sided OI

The backtest enforces temporal correctness at the EventBus level: `pop_until(ts)` is the sole entry point for event consumers, and `push()` rejects any event with ts < current_ts.

---

## Phase 0 Results (to be filled in after data collection)

### Layer 1: Position Reconstruction

- Addresses reconstructed: _TBD_ (target: top 500 by volume)
- OI coverage: _TBD_ (expect 60–80% of total on-chain OI)
- Known bias: excludes cross-margin complex portfolios where liq_px depends on total portfolio; these are treated as no liq_px and excluded from the map. This underestimates long-side liq risk in a falling market when cross-margin traders are underwater.

### Layer 2: Liq Map Validation

- Coins analyzed: SOL, AVAX, BTC, ETH
- Validation: _TBD_ — do large cascade events correspond to dense pre-event clusters?
- Expected finding: Yes on major market moves; weaker signal on overnight low-volume liquidations

### Layer 3: Liq Event Stream

- Capture rate: _TBD_% of on-chain liquidations (limited by WebSocket reliability)
- 95th percentile cascade notional (5-min window): _TBD_ by coin

---

## What Works (Hypothesis — to be validated)

1. **Signal A on BTC/ETH**: Large, dense liq clusters below/above market are observable hours before they trigger. Price momentum toward a cluster is a strong leading indicator on liquid markets.

2. **Signal B**: Post-cascade fades appear robust on liquid coins (BTC, ETH) where there's enough market depth to absorb the initial cascade and allow mean reversion.

3. **Signal C**: Extreme funding rates + asymmetric OI tends to precede short squeezes on alt coins where the market is thinner.

---

## What Doesn't Work (Hypothesis — to be validated)

1. **Signal A on micro-caps**: Liq maps for thin-market coins have high noise relative to signal. Cluster detection fires too frequently, signal quality degrades.

2. **Signal B in trending markets**: Post-cascade fades fail when the cascade is the START of a trend rather than an exhaustion. The strategy needs a regime filter (e.g., macro BTC trend direction).

3. **Signal C timing**: The "when" of the squeeze is hard to predict. The signal may be correct directionally but early by hours, burning on time stops.

---

## Capacity Estimate

This is a flow-based strategy that trades against forced selling/buying. Capacity is limited by:
1. How much notional we can put on without moving the market before the cascade
2. Cascade size — can't front-run a $500k cascade with $200k position

Estimated capacity: **$500k–$2M USD** for the combined strategy, per coin.
- Above $2M: our orders start to create anticipatory price moves, reducing the edge
- Above $5M: we ARE the market, the cascade may not materialize as predicted

---

## Failure Modes in Live

1. **Data latency**: If position snapshots are stale (>10 min), liq map clusters are wrong. The strategy degrades gracefully (no false signals) but misses real ones.

2. **API outages**: HL WebSocket is occasionally unreliable. Need robust reconnection (implemented). During outages, signal B becomes blind to cascade events.

3. **Map-to-reality gap**: Cross-margin traders don't have deterministic liq prices. Our map understates their liq risk. A large whale on cross margin with low equity can liquidate at any price level.

4. **Coordinated attacks**: If large players know our liq map (it's public), they can front-run our front-run by placing orders just ahead of the cluster. This is unlikely at current AUM but worth monitoring.

5. **Flash crashes**: A liquidation cascade in under 1 second doesn't give our 200ms execution model time to participate in Signal A. We mostly capture the 2nd-order cascade, not the initial impulse.

---

## Sensitivity: Liq Map Update Frequency

| Update Interval | Expected Impact |
|---|---|
| 1 second | Full edge. Clusters detected as they form. |
| 10 seconds | ~90% of edge retained. Misses fast-moving cluster formation. |
| 1 minute | ~70% of edge retained. Signal A degrades most (momentum timing off). |
| 10 minutes | ~40% edge. Signal B still mostly ok (cascade events themselves are timestamped). |
| 1 hour | Signal A effectively dead. Strategy runs on Signal B+C only. |

**Recommendation**: Minimum 60s refresh for production. 10s or better preferred.

---

## In-Sample vs Out-of-Sample

_To be filled after running full backtest._

IS period: 2024-01-01 → ~2025-01-01 (75%)
OOS period: ~2025-01-01 → 2025-05-01 (25%, not used for parameter tuning)

**Critical**: Do not iterate on OOS metrics. Report OOS once, prominently.

---

## Recommendations

1. Start with Signal B only in paper trading (most robust, easiest to validate)
2. Add Signal C after 4 weeks of live paper validation
3. Add Signal A last (most latency-sensitive, most infrastructure-dependent)
4. Maintain a "liq map quality" metric and auto-disable Signal A when data is stale

---

*Generated by the HL Liquidation Cascade Backtest Framework v0.1.0*
