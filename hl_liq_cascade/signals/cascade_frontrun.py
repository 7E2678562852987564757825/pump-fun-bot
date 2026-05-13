"""Signal A — front-run imminent liquidation cascades."""

from __future__ import annotations

import math

from hl_liq_cascade.liq_map.cluster import ClusterDetector
from hl_liq_cascade.types import ClusterInfo, LiqMap, MarketSnapshot, SignalOutput


class CascadeFrontrunSignal:
    """Generate a directional signal when a cascade is about to be triggered.

    The signal is positive (long) when shorts are about to be squeezed, i.e.
    the price is moving upward into a dense cluster of short liquidations.
    It is negative (short) when longs are about to be liquidated below.

    Score is in [-1, 1].  A score of 0.0 is returned whenever trigger
    conditions are not met.
    """

    SOURCE: str = "cascade_frontrun"

    def __init__(self, cfg: dict) -> None:
        """
        Parameters
        ----------
        cfg:
            The ``signals.cascade_frontrun`` section of the YAML config.
        """
        self.momentum_window_s: int = int(cfg["momentum_window_s"])          # 60
        self.cluster_depth_pct: float = float(cfg["cluster_depth_pct"])      # 0.005
        self.liq_imbalance_ratio: float = float(cfg["liq_imbalance_ratio"])  # 3.0
        self.min_cluster_vol_fraction: float = float(
            cfg["min_cluster_vol_fraction"]
        )                                                                      # 0.005
        self._min_velocity: float = 0.0005   # 0.05% in momentum window

        # ClusterDetector is constructed with a minimal cfg dict
        self._cluster_detector = ClusterDetector(
            {"cluster_min_notional_usd": 100_000}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        coin: str,
        ts: int,
        liq_map: LiqMap,
        snapshot: MarketSnapshot,
        price_history: list[tuple[int, float]],
    ) -> SignalOutput:
        """Compute the cascade frontrun signal for *coin* at *ts*.

        Parameters
        ----------
        coin:
            Coin symbol (e.g. ``"BTC"``).
        ts:
            Current timestamp in unix milliseconds.
        liq_map:
            Most recent liquidation map for *coin*.
        snapshot:
            Current market snapshot.
        price_history:
            Chronological list of ``(ts_ms, price)`` tuples.

        Returns
        -------
        SignalOutput
            ``score == 0.0`` when trigger conditions are not met.
        """
        _no_signal = SignalOutput(
            coin=coin,
            ts=ts,
            score=0.0,
            source=self.SOURCE,
            confidence=0.0,
            meta={},
        )

        # ------------------------------------------------------------------
        # Step 1 – price velocity over momentum_window_s
        # ------------------------------------------------------------------
        price_velocity = self._compute_velocity(
            ts, price_history, self.momentum_window_s
        )
        if price_velocity is None:
            return _no_signal

        current_px = snapshot.mid_px

        # ------------------------------------------------------------------
        # Step 2 – accumulate directional liq notional
        # ------------------------------------------------------------------
        lo = current_px * (1.0 - self.cluster_depth_pct)
        hi = current_px * (1.0 + self.cluster_depth_pct)

        liq_above: float = sum(
            b.short_notional for b in liq_map.buckets if b.price > current_px and b.price <= hi
        )
        liq_below: float = sum(
            b.long_notional for b in liq_map.buckets if b.price < current_px and b.price >= lo
        )

        if price_velocity > 0.0:
            # Momentum is upward → shorts at risk
            liq_in_direction = liq_above    # short liq clusters above
            liq_opposite = liq_below        # long liq clusters below
            momentum_direction = 1
        else:
            # Momentum is downward → longs at risk
            liq_in_direction = liq_below    # long liq clusters below
            liq_opposite = liq_above        # short liq clusters above
            momentum_direction = -1

        # ------------------------------------------------------------------
        # Step 3 – nearest cluster price for metadata
        # ------------------------------------------------------------------
        cluster_dir = "up" if price_velocity > 0.0 else "down"
        nearest: ClusterInfo | None = self._cluster_detector.find_nearest_cluster(
            liq_map, cluster_dir, self.cluster_depth_pct * 5
        )
        nearest_cluster_px: float | None = nearest.price if nearest else None

        # ------------------------------------------------------------------
        # Step 4 – trigger conditions
        # ------------------------------------------------------------------
        min_vol_notional = self.min_cluster_vol_fraction * snapshot.volume_24h
        liq_imbalance = (
            liq_in_direction / (liq_opposite + 1.0)
        )

        cond_imbalance = liq_imbalance >= self.liq_imbalance_ratio
        cond_cluster_size = liq_in_direction >= min_vol_notional
        cond_velocity = abs(price_velocity) >= self._min_velocity

        if not (cond_imbalance and cond_cluster_size and cond_velocity):
            return SignalOutput(
                coin=coin,
                ts=ts,
                score=0.0,
                source=self.SOURCE,
                confidence=0.0,
                meta={
                    "momentum_pct": price_velocity,
                    "cluster_notional": liq_in_direction,
                    "liq_imbalance": liq_imbalance,
                    "nearest_cluster_px": nearest_cluster_px,
                },
            )

        # ------------------------------------------------------------------
        # Step 5 – score and confidence
        # ------------------------------------------------------------------
        score_magnitude = min(
            liq_in_direction / (min_vol_notional * 10.0), 1.0
        )
        score = float(momentum_direction) * score_magnitude

        confidence = min(liq_imbalance / (self.liq_imbalance_ratio * 3.0), 1.0)

        return SignalOutput(
            coin=coin,
            ts=ts,
            score=score,
            source=self.SOURCE,
            confidence=confidence,
            meta={
                "momentum_pct": price_velocity,
                "cluster_notional": liq_in_direction,
                "liq_imbalance": liq_imbalance,
                "nearest_cluster_px": nearest_cluster_px,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_velocity(
        ts_now: int,
        price_history: list[tuple[int, float]],
        window_s: int,
    ) -> float | None:
        """Return (current_px - px_n_seconds_ago) / px_n_seconds_ago.

        Returns ``None`` if there is insufficient history.
        """
        if not price_history:
            return None

        window_ms = window_s * 1_000
        cutoff_ms = ts_now - window_ms

        # price_history is chronological; current price is the last entry
        current_px = price_history[-1][1]

        # Find the oldest sample that is within the window
        ref_px: float | None = None
        for sample_ts, sample_px in price_history:
            if sample_ts >= cutoff_ms:
                ref_px = sample_px
                break

        if ref_px is None or ref_px == 0.0:
            return None

        return (current_px - ref_px) / ref_px
