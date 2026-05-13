"""Cluster detection in liquidation maps."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from hl_liq_cascade.types import ClusterInfo, LiqMap


class ClusterDetector:
    """Detects liquidation clusters in a :class:`LiqMap` using peak finding."""

    def __init__(self, cfg: dict) -> None:
        """
        Parameters
        ----------
        cfg:
            The ``liq_map`` section of the YAML config.
        """
        self.min_notional: float = float(cfg["cluster_min_notional_usd"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, liq_map: LiqMap) -> list[ClusterInfo]:
        """Detect liquidation clusters for both sides.

        Uses a windowed sum to smooth the notional curve and then
        ``scipy.signal.find_peaks`` to locate local maxima.

        Parameters
        ----------
        liq_map:
            The liquidation map to analyse.

        Returns
        -------
        list[ClusterInfo]
            All clusters from both sides, sorted descending by density_score.
        """
        if not liq_map.buckets:
            return []

        prices = np.array([b.price for b in liq_map.buckets], dtype=np.float64)
        long_arr = np.array([b.long_notional for b in liq_map.buckets], dtype=np.float64)
        short_arr = np.array([b.short_notional for b in liq_map.buckets], dtype=np.float64)

        clusters: list[ClusterInfo] = []

        for side, arr in (("long", long_arr), ("short", short_arr)):
            if arr.sum() == 0.0:
                continue
            side_clusters = self._detect_side(
                liq_map=liq_map,
                prices=prices,
                notional=arr,
                side=side,
            )
            clusters.extend(side_clusters)

        clusters.sort(key=lambda c: c.density_score, reverse=True)
        return clusters

    def find_nearest_cluster(
        self,
        liq_map: LiqMap,
        direction: str,
        max_distance_pct: float,
    ) -> ClusterInfo | None:
        """Return the nearest cluster in *direction* within *max_distance_pct*.

        Parameters
        ----------
        liq_map:
            The map to search.
        direction:
            ``"up"`` for clusters above current price (short liquidations),
            ``"down"`` for clusters below (long liquidations).
        max_distance_pct:
            Maximum distance from current price as a fraction (e.g. 0.02 = 2%).

        Returns
        -------
        ClusterInfo | None
        """
        clusters = self.detect(liq_map)
        px = liq_map.current_px

        if direction == "up":
            candidates = [
                c
                for c in clusters
                if c.price > px
                and c.distance_pct <= max_distance_pct
                and c.side == "short"
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda c: c.distance_pct)

        if direction == "down":
            candidates = [
                c
                for c in clusters
                if c.price < px
                and c.distance_pct <= max_distance_pct
                and c.side == "long"
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda c: c.distance_pct)

        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    def score_asymmetry(
        self,
        liq_map: LiqMap,
        depth_pct: float = 0.02,
    ) -> tuple[float, str]:
        """Compute liq asymmetry within *depth_pct* of current price.

        Parameters
        ----------
        liq_map:
            Map to analyse.
        depth_pct:
            Distance from current price to consider "near".

        Returns
        -------
        tuple[float, str]
            ``(asymmetry_ratio, dominant_side)`` where dominant_side is
            ``"long"`` if more long notional is near, else ``"short"``.
        """
        px = liq_map.current_px
        lo = px * (1.0 - depth_pct)
        hi = px * (1.0 + depth_pct)

        long_near: float = 0.0
        short_near: float = 0.0

        for b in liq_map.buckets:
            if lo <= b.price < px:
                long_near += b.long_notional
            if px < b.price <= hi:
                short_near += b.short_notional

        near_min = min(long_near, short_near)
        near_max = max(long_near, short_near)

        if near_min > 0.0:
            asymmetry_ratio = near_max / near_min
        else:
            asymmetry_ratio = near_max if near_max > 0.0 else 1.0

        dominant_side = "long" if long_near >= short_near else "short"
        return asymmetry_ratio, dominant_side

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smooth(self, arr: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply a simple uniform moving-average smoothing."""
        if len(arr) < window:
            return arr.copy()
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(arr, kernel, mode="same")

    def _detect_side(
        self,
        liq_map: LiqMap,
        prices: np.ndarray,
        notional: np.ndarray,
        side: str,
    ) -> list[ClusterInfo]:
        """Find peaks on one side (long or short) of the liquidation map."""
        total = notional.sum()
        if total == 0.0:
            return []

        smoothed = self._smooth(notional, window=5)

        # Minimum prominence: 5% of the side total or min_notional, whichever larger
        min_prominence = max(self.min_notional, total * 0.05)

        peak_indices, properties = find_peaks(
            smoothed,
            height=self.min_notional,
            prominence=min_prominence,
        )

        px = liq_map.current_px
        clusters: list[ClusterInfo] = []

        for idx in peak_indices:
            peak_price = float(prices[idx])
            peak_notional = float(notional[idx])
            distance_pct = abs(peak_price - px) / px if px > 0.0 else 0.0
            density_score = peak_notional / total

            clusters.append(
                ClusterInfo(
                    coin=liq_map.coin,
                    price=peak_price,
                    side=side,
                    notional=peak_notional,
                    distance_pct=distance_pct,
                    density_score=density_score,
                    ts=liq_map.ts,
                )
            )

        return clusters
