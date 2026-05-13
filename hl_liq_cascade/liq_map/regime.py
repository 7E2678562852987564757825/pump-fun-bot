"""Market regime classification based on liquidation map state."""

from __future__ import annotations

from hl_liq_cascade.liq_map.cluster import ClusterDetector
from hl_liq_cascade.types import LiqMap, MarketSnapshot


class RegimeClassifier:
    """Classifies market conditions into qualitative regimes.

    Regime vocabulary
    -----------------
    squeeze_risk_short
        Dense cluster of *short* liquidations just above current price AND
        price is drifting upward.  Shorts are about to get squeezed.
    squeeze_risk_long
        Dense cluster of *long* liquidations just below current price AND
        price is drifting downward.  Longs are about to be liquidated.
    cascade_ongoing
        Rapid price move with elevated real-time liquidation flow.
    post_cascade
        Liquidation flow is decelerating after a recent spike.
    neutral
        No significant nearby clusters.
    """

    # Thresholds (could be moved to cfg if needed)
    _NEAR_CLUSTER_DEPTH_PCT: float = 0.01          # 1% = "nearby"
    _CASCADE_LIQ_SPIKE_RATIO: float = 3.0           # recent vs prev 5s
    _POST_CASCADE_DECEL_RATIO: float = 0.5          # recent < 50% of peak
    _MOMENTUM_MIN_PCT: float = 0.001                # 0.1% price move

    def __init__(self, cfg: dict) -> None:
        """
        Parameters
        ----------
        cfg:
            Full liq_map section of the YAML config.
        """
        self._cluster_detector = ClusterDetector(cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        liq_map: LiqMap,
        snapshot: MarketSnapshot,
        recent_liq_notional_5s: float,
        prev_liq_notional_5s: float,
    ) -> str:
        """Return a regime string for the current market state.

        Parameters
        ----------
        liq_map:
            Current liquidation density map for the coin.
        snapshot:
            Current market snapshot (price, funding, OI …).
        recent_liq_notional_5s:
            Total liquidation notional in the most recent 5-second window.
        prev_liq_notional_5s:
            Total liquidation notional in the previous 5-second window.
        """
        # 1. Cascade / post-cascade check (flow-based, takes priority)
        if (
            prev_liq_notional_5s > 0.0
            and recent_liq_notional_5s
            >= prev_liq_notional_5s * self._CASCADE_LIQ_SPIKE_RATIO
        ):
            return "cascade_ongoing"

        if (
            prev_liq_notional_5s > 0.0
            and recent_liq_notional_5s > 0.0
            and recent_liq_notional_5s
            <= prev_liq_notional_5s * self._POST_CASCADE_DECEL_RATIO
        ):
            return "post_cascade"

        # 2. Squeeze risk checks (cluster-based)
        is_risk, direction = self.is_cascade_risk(
            liq_map, snapshot, depth_pct=self._NEAR_CLUSTER_DEPTH_PCT
        )
        if is_risk:
            if direction == "up":
                return "squeeze_risk_short"
            if direction == "down":
                return "squeeze_risk_long"

        return "neutral"

    def is_cascade_risk(
        self,
        liq_map: LiqMap,
        snapshot: MarketSnapshot,
        depth_pct: float = 0.005,
    ) -> tuple[bool, str]:
        """Determine whether a liquidation cascade is imminent and its direction.

        Parameters
        ----------
        liq_map:
            Liquidation density map for the coin.
        snapshot:
            Current market snapshot.
        depth_pct:
            How close a cluster must be to count as "just ahead".

        Returns
        -------
        tuple[bool, str]
            ``(True, "down")`` – large long-liq cluster just below price,
            cascade risk to the downside.
            ``(True, "up")``   – large short-liq cluster just above price,
            cascade risk to the upside (short squeeze).
            ``(False, "")``    – no significant nearby risk.
        """
        px = liq_map.current_px
        lo = px * (1.0 - depth_pct)
        hi = px * (1.0 + depth_pct)

        long_notional_below: float = sum(
            b.long_notional
            for b in liq_map.buckets
            if lo <= b.price < px
        )
        short_notional_above: float = sum(
            b.short_notional
            for b in liq_map.buckets
            if px < b.price <= hi
        )

        total_long = liq_map.total_long_notional or 1.0
        total_short = liq_map.total_short_notional or 1.0

        long_frac = long_notional_below / total_long
        short_frac = short_notional_above / total_short

        # Significant = at least 10% of total side notional concentrated near price
        _SIGNIFICANCE_FRAC = 0.10

        if long_frac >= _SIGNIFICANCE_FRAC and long_frac >= short_frac:
            return True, "down"
        if short_frac >= _SIGNIFICANCE_FRAC and short_frac > long_frac:
            return True, "up"

        return False, ""
