"""Signal C — detect extreme one-sided positioning about to be squeezed."""

from __future__ import annotations

from hl_liq_cascade.types import LiqMap, MarketSnapshot, SignalOutput


class SqueezeSignal:
    """Signal that triggers when one side of the market is dangerously crowded.

    Conditions for a squeeze signal:
    1.  Liquidation asymmetry: one side has ``liq_asymmetry_ratio`` × more
        near-price liq notional than the other side.
    2.  Funding confirmation: funding rate confirms the crowded side is paying.
    3.  Open interest growth: new positions are being opened (pressure building).

    Score convention: negative = short the crowded long side (price will fall
    if longs get squeezed); positive = long (fade crowded shorts upward).
    """

    SOURCE: str = "squeeze"

    def __init__(self, cfg: dict) -> None:
        """
        Parameters
        ----------
        cfg:
            The ``signals.squeeze`` section of the YAML config.
        """
        self.liq_depth_pct: float = float(cfg["liq_depth_pct"])                    # 0.02
        self.liq_asymmetry_ratio: float = float(cfg["liq_asymmetry_ratio"])        # 5.0
        self.funding_extreme_threshold: float = float(
            cfg["funding_extreme_threshold"]
        )                                                                            # 0.01
        self.oi_growth_window_h: int = int(cfg["oi_growth_window_h"])              # 4
        self._oi_growth_min: float = 0.05    # 5% OI growth required

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        coin: str,
        ts: int,
        liq_map: LiqMap,
        snapshot: MarketSnapshot,
        oi_history: list[tuple[int, float, float]],
    ) -> SignalOutput:
        """Compute the squeeze signal.

        Parameters
        ----------
        coin:
            Coin symbol.
        ts:
            Current timestamp (unix ms).
        liq_map:
            Current liquidation density map.
        snapshot:
            Current market snapshot (includes funding rate, OI, etc.).
        oi_history:
            Chronological list of ``(ts_ms, long_oi_usd, short_oi_usd)`` tuples.

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
        # Step 1 – near-price liq notional per side
        # ------------------------------------------------------------------
        px = snapshot.mid_px
        lo = px * (1.0 - self.liq_depth_pct)
        hi = px * (1.0 + self.liq_depth_pct)

        long_liq_near: float = sum(
            b.long_notional for b in liq_map.buckets if lo <= b.price < px
        )
        short_liq_near: float = sum(
            b.short_notional for b in liq_map.buckets if px < b.price <= hi
        )

        # ------------------------------------------------------------------
        # Step 2 – asymmetry
        # ------------------------------------------------------------------
        near_min = min(long_liq_near, short_liq_near)
        near_max = max(long_liq_near, short_liq_near)

        if near_min > 0.0:
            asymmetry = near_max / near_min
        else:
            asymmetry = near_max if near_max > 0.0 else 1.0

        crowded_side = "long" if long_liq_near >= short_liq_near else "short"

        # ------------------------------------------------------------------
        # Step 3 – funding alignment
        # ------------------------------------------------------------------
        funding_rate = snapshot.funding_rate
        funding_aligned = (
            crowded_side == "long"
            and funding_rate > self.funding_extreme_threshold
        ) or (
            crowded_side == "short"
            and funding_rate < -self.funding_extreme_threshold
        )

        # ------------------------------------------------------------------
        # Step 4 – OI growth
        # ------------------------------------------------------------------
        oi_growth_pct, oi_growing = self._compute_oi_growth(ts, oi_history)

        # Current total OI for reference
        total_oi_current: float = (
            snapshot.open_interest_long + snapshot.open_interest_short
        )

        # ------------------------------------------------------------------
        # Step 5 – trigger conditions
        # ------------------------------------------------------------------
        cond_asymmetry = asymmetry >= self.liq_asymmetry_ratio
        cond_funding = funding_aligned
        cond_oi = oi_growing

        meta = {
            "asymmetry": asymmetry,
            "crowded_side": crowded_side,
            "funding_rate": funding_rate,
            "oi_growth_pct": oi_growth_pct,
            "long_liq_near": long_liq_near,
            "short_liq_near": short_liq_near,
        }

        if not (cond_asymmetry and cond_funding and cond_oi):
            return SignalOutput(
                coin=coin,
                ts=ts,
                score=0.0,
                source=self.SOURCE,
                confidence=0.0,
                meta=meta,
            )

        # ------------------------------------------------------------------
        # Step 6 – score
        # ------------------------------------------------------------------
        # Short the crowded long side (score negative = short signal)
        # Long against the crowded short side (score positive = long signal)
        side_sign = 1 if crowded_side == "long" else -1
        score_magnitude = min(asymmetry / (self.liq_asymmetry_ratio * 3.0), 1.0)
        # We trade *against* the crowded side, so flip sign
        score = -float(side_sign) * score_magnitude
        score = max(-1.0, min(1.0, score))

        confidence = min(asymmetry / (self.liq_asymmetry_ratio * 2.0), 1.0)

        return SignalOutput(
            coin=coin,
            ts=ts,
            score=score,
            source=self.SOURCE,
            confidence=confidence,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_oi_growth(
        self,
        ts_now: int,
        oi_history: list[tuple[int, float, float]],
    ) -> tuple[float, bool]:
        """Compute total OI growth over the last oi_growth_window_h hours.

        Parameters
        ----------
        ts_now:
            Current unix timestamp (ms).
        oi_history:
            Chronological ``(ts_ms, long_oi_usd, short_oi_usd)`` tuples.

        Returns
        -------
        tuple[float, bool]
            ``(oi_growth_pct, is_growing)`` where ``is_growing`` is True when
            total OI has grown by at least ``_oi_growth_min`` within the window.
        """
        if not oi_history:
            return 0.0, False

        window_ms = self.oi_growth_window_h * 3600 * 1_000
        cutoff = ts_now - window_ms

        # Current OI: last entry
        _, long_now, short_now = oi_history[-1]
        total_oi_now = long_now + short_now

        # Past OI: oldest entry within window
        total_oi_past: float | None = None
        for sample_ts, long_oi, short_oi in oi_history:
            if sample_ts >= cutoff:
                total_oi_past = long_oi + short_oi
                break

        if total_oi_past is None or total_oi_past == 0.0:
            return 0.0, False

        oi_growth_pct = (total_oi_now - total_oi_past) / total_oi_past
        is_growing = oi_growth_pct >= self._oi_growth_min

        return oi_growth_pct, is_growing
