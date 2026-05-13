"""Signal B — fade the move after a liquidation cascade exhausts."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from hl_liq_cascade.types import LiqEvent, MarketSnapshot, SignalOutput


class PostCascadeFadeSignal:
    """Fade the directional move once a cascade starts decelerating.

    Logic
    -----
    1.  Accumulate the 5-minute cascade notional window.
    2.  Check whether that recent window is in the 95th percentile relative to
        a rolling 30-day history (it was a statistically large cascade).
    3.  Confirm the cascade is decelerating (last 60 s < 20% of 5-min total).
    4.  Confirm the resulting price move is at least ``min_price_move_atr`` ATR.
    5.  Score = −sign(price_move) × clamp(move_in_atr / 5, 0, 1).

    Score convention: positive = go long (fade a down-move), negative = go short
    (fade an up-move).
    """

    SOURCE: str = "postcascade_fade"

    def __init__(self, cfg: dict) -> None:
        """
        Parameters
        ----------
        cfg:
            The ``signals.postcascade_fade`` section of the YAML config.
        """
        self.liq_window_s: int = int(cfg["liq_window_s"])                              # 300
        self.liq_percentile_lookback_days: int = int(
            cfg["liq_percentile_lookback_days"]
        )                                                                                # 30
        self.liq_percentile_threshold: float = float(
            cfg["liq_percentile_threshold"]
        )                                                                                # 95
        self.decel_window_s: int = int(cfg["decel_window_s"])                          # 60
        self.min_price_move_atr: float = float(cfg["min_price_move_atr"])              # 2.0
        self._decel_ratio: float = 0.20          # recent must be < 20% of 5-min total

        # Rolling history: deque of (ts_ms, notional_5min) tuples,
        # kept for liq_percentile_lookback_days days.
        self._notional_history: deque[tuple[int, float]] = deque()
        self._max_history_ms: int = (
            self.liq_percentile_lookback_days * 24 * 3600 * 1_000
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_history(self, ts: int, liq_events: list[LiqEvent]) -> None:
        """Maintain the rolling 30-day notional history.

        Call this every tick *before* ``compute`` so the percentile lookup is
        populated.

        Parameters
        ----------
        ts:
            Current unix timestamp (ms).
        liq_events:
            All liquidation events available at this tick (the method sums the
            most recent ``liq_window_s`` seconds internally).
        """
        window_ms = self.liq_window_s * 1_000
        cutoff = ts - window_ms
        window_notional = sum(
            e.notional for e in liq_events if e.ts >= cutoff
        )

        self._notional_history.append((ts, window_notional))

        # Prune entries older than lookback window
        oldest_allowed = ts - self._max_history_ms
        while self._notional_history and self._notional_history[0][0] < oldest_allowed:
            self._notional_history.popleft()

    def compute(
        self,
        coin: str,
        ts: int,
        snapshot: MarketSnapshot,
        liq_events: list[LiqEvent],
        price_history: list[tuple[int, float]],
        atr: float,
    ) -> SignalOutput:
        """Compute the post-cascade fade signal.

        Parameters
        ----------
        coin:
            Coin symbol.
        ts:
            Current timestamp (unix ms).
        snapshot:
            Current market snapshot.
        liq_events:
            Recent liquidation events (all sides / coins for this coin).
        price_history:
            Chronological ``(ts_ms, price)`` tuples.
        atr:
            Average True Range for position sizing context.

        Returns
        -------
        SignalOutput
            ``score == 0.0`` when conditions are not met.
        """
        _no_signal = SignalOutput(
            coin=coin,
            ts=ts,
            score=0.0,
            source=self.SOURCE,
            confidence=0.0,
            meta={},
        )

        if atr <= 0.0:
            return _no_signal

        # ------------------------------------------------------------------
        # Step 1 – recent notional windows
        # ------------------------------------------------------------------
        liq_window_ms = self.liq_window_s * 1_000
        decel_window_ms = self.decel_window_s * 1_000

        cascade_cutoff = ts - liq_window_ms
        decel_cutoff = ts - decel_window_ms

        cascade_notional_5m: float = sum(
            e.notional for e in liq_events if e.ts >= cascade_cutoff
        )
        recent_notional_1m: float = sum(
            e.notional for e in liq_events if e.ts >= decel_cutoff
        )

        # ------------------------------------------------------------------
        # Step 2 – percentile threshold from rolling history
        # ------------------------------------------------------------------
        percentile_threshold = self._compute_percentile_threshold()

        # ------------------------------------------------------------------
        # Step 3 – cascade direction (dominant liquidated side)
        # ------------------------------------------------------------------
        long_liq_notional: float = sum(
            e.notional
            for e in liq_events
            if e.ts >= cascade_cutoff and e.side == "long"
        )
        short_liq_notional: float = sum(
            e.notional
            for e in liq_events
            if e.ts >= cascade_cutoff and e.side == "short"
        )
        # If more longs were liquidated → price moved down → fade = go long (+1)
        # If more shorts were liquidated → price moved up → fade = go short (-1)
        if long_liq_notional >= short_liq_notional:
            cascade_direction = -1   # price went down
            fade_direction = 1
        else:
            cascade_direction = 1    # price went up
            fade_direction = -1

        # ------------------------------------------------------------------
        # Step 4 – price move in ATR units
        # ------------------------------------------------------------------
        px_start = self._lookup_price_at(ts - liq_window_ms, price_history)
        px_current = snapshot.mid_px

        if px_start is None or px_start == 0.0 or atr == 0.0:
            return _no_signal

        price_move = px_current - px_start
        move_in_atr = abs(price_move) / atr

        # ------------------------------------------------------------------
        # Step 5 – trigger conditions
        # ------------------------------------------------------------------
        cond_big_cascade = (
            percentile_threshold is not None
            and cascade_notional_5m >= percentile_threshold
        )
        cond_decelerating = (
            cascade_notional_5m > 0.0
            and recent_notional_1m < cascade_notional_5m * self._decel_ratio
        )
        cond_price_move = move_in_atr >= self.min_price_move_atr

        if not (cond_big_cascade and cond_decelerating and cond_price_move):
            return SignalOutput(
                coin=coin,
                ts=ts,
                score=0.0,
                source=self.SOURCE,
                confidence=0.0,
                meta={
                    "cascade_notional_5m": cascade_notional_5m,
                    "recent_notional_1m": recent_notional_1m,
                    "move_in_atr": move_in_atr,
                    "percentile_threshold": percentile_threshold,
                    "cond_big_cascade": cond_big_cascade,
                    "cond_decelerating": cond_decelerating,
                    "cond_price_move": cond_price_move,
                },
            )

        # ------------------------------------------------------------------
        # Step 6 – score
        # ------------------------------------------------------------------
        score = float(fade_direction) * min(move_in_atr / 5.0, 1.0)
        score = max(-1.0, min(1.0, score))

        # Confidence: how extreme was the cascade relative to threshold
        if percentile_threshold and percentile_threshold > 0.0:
            confidence = min(cascade_notional_5m / (percentile_threshold * 2.0), 1.0)
        else:
            confidence = 0.5

        return SignalOutput(
            coin=coin,
            ts=ts,
            score=score,
            source=self.SOURCE,
            confidence=confidence,
            meta={
                "cascade_notional_5m": cascade_notional_5m,
                "recent_notional_1m": recent_notional_1m,
                "move_in_atr": move_in_atr,
                "percentile_threshold": percentile_threshold,
                "cascade_direction": cascade_direction,
                "fade_direction": fade_direction,
                "long_liq_notional": long_liq_notional,
                "short_liq_notional": short_liq_notional,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_percentile_threshold(self) -> float | None:
        """Return the Nth percentile of stored 5-min cascade notional values.

        Returns ``None`` if there is insufficient history (< 10 samples).
        """
        if len(self._notional_history) < 10:
            return None

        values = np.array(
            [notional for _, notional in self._notional_history], dtype=np.float64
        )
        return float(np.percentile(values, self.liq_percentile_threshold))

    @staticmethod
    def _lookup_price_at(
        target_ts: int,
        price_history: list[tuple[int, float]],
    ) -> float | None:
        """Return the price closest to *target_ts* from *price_history*.

        Uses the last sample at or before target_ts; if none exists, uses the
        first available sample.
        """
        if not price_history:
            return None

        best_px: float | None = None
        for sample_ts, sample_px in price_history:
            if sample_ts <= target_ts:
                best_px = sample_px
            else:
                break  # price_history is chronological

        return best_px if best_px is not None else price_history[0][1]
