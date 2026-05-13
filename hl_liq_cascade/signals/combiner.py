"""Weighted signal combiner — merges all per-signal outputs into one."""

from __future__ import annotations

from hl_liq_cascade.types import CombinedSignal, SignalOutput


class SignalCombiner:
    """Combine multiple :class:`SignalOutput` objects into a single score.

    Only signals that are *present* in the input list contribute to the
    normalisation denominator, so a signal that did not fire (or was not
    computed) does not dilute the combined score.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        """
        Parameters
        ----------
        weights:
            Mapping of signal source name → weight, e.g.
            ``{"cascade_frontrun": 0.4, "postcascade_fade": 0.35, "squeeze": 0.25}``.
        """
        self.weights: dict[str, float] = weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine(
        self,
        signals: list[SignalOutput],
        coin: str,
        ts: int,
    ) -> CombinedSignal:
        """Compute weighted combination of *signals*.

        Parameters
        ----------
        signals:
            List of :class:`SignalOutput` objects (one per fired signal, for
            the same *coin* and *ts*).
        coin:
            Coin symbol — forwarded to the output.
        ts:
            Timestamp — forwarded to the output.

        Returns
        -------
        CombinedSignal
            Weighted-average score in ``[-1, 1]``.  If *signals* is empty,
            returns a zero-score signal.
        """
        # Index by source
        by_source: dict[str, SignalOutput] = {s.source: s for s in signals}

        if not by_source:
            return CombinedSignal(
                coin=coin,
                ts=ts,
                score=0.0,
                components=by_source,
                weights=self.weights,
            )

        # Sum weights only for sources that are present
        weight_sum: float = 0.0
        weighted_score: float = 0.0
        weighted_confidence: float = 0.0

        for source, sig in by_source.items():
            w = self.weights.get(source, 0.0)
            weighted_score += w * sig.score
            weighted_confidence += w * sig.confidence
            weight_sum += w

        if weight_sum == 0.0:
            # All present signals have weight 0 — fall back to equal weights
            n = len(by_source)
            weighted_score = sum(s.score for s in by_source.values()) / n
            weighted_confidence = sum(s.confidence for s in by_source.values()) / n
        else:
            weighted_score /= weight_sum
            weighted_confidence /= weight_sum

        # Clamp to [-1, 1]
        combined_score = max(-1.0, min(1.0, weighted_score))

        return CombinedSignal(
            coin=coin,
            ts=ts,
            score=combined_score,
            components=by_source,
            weights=self.weights,
        )

    def filter_by_threshold(
        self,
        signal: CombinedSignal,
        min_abs_score: float,
    ) -> bool:
        """Return ``True`` when *signal* meets the minimum score threshold.

        Parameters
        ----------
        signal:
            The combined signal to check.
        min_abs_score:
            Minimum absolute score required for the signal to pass through
            (e.g. ``0.3`` from ``signals.min_combined_score`` in config).
        """
        return abs(signal.score) >= min_abs_score
