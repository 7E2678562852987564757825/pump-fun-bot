"""Builds liquidation maps from position snapshots."""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Any

from hl_liq_cascade.types import LiqBucket, LiqMap, Position


class LiqMapBuilder:
    """Constructs per-coin liquidation density maps from position data."""

    def __init__(self, cfg: dict) -> None:
        """
        Parameters
        ----------
        cfg:
            The ``liq_map`` section of the YAML config.
        """
        self.bucket_pct: float = float(cfg["bucket_pct"])          # e.g. 0.001
        self.density_window_pct: float = float(cfg["density_window_pct"])  # e.g. 0.005
        self._price_range_pct: float = 0.20                         # ±20%

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        positions: list[Position] | pl.DataFrame,
        current_prices: dict[str, float],
        ts: int,
    ) -> dict[str, LiqMap]:
        """Build liquidation maps for every coin present in *positions*.

        Parameters
        ----------
        positions:
            Either a list of :class:`~hl_liq_cascade.types.Position` objects or
            a Polars DataFrame with the same schema (columns: coin, size, liq_px,
            entry_px, ...).
        current_prices:
            Mapping of coin → current mid price.
        ts:
            Unix timestamp (ms) of the snapshot.

        Returns
        -------
        dict[str, LiqMap]
            One :class:`~hl_liq_cascade.types.LiqMap` per coin.
        """
        pos_list: list[Position] = self._normalise_positions(positions)

        # Group by coin
        by_coin: dict[str, list[Position]] = {}
        for p in pos_list:
            if p.liq_px is None:
                continue
            by_coin.setdefault(p.coin, []).append(p)

        result: dict[str, LiqMap] = {}
        for coin, coin_positions in by_coin.items():
            if coin not in current_prices:
                continue
            current_px = current_prices[coin]
            liq_map = self._build_single(coin, coin_positions, current_px, ts)
            result[coin] = liq_map

        return result

    def compute_density(
        self,
        liq_map: LiqMap,
        window_pct: float | None = None,
    ) -> dict[str, float]:
        """Compute a suite of density metrics for *liq_map*.

        Parameters
        ----------
        liq_map:
            The map to analyse.
        window_pct:
            Half-width of the near window as a fraction of current price.
            Defaults to ``density_window_pct`` from config.

        Returns
        -------
        dict with keys:
            - ``long_density_near``  – long notional within window below price
            - ``short_density_near`` – short notional within window above price
            - ``imbalance_ratio``    – max/min of the two near values
            - ``liq_imbalance_direction`` – "long_biased" or "short_biased"
            - ``weighted_density``   – proximity-weighted total notional
        """
        window = window_pct if window_pct is not None else self.density_window_pct
        px = liq_map.current_px
        lo = px * (1.0 - window)   # lower bound for longs below price
        hi = px * (1.0 + window)   # upper bound for shorts above price

        long_density_near: float = 0.0
        short_density_near: float = 0.0
        weighted_density: float = 0.0

        for bucket in liq_map.buckets:
            bp = bucket.price
            distance_frac = abs(bp - px) / px if px > 0.0 else 0.0
            proximity_weight = 1.0 / (1.0 + distance_frac)

            # Long liq levels are below current price
            if lo <= bp < px:
                long_density_near += bucket.long_notional

            # Short liq levels are above current price
            if px < bp <= hi:
                short_density_near += bucket.short_notional

            weighted_density += (
                bucket.long_notional + bucket.short_notional
            ) * proximity_weight

        near_min = min(long_density_near, short_density_near)
        near_max = max(long_density_near, short_density_near)

        if near_min > 0.0:
            imbalance_ratio = near_max / near_min
        else:
            imbalance_ratio = near_max if near_max > 0.0 else 1.0

        direction = (
            "long_biased" if long_density_near >= short_density_near else "short_biased"
        )

        result: dict[str, Any] = {
            "long_density_near": long_density_near,
            "short_density_near": short_density_near,
            "imbalance_ratio": imbalance_ratio,
            "liq_imbalance_direction": direction,
            "weighted_density": weighted_density,
        }
        return result

    def to_dataframe(self, liq_map: LiqMap) -> pl.DataFrame:
        """Convert a :class:`LiqMap` to a Polars DataFrame.

        Columns: ``price``, ``long_notional``, ``short_notional``,
        ``net_notional``, ``distance_pct``.
        """
        if not liq_map.buckets:
            return pl.DataFrame(
                schema={
                    "price": pl.Float64,
                    "long_notional": pl.Float64,
                    "short_notional": pl.Float64,
                    "net_notional": pl.Float64,
                    "distance_pct": pl.Float64,
                }
            )

        px = liq_map.current_px
        prices: list[float] = []
        longs: list[float] = []
        shorts: list[float] = []
        nets: list[float] = []
        distances: list[float] = []

        for b in liq_map.buckets:
            prices.append(b.price)
            longs.append(b.long_notional)
            shorts.append(b.short_notional)
            nets.append(b.net_notional)
            distances.append((b.price - px) / px if px > 0.0 else 0.0)

        return pl.DataFrame(
            {
                "price": prices,
                "long_notional": longs,
                "short_notional": shorts,
                "net_notional": nets,
                "distance_pct": distances,
            }
        )

    def liq_map_to_series(self, liq_maps: dict[str, LiqMap]) -> pl.DataFrame:
        """Combine all per-coin maps into a single wide DataFrame for storage.

        Rows correspond to (coin, bucket_price) pairs. Additional columns
        carry the timestamp and current price for each coin.
        """
        frames: list[pl.DataFrame] = []
        for coin, liq_map in liq_maps.items():
            df = self.to_dataframe(liq_map)
            df = df.with_columns(
                [
                    pl.lit(coin).alias("coin"),
                    pl.lit(liq_map.ts).alias("ts"),
                    pl.lit(liq_map.current_px).alias("current_px"),
                ]
            )
            frames.append(df)

        if not frames:
            return pl.DataFrame(
                schema={
                    "coin": pl.Utf8,
                    "ts": pl.Int64,
                    "current_px": pl.Float64,
                    "price": pl.Float64,
                    "long_notional": pl.Float64,
                    "short_notional": pl.Float64,
                    "net_notional": pl.Float64,
                    "distance_pct": pl.Float64,
                }
            )

        return pl.concat(frames, how="diagonal_relaxed")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_positions(
        positions: list[Position] | pl.DataFrame,
    ) -> list[Position]:
        """Return a plain list of Position objects regardless of input type."""
        if isinstance(positions, pl.DataFrame):
            result: list[Position] = []
            for row in positions.iter_rows(named=True):
                result.append(
                    Position(
                        address=row.get("address", ""),
                        coin=row["coin"],
                        size=float(row["size"]),
                        entry_px=float(row.get("entry_px", 0.0) or 0.0),
                        liq_px=(
                            float(row["liq_px"])
                            if row.get("liq_px") is not None
                            else None
                        ),
                        margin_used=float(row.get("margin_used", 0.0) or 0.0),
                        unrealized_pnl=float(
                            row.get("unrealized_pnl", 0.0) or 0.0
                        ),
                        leverage=int(row.get("leverage", 1) or 1),
                        leverage_type=row.get("leverage_type", "cross") or "cross",
                        snapshot_ts=int(row.get("snapshot_ts", 0) or 0),
                    )
                )
            return result
        return positions  # already list[Position]

    def _build_single(
        self,
        coin: str,
        positions: list[Position],
        current_px: float,
        ts: int,
    ) -> LiqMap:
        """Build one LiqMap for *coin* from filtered position list."""
        px_min = current_px * (1.0 - self._price_range_pct)
        px_max = current_px * (1.0 + self._price_range_pct)
        bucket_width = current_px * self.bucket_pct

        # Create bucket edges (left edges of each bucket)
        edges: np.ndarray = np.arange(px_min, px_max, bucket_width)
        n_buckets = len(edges)

        long_notional: np.ndarray = np.zeros(n_buckets, dtype=np.float64)
        short_notional: np.ndarray = np.zeros(n_buckets, dtype=np.float64)

        for pos in positions:
            liq_px = pos.liq_px
            if liq_px is None:
                continue
            if liq_px < px_min or liq_px >= px_max:
                continue  # outside our displayed range

            notional = abs(pos.size) * liq_px
            # np.searchsorted returns the index of the first edge > liq_px,
            # so bucket index = that value - 1 (clamped to valid range).
            idx = int(np.searchsorted(edges, liq_px, side="right")) - 1
            idx = max(0, min(idx, n_buckets - 1))

            if pos.size > 0.0:
                # Long position → liquidated when price drops below liq_px
                long_notional[idx] += notional
            else:
                # Short position → liquidated when price rises above liq_px
                short_notional[idx] += notional

        buckets: list[LiqBucket] = [
            LiqBucket(
                price=float(edges[i]),
                long_notional=float(long_notional[i]),
                short_notional=float(short_notional[i]),
            )
            for i in range(n_buckets)
        ]

        return LiqMap(
            coin=coin,
            ts=ts,
            current_px=current_px,
            buckets=buckets,
            bucket_pct=self.bucket_pct,
        )
