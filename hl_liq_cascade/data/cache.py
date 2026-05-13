"""Parquet-based caching layer for the HL liquidation cascade framework."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from hl_liq_cascade.types import Fill, FundingPayment, Position

logger = logging.getLogger(__name__)


def _ms_to_date_str(ts_ms: int) -> str:
    """Convert a millisecond timestamp to a YYYY-MM-DD string (UTC)."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def _fills_to_df(fills: list[Fill]) -> pl.DataFrame:
    """Convert a list of Fill objects to a Polars DataFrame."""
    if not fills:
        return pl.DataFrame(schema={
            "tid": pl.Int64,
            "coin": pl.Utf8,
            "px": pl.Utf8,
            "sz": pl.Utf8,
            "side": pl.Utf8,
            "time": pl.Int64,
            "startPosition": pl.Utf8,
            "dir": pl.Utf8,
            "closedPnl": pl.Utf8,
            "hash": pl.Utf8,
            "oid": pl.Int64,
            "crossed": pl.Boolean,
            "fee": pl.Utf8,
            "feeToken": pl.Utf8,
            "liquidation": pl.Utf8,
        })
    rows = [
        {
            "tid": f.tid,
            "coin": f.coin,
            "px": f.px,
            "sz": f.sz,
            "side": f.side,
            "time": f.time,
            "startPosition": f.startPosition,
            "dir": f.dir,
            "closedPnl": f.closedPnl,
            "hash": f.hash,
            "oid": f.oid,
            "crossed": f.crossed,
            "fee": f.fee,
            "feeToken": f.feeToken,
            "liquidation": f.liquidation,
        }
        for f in fills
    ]
    return pl.DataFrame(rows)


def _funding_to_df(payments: list[FundingPayment]) -> pl.DataFrame:
    """Convert a list of FundingPayment objects to a Polars DataFrame."""
    if not payments:
        return pl.DataFrame(schema={
            "coin": pl.Utf8,
            "usdc": pl.Utf8,
            "szi": pl.Utf8,
            "fundingRate": pl.Utf8,
            "time": pl.Int64,
            "hash": pl.Utf8,
        })
    rows = [
        {
            "coin": p.coin,
            "usdc": p.usdc,
            "szi": p.szi,
            "fundingRate": p.fundingRate,
            "time": p.time,
            "hash": p.hash,
        }
        for p in payments
    ]
    return pl.DataFrame(rows)


def _positions_to_df(positions: list[Position], snapshot_ts: int) -> pl.DataFrame:
    """Convert a list of Position objects to a Polars DataFrame with snapshot_ts."""
    if not positions:
        return pl.DataFrame(schema={
            "snapshot_ts": pl.Int64,
            "address": pl.Utf8,
            "coin": pl.Utf8,
            "size": pl.Float64,
            "entry_px": pl.Float64,
            "liq_px": pl.Float64,
            "margin_used": pl.Float64,
            "unrealized_pnl": pl.Float64,
            "leverage": pl.Int64,
            "leverage_type": pl.Utf8,
        })
    rows = [
        {
            "snapshot_ts": snapshot_ts,
            "address": p.address,
            "coin": p.coin,
            "size": p.size,
            "entry_px": p.entry_px,
            "liq_px": p.liq_px,
            "margin_used": p.margin_used,
            "unrealized_pnl": p.unrealized_pnl,
            "leverage": p.leverage,
            "leverage_type": p.leverage_type,
        }
        for p in positions
    ]
    return pl.DataFrame(rows)


class Cache:
    """Parquet-backed cache for fills, funding, position snapshots, liq maps, and candles."""

    def __init__(self, cache_dir: str | Path, compression: str = "zstd") -> None:
        self._root = Path(cache_dir)
        self._compression = compression
        self._root.mkdir(parents=True, exist_ok=True)
        logger.debug("Cache initialised at %s (compression=%s)", self._root, compression)

    @property
    def cache_dir(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_parquet(self, path: Path, df: pl.DataFrame) -> None:
        """Write a Polars DataFrame to parquet using pyarrow."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = df.to_arrow()
        pq.write_table(table, str(path), compression=self._compression)
        logger.debug("Wrote %d rows to %s", len(df), path)

    def _read_parquet(self, path: Path) -> pl.DataFrame | None:
        """Read parquet file into a Polars DataFrame. Returns None if file missing."""
        if not path.exists():
            return None
        try:
            table = pq.read_table(str(path))
            return pl.from_arrow(table)  # type: ignore[return-value]
        except Exception as exc:
            logger.error("Failed to read parquet %s: %s", path, exc)
            return None

    def _append_and_dedup(
        self,
        path: Path,
        new_df: pl.DataFrame,
        dedup_cols: list[str],
        sort_col: str | None = None,
    ) -> None:
        """Append new_df to existing parquet at path, dedup by dedup_cols, write back."""
        existing = self._read_parquet(path)
        if existing is not None and len(existing) > 0:
            combined = pl.concat([existing, new_df], how="diagonal")
        else:
            combined = new_df

        combined = combined.unique(subset=dedup_cols, keep="last")
        if sort_col and sort_col in combined.columns:
            combined = combined.sort(sort_col)

        self._write_parquet(path, combined)

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def _fills_path(self, address: str) -> Path:
        return self._root / "fills" / address[:8] / f"{address}.parquet"

    def save_fills(self, address: str, fills: list[Fill]) -> None:
        """Persist fills, appending to existing data and deduplicating by tid."""
        new_df = _fills_to_df(fills)
        if len(new_df) == 0:
            logger.debug("save_fills: no fills to save for %s", address[:8])
            return
        path = self._fills_path(address)
        self._append_and_dedup(path, new_df, dedup_cols=["tid"], sort_col="time")
        logger.debug("save_fills: saved %d fills for %s", len(fills), address[:8])

    def load_fills(self, address: str) -> pl.DataFrame | None:
        """Load fills for an address. Returns None if not cached."""
        path = self._fills_path(address)
        df = self._read_parquet(path)
        if df is not None:
            logger.debug("load_fills: loaded %d rows for %s", len(df), address[:8])
        return df

    # ------------------------------------------------------------------
    # Funding payments
    # ------------------------------------------------------------------

    def _funding_path(self, address: str) -> Path:
        return self._root / "funding" / address[:8] / f"{address}.parquet"

    def save_funding(self, address: str, payments: list[FundingPayment]) -> None:
        """Persist funding payments, deduplicating by (time, coin)."""
        new_df = _funding_to_df(payments)
        if len(new_df) == 0:
            logger.debug("save_funding: no payments to save for %s", address[:8])
            return
        path = self._funding_path(address)
        self._append_and_dedup(path, new_df, dedup_cols=["time", "coin"], sort_col="time")
        logger.debug("save_funding: saved %d payments for %s", len(payments), address[:8])

    def load_funding(self, address: str) -> pl.DataFrame | None:
        """Load funding payments for an address. Returns None if not cached."""
        path = self._funding_path(address)
        df = self._read_parquet(path)
        if df is not None:
            logger.debug("load_funding: loaded %d rows for %s", len(df), address[:8])
        return df

    # ------------------------------------------------------------------
    # Position snapshots
    # ------------------------------------------------------------------

    def _snapshot_path(self, ts: int) -> Path:
        date_str = _ms_to_date_str(ts)
        return self._root / "snapshots" / date_str / f"{ts}.parquet"

    def save_position_snapshot(self, ts: int, positions: list[Position]) -> None:
        """Save a point-in-time snapshot of all positions."""
        df = _positions_to_df(positions, snapshot_ts=ts)
        path = self._snapshot_path(ts)
        # Snapshots are immutable point-in-time, so we overwrite directly
        self._write_parquet(path, df)
        logger.debug("save_position_snapshot: %d positions at ts=%d", len(positions), ts)

    def load_position_snapshots(self, start_ts: int, end_ts: int) -> pl.DataFrame:
        """Load all position snapshots whose ts falls in [start_ts, end_ts]."""
        snapshots_root = self._root / "snapshots"
        if not snapshots_root.exists():
            return pl.DataFrame()

        frames: list[pl.DataFrame] = []
        # Iterate over all date directories and collect matching parquet files
        for date_dir in sorted(snapshots_root.iterdir()):
            if not date_dir.is_dir():
                continue
            for parquet_file in sorted(date_dir.glob("*.parquet")):
                try:
                    # File stem is the ts (milliseconds)
                    file_ts = int(parquet_file.stem)
                except ValueError:
                    continue
                if start_ts <= file_ts <= end_ts:
                    df = self._read_parquet(parquet_file)
                    if df is not None and len(df) > 0:
                        frames.append(df)

        if not frames:
            return pl.DataFrame()

        combined = pl.concat(frames, how="diagonal")
        if "snapshot_ts" in combined.columns:
            combined = combined.sort("snapshot_ts")
        logger.debug(
            "load_position_snapshots: loaded %d rows for ts=[%d, %d]",
            len(combined), start_ts, end_ts,
        )
        return combined

    # ------------------------------------------------------------------
    # Liquidation maps
    # ------------------------------------------------------------------

    def _liq_map_path(self, coin: str, ts: int) -> Path:
        date_str = _ms_to_date_str(ts)
        return self._root / "liq_maps" / coin / date_str / f"{ts}.parquet"

    def save_liq_map(self, coin: str, ts: int, liq_map_df: pl.DataFrame) -> None:
        """Save a liquidation map DataFrame for a coin at a given timestamp."""
        path = self._liq_map_path(coin, ts)
        self._write_parquet(path, liq_map_df)
        logger.debug("save_liq_map: %s ts=%d rows=%d", coin, ts, len(liq_map_df))

    def load_liq_maps(self, coin: str, start_ts: int, end_ts: int) -> pl.DataFrame | None:
        """Load all liq map snapshots for a coin in [start_ts, end_ts]."""
        coin_dir = self._root / "liq_maps" / coin
        if not coin_dir.exists():
            return None

        frames: list[pl.DataFrame] = []
        for date_dir in sorted(coin_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for parquet_file in sorted(date_dir.glob("*.parquet")):
                try:
                    file_ts = int(parquet_file.stem)
                except ValueError:
                    continue
                if start_ts <= file_ts <= end_ts:
                    df = self._read_parquet(parquet_file)
                    if df is not None and len(df) > 0:
                        frames.append(df)

        if not frames:
            return None

        combined = pl.concat(frames, how="diagonal")
        logger.debug(
            "load_liq_maps: %s loaded %d rows for ts=[%d, %d]",
            coin, len(combined), start_ts, end_ts,
        )
        return combined

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def _candles_path(self, coin: str, interval: str) -> Path:
        return self._root / "candles" / coin / f"{interval}.parquet"

    def save_candles(self, coin: str, interval: str, df: pl.DataFrame) -> None:
        """Persist candle data, appending and deduplicating by open time (t)."""
        if len(df) == 0:
            return
        path = self._candles_path(coin, interval)
        self._append_and_dedup(path, df, dedup_cols=["t"], sort_col="t")
        logger.debug("save_candles: %s/%s %d rows", coin, interval, len(df))

    def load_candles(self, coin: str, interval: str) -> pl.DataFrame | None:
        """Load candle data for a coin+interval. Returns None if not cached."""
        path = self._candles_path(coin, interval)
        df = self._read_parquet(path)
        if df is not None:
            logger.debug("load_candles: %s/%s %d rows", coin, interval, len(df))
        return df
