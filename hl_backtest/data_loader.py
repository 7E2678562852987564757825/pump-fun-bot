"""
Hyperliquid data loader with parquet disk cache.

All API calls are POST to https://api.hyperliquid.xyz/info.
Cache files live under data/cache/ keyed by a deterministic hash of the request params.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.hyperliquid.xyz/info"
_CACHE_DIR = Path(__file__).parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CLIENT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _post(payload: dict[str, Any], retries: int = 4) -> Any:
    """POST to HL API with exponential backoff."""
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=_CLIENT_TIMEOUT) as client:
                resp = client.post(_BASE_URL, json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                logger.warning("API error (attempt %d/%d): %s — retrying in %.0fs", attempt + 1, retries, exc, delay)
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"All {retries} API attempts failed") from last_exc


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_path(payload: dict[str, Any], suffix: str = "") -> Path:
    return _CACHE_DIR / f"{_cache_key(payload)}{suffix}.parquet"


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, path, compression="snappy")
    logger.debug("Cached %d rows → %s", len(df), path.name)


def _load_parquet(path: Path) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_meta() -> list[dict[str, Any]]:
    """Return list of perpetual market metadata dicts."""
    cache_path = _CACHE_DIR / "meta.json"
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.debug("meta: cache hit (%.1fh old)", age_hours)
            with open(cache_path) as f:
                return json.load(f)

    logger.info("Fetching /meta")
    data = _post({"type": "meta"})
    universe: list[dict] = data.get("universe", [])

    with open(cache_path, "w") as f:
        json.dump(universe, f)
    logger.info("meta: %d instruments", len(universe))
    return universe


def get_funding_history(
    coin: str,
    start_ms: int,
    end_ms: int | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly funding rate history for *coin*.

    Returns DataFrame with columns:
        time (datetime64[ns, UTC]), fundingRate (float), premium (float)
    Indexed by time.
    """
    payload: dict[str, Any] = {"type": "fundingHistory", "coin": coin, "startTime": start_ms}
    if end_ms is not None:
        payload["endTime"] = end_ms

    path = _cache_path(payload, suffix=f"_{coin}_funding")
    if path.exists() and not force_refresh:
        logger.debug("funding %s: cache hit", coin)
        return _load_parquet(path)

    logger.info("Fetching funding history: %s", coin)
    raw: list[dict] = _post(payload)

    if not raw:
        empty = pd.DataFrame(columns=["fundingRate", "premium"])
        empty.index = pd.DatetimeIndex([], tz="UTC", name="time")
        _save_parquet(empty, path)
        return empty

    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    if "premium" in df.columns:
        df["premium"] = df["premium"].astype(float)
    else:
        df["premium"] = float("nan")

    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    _save_parquet(df, path)
    return df


def get_candles(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles for *coin*.

    Returns DataFrame with columns:
        time (datetime64[ns, UTC]), open, high, low, close, volume
    Indexed by time.
    Empty results are cached to avoid redundant API calls.
    """
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    path = _cache_path(payload, suffix=f"_{coin}_{interval}_candles")
    if path.exists() and not force_refresh:
        logger.debug("candles %s %s: cache hit", coin, interval)
        return _load_parquet(path)

    logger.info("Fetching candles: %s %s", coin, interval)
    raw: list[dict] = _post(payload)

    if not raw:
        # Cache empty result as tombstone to avoid re-fetching pre-listing periods
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index = pd.DatetimeIndex([], tz="UTC", name="time")
        _save_parquet(empty, path)
        return empty

    df = pd.DataFrame(raw)
    # HL returns: t=open_time, o, h, l, c, v
    rename = {"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=rename)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("time")[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    _save_parquet(df, path)
    return df


def get_candles_chunked(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    chunk_days: int = 60,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch candles in *chunk_days* chunks to avoid API limits, merging results.
    Uses a single merged cache file keyed on the full range.
    Skips forward aggressively when early chunks return no data (pre-listing).
    """
    merged_payload = {
        "type": "candleSnapshot_merged",
        "coin": coin,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
    }
    merged_path = _cache_path(merged_payload, suffix=f"_{coin}_{interval}_merged")
    if merged_path.exists() and not force_refresh:
        logger.debug("candles_merged %s %s: cache hit", coin, interval)
        return _load_parquet(merged_path)

    ms_per_chunk = chunk_days * 24 * 3600 * 1000
    ms_per_prelisting_skip = 120 * 24 * 3600 * 1000  # jump 4 months ahead if no data yet
    chunks: list[pd.DataFrame] = []
    cursor = start_ms
    pre_listing_empty = 0

    while cursor < end_ms:
        chunk_end = min(cursor + ms_per_chunk, end_ms)
        chunk = get_candles(coin, interval, cursor, chunk_end, force_refresh=force_refresh)
        if not chunk.empty:
            chunks.append(chunk)
            pre_listing_empty = 0
            cursor = chunk_end
        else:
            pre_listing_empty += 1
            if chunks:
                # Had data before — just advance normally (gap in data or delisted)
                cursor = chunk_end
            elif pre_listing_empty >= 2:
                # Still in pre-listing period — skip forward faster
                cursor += ms_per_prelisting_skip
            else:
                cursor = chunk_end
        time.sleep(0.05)

    if not chunks:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index = pd.DatetimeIndex([], tz="UTC", name="time")
        _save_parquet(empty, merged_path)
        return empty

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    _save_parquet(df, merged_path)
    return df


def get_funding_chunked(
    coin: str,
    start_ms: int,
    end_ms: int,
    chunk_days: int = 90,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch funding history in chunks, merged into one DataFrame.
    """
    merged_payload = {
        "type": "fundingHistory_merged",
        "coin": coin,
        "startTime": start_ms,
        "endTime": end_ms,
    }
    merged_path = _cache_path(merged_payload, suffix=f"_{coin}_funding_merged")
    if merged_path.exists() and not force_refresh:
        logger.debug("funding_merged %s: cache hit", coin)
        return _load_parquet(merged_path)

    ms_per_chunk = chunk_days * 24 * 3600 * 1000
    ms_per_prelisting_skip = 120 * 24 * 3600 * 1000
    chunks: list[pd.DataFrame] = []
    cursor = start_ms
    pre_listing_empty = 0

    while cursor < end_ms:
        chunk_end = min(cursor + ms_per_chunk, end_ms)
        chunk = get_funding_history(coin, cursor, chunk_end, force_refresh=force_refresh)
        if not chunk.empty:
            chunks.append(chunk)
            pre_listing_empty = 0
            cursor = chunk_end
        else:
            pre_listing_empty += 1
            if chunks:
                cursor = chunk_end
            elif pre_listing_empty >= 2:
                cursor += ms_per_prelisting_skip
            else:
                cursor = chunk_end
        time.sleep(0.05)

    if not chunks:
        empty = pd.DataFrame(columns=["fundingRate", "premium"])
        empty.index = pd.DatetimeIndex([], tz="UTC", name="time")
        _save_parquet(empty, merged_path)
        return empty

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    _save_parquet(df, merged_path)
    return df


def get_asset_contexts() -> list[dict[str, Any]]:
    """
    Fetch current market context (OI, 24h volume, etc.) for all assets.
    Returns list of dicts with markPx, dayNtlVlm, openInterest, etc.
    """
    cache_path = _CACHE_DIR / "asset_contexts.json"
    if cache_path.exists():
        age_minutes = (time.time() - cache_path.stat().st_mtime) / 60
        if age_minutes < 60:
            logger.debug("asset_contexts: cache hit (%.1fm old)", age_minutes)
            with open(cache_path) as f:
                return json.load(f)

    logger.info("Fetching asset contexts")
    data = _post({"type": "metaAndAssetCtxs"})
    # Returns [meta_dict, [ctx0, ctx1, ...]]
    if isinstance(data, list) and len(data) == 2:
        contexts = data[1]
    else:
        contexts = []

    with open(cache_path, "w") as f:
        json.dump(contexts, f)
    return contexts
