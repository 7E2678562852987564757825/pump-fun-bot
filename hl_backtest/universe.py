"""
Monthly universe selection: top 30 alt perps by 30d rolling volume.
Excludes BTC and ETH. Uses historical candle data to avoid look-ahead.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import NamedTuple

import pandas as pd

from data_loader import get_meta, get_candles_chunked

logger = logging.getLogger(__name__)

_EXCLUDE = {"BTC", "ETH"}
_TOP_N = 30
_VOLUME_LOOKBACK_DAYS = 30
# Minimum trading days required to be included in universe
_MIN_TRADING_DAYS = 20


class UniverseSnapshot(NamedTuple):
    date: datetime
    coins: list[str]
    volumes: pd.Series  # coin -> 30d notional volume


def get_all_alt_perps() -> list[str]:
    """Return all alt perp coin names (excludes BTC/ETH)."""
    meta = get_meta()
    coins = [m["name"] for m in meta if m["name"] not in _EXCLUDE]
    logger.info("Total alt perps available: %d", len(coins))
    return coins


def select_universe_at(
    date: datetime,
    all_coins: list[str],
    candles_map: dict[str, pd.DataFrame],
) -> UniverseSnapshot:
    """
    Select top-30 coins by 30d notional volume ending at *date*.
    Uses only data available at *date* (no look-ahead).
    """
    end = pd.Timestamp(date).tz_localize("UTC") if date.tzinfo is None else pd.Timestamp(date)
    start = end - pd.Timedelta(days=_VOLUME_LOOKBACK_DAYS)

    volumes: dict[str, float] = {}
    for coin in all_coins:
        if coin not in candles_map:
            continue
        df = candles_map[coin]
        window = df.loc[(df.index >= start) & (df.index < end)]
        if len(window) < _MIN_TRADING_DAYS:
            continue
        # Notional volume = sum(close * volume) over window
        notional = (window["close"] * window["volume"]).sum()
        if notional > 0:
            volumes[coin] = float(notional)

    if not volumes:
        logger.warning("No volume data available for universe at %s", date.date())
        return UniverseSnapshot(date=date, coins=[], volumes=pd.Series(dtype=float))

    vol_series = pd.Series(volumes).sort_values(ascending=False)
    top = vol_series.head(_TOP_N)
    coins = top.index.tolist()

    logger.info(
        "Universe at %s: %d coins | top5: %s",
        date.date(),
        len(coins),
        ", ".join(coins[:5]),
    )
    return UniverseSnapshot(date=date, coins=coins, volumes=top)


def build_monthly_universes(
    start: datetime,
    end: datetime,
    candles_map: dict[str, pd.DataFrame],
) -> list[UniverseSnapshot]:
    """
    Build universe snapshots at monthly intervals between *start* and *end*.
    First snapshot uses data up to the first full month after *start*.
    """
    all_coins = list(candles_map.keys())

    snapshots: list[UniverseSnapshot] = []
    # Align to first-of-month boundaries
    cursor = pd.Timestamp(start).tz_localize("UTC") if start.tzinfo is None else pd.Timestamp(start)
    cursor = cursor.normalize() + pd.offsets.MonthBegin(1)
    end_ts = pd.Timestamp(end).tz_localize("UTC") if end.tzinfo is None else pd.Timestamp(end)

    while cursor <= end_ts:
        snap = select_universe_at(cursor.to_pydatetime(), all_coins, candles_map)
        snapshots.append(snap)
        cursor += pd.offsets.MonthBegin(1)

    logger.info("Built %d monthly universe snapshots", len(snapshots))
    return snapshots


def get_universe_for_date(
    date: pd.Timestamp,
    snapshots: list[UniverseSnapshot],
) -> list[str]:
    """Return the most recent universe snapshot valid for *date*."""
    if not snapshots:
        return []
    date_ts = date if date.tzinfo is not None else date.tz_localize("UTC")
    valid = [s for s in snapshots if pd.Timestamp(s.date) <= date_ts]
    if not valid:
        return snapshots[0].coins
    return valid[-1].coins


def print_universe_summary(snapshots: list[UniverseSnapshot]) -> None:
    """Pretty-print universe history to console. Shows first populated month and latest 3."""
    populated = [s for s in snapshots if s.coins]
    if not populated:
        print("\nNo populated universe snapshots found.")
        return

    print("\n" + "=" * 65)
    print("UNIVERSE SELECTION SUMMARY")
    print(f"  Total monthly snapshots: {len(snapshots)} "
          f"| First with data: {populated[0].date.strftime('%Y-%m')}")
    print("=" * 65)

    # Show first populated month + a few middle + last 3
    to_show = []
    to_show.append(populated[0])
    if len(populated) > 4:
        mid = populated[len(populated) // 2]
        if mid not in to_show:
            to_show.append(mid)
    for s in populated[-3:]:
        if s not in to_show:
            to_show.append(s)

    for snap in to_show:
        print(f"\n{snap.date.strftime('%Y-%m-%d')} — top {min(10, len(snap.coins))} of {len(snap.coins)} coins:")
        print(f"  {'Rank':<6} {'Coin':<12} {'30d Notional Volume':>22}")
        print(f"  {'-'*6} {'-'*12} {'-'*22}")
        for i, coin in enumerate(snap.coins[:10], 1):
            vol = snap.volumes.get(coin, 0)
            print(f"  {i:<6} {coin:<12} ${vol:>21,.0f}")
        if len(snap.coins) > 10:
            print(f"  ... and {len(snap.coins) - 10} more")

    print(f"\n{'='*65}\n")
