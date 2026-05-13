"""Shared types for the HL liquidation cascade framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import msgspec


# ---------------------------------------------------------------------------
# HL API response types (parsed with msgspec for speed)
# ---------------------------------------------------------------------------

class AssetMeta(msgspec.Struct):
    name: str
    szDecimals: int
    maxLeverage: int
    onlyIsolated: bool = False


class AssetCtx(msgspec.Struct):
    funding: str
    openInterest: str
    prevDayPx: str
    dayNtlVlm: str
    premium: Optional[str]
    oraclePx: str
    markPx: str
    midPx: Optional[str]
    impactPxs: Optional[list[str]]
    dayBaseVlm: Optional[str] = None


class PositionData(msgspec.Struct):
    coin: str           # e.g. "BTC"
    szi: str            # signed size (+ long, - short)
    entryPx: Optional[str]
    positionValue: str
    unrealizedPnl: str
    returnOnEquity: str
    liquidationPx: Optional[str]
    marginUsed: str
    maxLeverage: int
    leverage: dict       # {"type": "cross"|"isolated", "value": int}
    cumFunding: dict


class AssetPosition(msgspec.Struct):
    position: PositionData
    type: str            # "oneWay"


class MarginSummary(msgspec.Struct):
    accountValue: str
    totalNtlPos: str
    totalRawUsd: str
    totalMarginUsed: str


class ClearinghouseState(msgspec.Struct):
    assetPositions: list[AssetPosition]
    crossMaintenanceMarginUsed: str
    crossMarginSummary: MarginSummary
    marginSummary: MarginSummary
    withdrawable: str
    time: int


class Fill(msgspec.Struct):
    coin: str
    px: str
    sz: str
    side: str            # "B" buy, "A" sell
    time: int
    startPosition: str
    dir: str
    closedPnl: str
    hash: str
    oid: int
    crossed: bool
    fee: str
    tid: int
    feeToken: str = "USDC"
    liquidation: Optional[str] = None
    builderFee: Optional[str] = None
    twapId: Optional[int] = None


class FundingPayment(msgspec.Struct):
    coin: str
    usdc: str           # positive = received
    szi: str            # position size at time
    fundingRate: str
    time: int
    hash: str = ""


class Trade(msgspec.Struct):
    coin: str
    side: str
    px: str
    sz: str
    time: int
    hash: str
    tid: int
    users: list[str] = msgspec.field(default_factory=list)


class Candle(msgspec.Struct):
    t: int              # open time ms
    T: int              # close time ms
    s: str              # coin
    i: str              # interval
    o: str              # open
    c: str              # close
    h: str              # high
    l: str              # low
    v: str              # volume
    n: int              # num trades


# ---------------------------------------------------------------------------
# Internal domain types (dataclasses for mutability)
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Normalised, in-memory representation of one user position."""
    address: str
    coin: str
    size: float          # + = long, - = short (in coin units)
    entry_px: float
    liq_px: Optional[float]
    margin_used: float
    unrealized_pnl: float
    leverage: int
    leverage_type: str   # "cross" | "isolated"
    snapshot_ts: int     # unix ms when reconstructed


@dataclass
class LiqBucket:
    """One price bucket in the liquidation map."""
    price: float
    long_notional: float    # USD notional of long positions that liq here
    short_notional: float   # USD notional of short positions that liq here

    @property
    def net_notional(self) -> float:
        return self.long_notional - self.short_notional


@dataclass
class LiqMap:
    """Liquidation density map for one coin at one point in time."""
    coin: str
    ts: int                  # unix ms
    current_px: float
    buckets: list[LiqBucket]
    bucket_pct: float        # width of each bucket as fraction of price
    total_long_notional: float = 0.0
    total_short_notional: float = 0.0

    def __post_init__(self) -> None:
        if not self.total_long_notional:
            self.total_long_notional = sum(b.long_notional for b in self.buckets)
        if not self.total_short_notional:
            self.total_short_notional = sum(b.short_notional for b in self.buckets)


@dataclass
class ClusterInfo:
    """A detected cluster of liquidations."""
    coin: str
    price: float
    side: str                # "long" (longs get liq'd) or "short"
    notional: float
    distance_pct: float      # from current price
    density_score: float
    ts: int


@dataclass
class LiqEvent:
    """A single liquidation event captured from the trade feed."""
    coin: str
    side: str                # side that was liquidated: "long" or "short"
    size: float
    px: float
    notional: float
    ts: int
    address: Optional[str] = None


@dataclass
class SignalOutput:
    """Output of one signal generator for one coin at one time."""
    coin: str
    ts: int
    score: float             # [-1, 1]  -1=short, +1=long
    source: str              # "cascade_frontrun" | "postcascade_fade" | "squeeze"
    confidence: float        # [0, 1]
    meta: dict = field(default_factory=dict)


@dataclass
class CombinedSignal:
    """Weighted combination of all signal outputs."""
    coin: str
    ts: int
    score: float             # [-1, 1]
    components: dict[str, SignalOutput]
    weights: dict[str, float]


@dataclass
class TradeOrder:
    """Proposed order from the signal."""
    coin: str
    ts: int
    side: str                # "long" | "short"
    size_usd: float
    entry_type: str          # "taker" | "maker"
    signal: CombinedSignal
    stop_px: Optional[float] = None
    take_px: Optional[float] = None


@dataclass
class BacktestPosition:
    """An open position in the backtest engine."""
    coin: str
    side: str
    size_usd: float
    entry_px: float
    entry_ts: int
    stop_px: float
    signal_source: str
    maker: bool = False
    fill_ts: Optional[int] = None
    exit_px: Optional[float] = None
    exit_ts: Optional[int] = None
    exit_reason: Optional[str] = None  # "stop", "time", "signal", "taker_convert"
    pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_ts is None


@dataclass
class MarketSnapshot:
    """Point-in-time market state for one coin."""
    coin: str
    ts: int
    mid_px: float
    bid_px: float
    ask_px: float
    mark_px: float
    funding_rate: float       # per 8h
    open_interest_long: float # USD
    open_interest_short: float
    volume_24h: float
    atr_1h: float = 0.0       # populated lazily

    @property
    def spread(self) -> float:
        return self.ask_px - self.bid_px


# ---------------------------------------------------------------------------
# Event types for the backtest event bus
# ---------------------------------------------------------------------------

class EventKind:
    MARKET_TICK = "market_tick"
    LIQ_EVENT   = "liq_event"
    LIQ_MAP     = "liq_map"
    FILL        = "fill"
    FUNDING     = "funding"
    CANDLE      = "candle"


@dataclass
class Event:
    """Generic timestamped event on the bus."""
    ts: int
    kind: str
    payload: Any

    def __lt__(self, other: "Event") -> bool:
        return self.ts < other.ts
