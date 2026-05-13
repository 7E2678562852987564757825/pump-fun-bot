"""Event-driven backtest engine with strict no-lookahead enforcement.

Architecture
------------
The engine consumes events from an EventBus in strict timestamp order.
At each simulation step T, it can only see events with ts <= T.
All state mutations (liq map updates, price history, signal computation)
happen AFTER events are popped from the bus — never before.
"""

from __future__ import annotations

import logging
import random
from collections import deque, defaultdict
from typing import Any

import numpy as np
import polars as pl

from hl_liq_cascade.types import (
    Event, EventKind, MarketSnapshot, LiqMap, LiqEvent,
    CombinedSignal, BacktestPosition, TradeOrder,
)
from hl_liq_cascade.backtest.event_bus import EventBus
from hl_liq_cascade.signals.cascade_frontrun import CascadeFrontrunSignal
from hl_liq_cascade.signals.postcascade_fade import PostCascadeFadeSignal
from hl_liq_cascade.signals.squeeze import SqueezeSignal
from hl_liq_cascade.signals.combiner import SignalCombiner
from hl_liq_cascade.risk.sizing import PositionSizer
from hl_liq_cascade.risk.stops import StopManager
from hl_liq_cascade.risk.exposure import ExposureManager

logger = logging.getLogger(__name__)

# Maximum price history length per coin (6000 ≈ 100min at 1s ticks)
_MAX_PRICE_HIST = 6_000
# Maximum liq event buffer per coin (30-day rolling)
_MAX_LIQ_EVENTS = 100_000
# Maximum OI history per coin
_MAX_OI_HIST = 500

_MAKER_TIMEOUT_MS = 30_000   # 30 seconds
_EXEC_LATENCY_MS = 200


class BacktestEngine:
    """Strict no-lookahead event-driven backtest engine."""

    def __init__(self, cfg: dict, initial_capital: float = 100_000) -> None:
        self.cfg = cfg
        self.equity = initial_capital
        self._initial_capital = initial_capital
        self._rng = random.Random(cfg.get("backtest", {}).get("seed", 42))

        sig_cfg = cfg.get("signals", {})
        exec_cfg = cfg.get("execution", {})
        risk_cfg = cfg.get("risk", {})

        self._signals_a = CascadeFrontrunSignal(sig_cfg.get("cascade_frontrun", {}))
        self._signals_b = PostCascadeFadeSignal(sig_cfg.get("postcascade_fade", {}))
        self._signals_c = SqueezeSignal(sig_cfg.get("squeeze", {}))
        self._combiner = SignalCombiner(sig_cfg.get("weights", {
            "cascade_frontrun": 0.4, "postcascade_fade": 0.35, "squeeze": 0.25,
        }))
        self._min_score: float = sig_cfg.get("min_combined_score", 0.3)

        self._sizer = PositionSizer(risk_cfg)
        self._stop_mgr = StopManager(risk_cfg)
        self._exposure_mgr = ExposureManager(risk_cfg)

        self._hard_stop_pct: float = risk_cfg.get("hard_stop_pct", 0.03)
        self._major_coins: set[str] = set(exec_cfg.get("major_coins", ["BTC", "ETH"]))
        self._slippage_major = exec_cfg.get("slippage_major_bps", 2) / 10_000
        self._slippage_alt = exec_cfg.get("slippage_alt_bps", 5) / 10_000
        self._maker_fill_rate: float = exec_cfg.get("maker_fill_rate", 0.60)

        # Mutable state
        self.open_positions: list[BacktestPosition] = []
        self.closed_positions: list[BacktestPosition] = []
        self._price_hist: dict[str, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=_MAX_PRICE_HIST)
        )
        self._liq_event_buf: dict[str, deque[LiqEvent]] = defaultdict(
            lambda: deque(maxlen=_MAX_LIQ_EVENTS)
        )
        self._oi_hist: dict[str, deque[tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=_MAX_OI_HIST)
        )
        self._atr_cache: dict[str, float] = {}
        self._liq_map_cache: dict[str, LiqMap] = {}
        self._snap_cache: dict[str, MarketSnapshot] = {}
        # (expire_ts, position) for passive orders awaiting fill
        self._pending_makers: list[tuple[int, BacktestPosition]] = []
        # Equity curve: list of (ts, equity)
        self._equity_log: list[tuple[int, float]] = [(0, initial_capital)]
        # Signal score logs for analytics
        self._signal_log: list[dict] = []
        # Last ts processed
        self._last_ts: int = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, bus: EventBus) -> pl.DataFrame:
        """Consume all events from *bus* and return closed trades as DataFrame."""
        logger.info("Starting backtest simulation")

        while not bus.is_empty():
            next_ts = bus.peek_next_ts()
            if next_ts is None:
                break

            batch = list(bus.pop_until(next_ts))
            for event in batch:
                self._dispatch(event)

            self._last_ts = next_ts
            self._check_pending_makers(next_ts)
            self._check_stops(next_ts)
            self._try_generate_signals(next_ts)

        # Force-close all open positions at last known price
        for pos in list(self.open_positions):
            snap = self._snap_cache.get(pos.coin)
            px = snap.mid_px if snap else pos.entry_px
            self._close_position(pos, px, self._last_ts, reason="end_of_backtest")

        logger.info(
            "Backtest complete: %d trades, final equity=%.2f",
            len(self.closed_positions), self.equity,
        )
        return self.results_to_df()

    def _dispatch(self, event: Event) -> None:
        kind = event.kind
        if kind == EventKind.MARKET_TICK:
            self._handle_market_tick(event)
        elif kind == EventKind.LIQ_EVENT:
            self._handle_liq_event(event)
        elif kind == EventKind.LIQ_MAP:
            self._handle_liq_map(event)
        elif kind == EventKind.CANDLE:
            self._handle_candle(event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_market_tick(self, event: Event) -> None:
        payload: dict = event.payload  # type: ignore
        mids: dict[str, float] = payload.get("mids", {})
        for coin, mid in mids.items():
            self._price_hist[coin].append((event.ts, mid))
            # Update snapshot cache mid_px if we have one
            if coin in self._snap_cache:
                snap = self._snap_cache[coin]
                slip = self._slippage_major if coin in self._major_coins else self._slippage_alt
                spread_half = mid * slip / 2
                self._snap_cache[coin] = MarketSnapshot(
                    coin=snap.coin, ts=event.ts, mid_px=mid,
                    bid_px=mid - spread_half, ask_px=mid + spread_half,
                    mark_px=mid, funding_rate=snap.funding_rate,
                    open_interest_long=snap.open_interest_long,
                    open_interest_short=snap.open_interest_short,
                    volume_24h=snap.volume_24h, atr_1h=snap.atr_1h,
                )

    def _handle_liq_event(self, event: Event) -> None:
        ev: LiqEvent = event.payload  # type: ignore
        self._liq_event_buf[ev.coin].append(ev)
        self._signals_b.update_history(event.ts, [ev])

    def _handle_liq_map(self, event: Event) -> None:
        lm: LiqMap = event.payload  # type: ignore
        self._liq_map_cache[lm.coin] = lm

    def _handle_candle(self, event: Event) -> None:
        payload: dict = event.payload  # type: ignore
        coin: str = payload["coin"]
        # Update ATR (EMA of |H-L|, 14-period, stored as single scalar)
        hl = payload.get("h", 0.0) - payload.get("l", 0.0)
        if coin in self._atr_cache:
            alpha = 2 / (14 + 1)
            self._atr_cache[coin] = alpha * hl + (1 - alpha) * self._atr_cache[coin]
        else:
            self._atr_cache[coin] = hl

        # Update MarketSnapshot from candle data
        mid = float(payload.get("c", 0.0))
        if mid > 0:
            slip = self._slippage_major if coin in self._major_coins else self._slippage_alt
            spread_half = mid * slip / 2
            existing = self._snap_cache.get(coin)
            self._snap_cache[coin] = MarketSnapshot(
                coin=coin, ts=event.ts, mid_px=mid,
                bid_px=mid - spread_half, ask_px=mid + spread_half,
                mark_px=mid,
                funding_rate=payload.get("funding_rate", 0.0),
                open_interest_long=payload.get("oi_long", 0.0),
                open_interest_short=payload.get("oi_short", 0.0),
                volume_24h=payload.get("vol_24h", 0.0),
                atr_1h=self._atr_cache.get(coin, 0.0),
            )
            # Update OI history
            long_oi = payload.get("oi_long", 0.0)
            short_oi = payload.get("oi_short", 0.0)
            if long_oi > 0 or short_oi > 0:
                self._oi_hist[coin].append((event.ts, long_oi, short_oi))

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _try_generate_signals(self, ts: int) -> None:
        # Only generate on coins with a liq map and market snapshot
        coins = set(self._liq_map_cache) & set(self._snap_cache)
        for coin in coins:
            lm = self._liq_map_cache[coin]
            snap = self._snap_cache[coin]
            price_hist = list(self._price_hist[coin])
            liq_events = list(self._liq_event_buf[coin])
            oi_hist = list(self._oi_hist[coin])
            atr = self._atr_cache.get(coin, snap.mid_px * 0.01)  # 1% fallback

            signals = []

            sig_a = self._signals_a.compute(coin, ts, lm, snap, price_hist)
            if sig_a.score != 0.0:
                signals.append(sig_a)

            sig_b = self._signals_b.compute(coin, ts, snap, liq_events, price_hist, atr)
            if sig_b.score != 0.0:
                signals.append(sig_b)

            sig_c = self._signals_c.compute(coin, ts, lm, snap, oi_hist)
            if sig_c.score != 0.0:
                signals.append(sig_c)

            if not signals:
                continue

            combined = self._combiner.combine(signals, coin, ts)
            if not self._combiner.filter_by_threshold(combined, self._min_score):
                continue

            self._try_open(combined, snap, ts)

    def _try_open(self, signal: CombinedSignal, snap: MarketSnapshot, ts: int) -> None:
        side = "long" if signal.score > 0 else "short"
        entry_px = snap.ask_px if side == "long" else snap.bid_px
        stop_px = self._sizer.compute_stop(side, entry_px, self._hard_stop_pct)

        atr = snap.atr_1h if snap.atr_1h > 0 else entry_px * 0.01
        size_usd = self._sizer.size(
            self.equity, signal.score, atr, entry_px, stop_px, signal.coin
        )

        if size_usd < 10.0:
            return

        allowed, reason = self._exposure_mgr.can_open(
            signal.coin, side, size_usd, self.equity, self.open_positions
        )
        if not allowed:
            logger.debug("Position blocked by %s for %s %s", reason, side, signal.coin)
            return

        slippage = self._slippage_major if signal.coin in self._major_coins else self._slippage_alt
        actual_entry = entry_px * (1 + slippage) if side == "long" else entry_px * (1 - slippage)

        # Build signal_source from dominant component
        dominant = max(signal.components.values(), key=lambda s: abs(s.score) * signal.weights.get(s.source, 1))
        cascade_notional = dominant.meta.get("cascade_notional", 0.0)

        pos = BacktestPosition(
            coin=signal.coin,
            side=side,
            size_usd=size_usd,
            entry_px=entry_px,
            entry_ts=ts,
            stop_px=stop_px,
            signal_source=dominant.source,
            maker=False,
            fill_ts=ts + _EXEC_LATENCY_MS,
            slippage=size_usd * slippage,
        )
        # Stash cascade_notional for analytics
        object.__setattr__(pos, "_cascade_notional", cascade_notional)

        self.open_positions.append(pos)
        logger.debug(
            "Opened %s %s %.0f USD @ %.4f (stop %.4f) [%s score=%.3f]",
            side, signal.coin, size_usd, actual_entry, stop_px, dominant.source, signal.score,
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _check_stops(self, ts: int) -> None:
        for pos in list(self.open_positions):
            snap = self._snap_cache.get(pos.coin)
            if snap is None:
                continue

            if self._stop_mgr.check_hard_stop(pos, snap.mid_px):
                close_px = self._stop_mgr.stop_fill_px(pos, snap.mid_px, snap.bid_px, snap.ask_px)
                self._close_position(pos, close_px, ts, reason="hard_stop")
            elif self._stop_mgr.check_time_stop(pos, ts):
                self._close_position(pos, snap.mid_px, ts, reason="time_stop")

    def _check_pending_makers(self, ts: int, snap: MarketSnapshot | None = None) -> None:
        still_pending = []
        for expire_ts, pos in self._pending_makers:
            coin_snap = self._snap_cache.get(pos.coin)
            if coin_snap is None:
                still_pending.append((expire_ts, pos))
                continue

            # Attempt fill (60% fill rate)
            seed_val = hash((ts, pos.coin, pos.entry_ts)) & 0xFFFF_FFFF
            self._rng.seed(seed_val)
            if self._rng.random() < self._maker_fill_rate:
                pos.fill_ts = ts
                pos.entry_px = coin_snap.mid_px
                self.open_positions.append(pos)
            elif ts > expire_ts:
                # Convert to taker
                slip = self._slippage_major if pos.coin in self._major_coins else self._slippage_alt
                taker_px = coin_snap.ask_px * (1 + slip) if pos.side == "long" else coin_snap.bid_px * (1 - slip)
                pos.fill_ts = ts
                pos.entry_px = taker_px
                pos.maker = False
                pos.slippage = pos.size_usd * slip
                self.open_positions.append(pos)
            else:
                still_pending.append((expire_ts, pos))
        self._pending_makers = still_pending

    def _close_position(self, pos: BacktestPosition, px: float, ts: int, reason: str) -> None:
        if pos in self.open_positions:
            self.open_positions.remove(pos)

        entry = pos.entry_px
        if entry <= 0:
            return

        ret = (px - entry) / entry
        if pos.side == "short":
            ret = -ret

        pnl_raw = pos.size_usd * ret
        slip = self._slippage_major if pos.coin in self._major_coins else self._slippage_alt
        exit_slippage = pos.size_usd * slip
        # HL taker fee ~0.02% per side (for BTC/ETH), ~0.05% for alts
        fee_rate = 0.0002 if pos.coin in self._major_coins else 0.0005
        fees = pos.size_usd * fee_rate * 2  # entry + exit

        pos.exit_px = px
        pos.exit_ts = ts
        pos.exit_reason = reason
        pos.pnl = pnl_raw - fees - exit_slippage
        pos.fees = fees
        pos.slippage += exit_slippage

        self.equity += pos.pnl
        self.closed_positions.append(pos)
        self._equity_log.append((ts, self.equity))

        logger.debug(
            "Closed %s %s @ %.4f (entry %.4f) reason=%s pnl=%.2f",
            pos.side, pos.coin, px, entry, reason, pos.pnl,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def results_to_df(self) -> pl.DataFrame:
        if not self.closed_positions:
            return pl.DataFrame()
        rows = []
        for pos in self.closed_positions:
            rows.append({
                "coin": pos.coin,
                "side": pos.side,
                "size_usd": pos.size_usd,
                "entry_px": pos.entry_px,
                "exit_px": pos.exit_px,
                "entry_ts": pos.entry_ts,
                "exit_ts": pos.exit_ts,
                "exit_reason": pos.exit_reason,
                "pnl": pos.pnl,
                "fees": pos.fees,
                "slippage": pos.slippage,
                "signal_source": pos.signal_source,
                "cascade_notional": getattr(pos, "_cascade_notional", 0.0),
                "maker": pos.maker,
            })
        return pl.DataFrame(rows)

    def equity_curve(self) -> pl.DataFrame:
        if not self._equity_log:
            return pl.DataFrame({"ts": [], "equity": []})
        ts_list, eq_list = zip(*self._equity_log)
        return pl.DataFrame({"ts": list(ts_list), "equity": list(eq_list)})
