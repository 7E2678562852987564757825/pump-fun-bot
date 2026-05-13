"""Paper trading engine — real-time signals, no real orders."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from hl_liq_cascade.config import load_config
from hl_liq_cascade.types import Event, EventKind, LiqEvent, BacktestPosition, LiqMap
from hl_liq_cascade.signals.cascade_frontrun import CascadeFrontrunSignal
from hl_liq_cascade.signals.postcascade_fade import PostCascadeFadeSignal
from hl_liq_cascade.signals.squeeze import SqueezeSignal
from hl_liq_cascade.signals.combiner import SignalCombiner
from hl_liq_cascade.risk.sizing import PositionSizer
from hl_liq_cascade.risk.stops import StopManager
from hl_liq_cascade.risk.exposure import ExposureManager
from hl_liq_cascade.live.feed import LiveFeed

logger = logging.getLogger(__name__)

_LOG_DIR = Path("logs/paper_trades")


class PaperTrader:
    def __init__(self, cfg: dict, initial_capital: float = 100_000) -> None:
        self.cfg = cfg
        self.equity = initial_capital
        self._initial = initial_capital

        sig_cfg = cfg.get("signals", {})
        risk_cfg = cfg.get("risk", {})

        self._sig_a = CascadeFrontrunSignal(sig_cfg.get("cascade_frontrun", {}))
        self._sig_b = PostCascadeFadeSignal(sig_cfg.get("postcascade_fade", {}))
        self._sig_c = SqueezeSignal(sig_cfg.get("squeeze", {}))
        self._combiner = SignalCombiner(sig_cfg.get("weights", {
            "cascade_frontrun": 0.4, "postcascade_fade": 0.35, "squeeze": 0.25,
        }))
        self._min_score: float = sig_cfg.get("min_combined_score", 0.3)
        self._sizer = PositionSizer(risk_cfg)
        self._stop_mgr = StopManager(risk_cfg)
        self._exposure_mgr = ExposureManager(risk_cfg)

        self.positions: list[BacktestPosition] = []
        self.trade_log: list[dict] = []

        self._price_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=6000))
        self._liq_events: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._oi_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._snap_cache: dict = {}
        self._liq_map_cache: dict[str, LiqMap] = {}
        self._atr_cache: dict[str, float] = {}

        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = _LOG_DIR / f"session_{int(time.time())}.jsonl"

    async def on_event(self, event: Event) -> None:
        if event.kind == EventKind.MARKET_TICK:
            mids = event.payload.get("mids", {})
            for coin, mid in mids.items():
                self._price_hist[coin].append((event.ts, mid))
        elif event.kind == EventKind.LIQ_EVENT:
            ev: LiqEvent = event.payload
            self._liq_events[ev.coin].append(ev)
            self._sig_b.update_history(event.ts, [ev])
        elif event.kind == EventKind.LIQ_MAP:
            lm: LiqMap = event.payload
            self._liq_map_cache[lm.coin] = lm
        elif event.kind == EventKind.CANDLE:
            p = event.payload
            coin = p["coin"]
            mid = float(p.get("c", 0.0))
            if mid > 0 and coin in self._liq_map_cache:
                from hl_liq_cascade.types import MarketSnapshot
                snap = MarketSnapshot(
                    coin=coin, ts=event.ts, mid_px=mid,
                    bid_px=mid * 0.9995, ask_px=mid * 1.0005,
                    mark_px=mid, funding_rate=p.get("funding_rate", 0.0),
                    open_interest_long=p.get("oi_long", 0.0),
                    open_interest_short=p.get("oi_short", 0.0),
                    volume_24h=p.get("vol_24h", 0.0),
                    atr_1h=self._atr_cache.get(coin, mid * 0.01),
                )
                self._snap_cache[coin] = snap

        # Check stops on every event
        self._check_stops(event.ts)
        # Try signals if we have liq maps
        self._try_signals(event.ts)

    def _check_stops(self, ts: int) -> None:
        for pos in list(self.positions):
            snap = self._snap_cache.get(pos.coin)
            if snap is None:
                continue
            if self._stop_mgr.check_hard_stop(pos, snap.mid_px):
                self._close(pos, snap.bid_px if pos.side == "long" else snap.ask_px, ts, "hard_stop")
            elif self._stop_mgr.check_time_stop(pos, ts):
                self._close(pos, snap.mid_px, ts, "time_stop")

    def _try_signals(self, ts: int) -> None:
        coins = set(self._liq_map_cache) & set(self._snap_cache)
        for coin in coins:
            lm = self._liq_map_cache[coin]
            snap = self._snap_cache[coin]
            ph = list(self._price_hist[coin])
            le = list(self._liq_events[coin])
            oi = list(self._oi_hist[coin])
            atr = self._atr_cache.get(coin, snap.mid_px * 0.01)

            sigs = []
            for sig_fn in [
                lambda: self._sig_a.compute(coin, ts, lm, snap, ph),
                lambda: self._sig_b.compute(coin, ts, snap, le, ph, atr),
                lambda: self._sig_c.compute(coin, ts, lm, snap, oi),
            ]:
                try:
                    s = sig_fn()
                    if s.score != 0.0:
                        sigs.append(s)
                except Exception:
                    pass

            if not sigs:
                continue

            combined = self._combiner.combine(sigs, coin, ts)
            if not self._combiner.filter_by_threshold(combined, self._min_score):
                continue

            side = "long" if combined.score > 0 else "short"
            entry_px = snap.ask_px if side == "long" else snap.bid_px
            stop_px = self._sizer.compute_stop(side, entry_px)
            size_usd = self._sizer.size(self.equity, combined.score, atr, entry_px, stop_px, coin)

            if size_usd < 10.0:
                continue

            allowed, reason = self._exposure_mgr.can_open(coin, side, size_usd, self.equity, self.positions)
            if not allowed:
                continue

            pos = BacktestPosition(
                coin=coin, side=side, size_usd=size_usd, entry_px=entry_px,
                entry_ts=ts, stop_px=stop_px,
                signal_source=max(sigs, key=lambda s: abs(s.score)).source,
                fill_ts=ts,
            )
            self.positions.append(pos)
            self._log_trade("open", pos)
            logger.info("PAPER: Opened %s %s %.0f @ %.4f", side, coin, size_usd, entry_px)

    def _close(self, pos: BacktestPosition, px: float, ts: int, reason: str) -> None:
        if pos in self.positions:
            self.positions.remove(pos)
        ret = (px - pos.entry_px) / pos.entry_px
        if pos.side == "short":
            ret = -ret
        pos.pnl = pos.size_usd * ret
        pos.exit_px = px
        pos.exit_ts = ts
        pos.exit_reason = reason
        self.equity += pos.pnl
        self.trade_log.append({"action": "close", "coin": pos.coin, "pnl": pos.pnl, "reason": reason})
        self._log_trade("close", pos)
        logger.info("PAPER: Closed %s %s pnl=%.2f [%s]", pos.side, pos.coin, pos.pnl, reason)

    def _log_trade(self, action: str, pos: BacktestPosition) -> None:
        record = {
            "action": action, "coin": pos.coin, "side": pos.side,
            "size_usd": pos.size_usd, "entry_px": pos.entry_px,
            "exit_px": pos.exit_px, "pnl": pos.pnl, "ts": pos.exit_ts or pos.entry_ts,
        }
        with self._log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    async def run(self, coins: list[str]) -> None:
        feed = LiveFeed(self.cfg, on_event=self.on_event)
        await feed.start(coins)
        logger.info("Paper trader running. Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(30)
                logger.info(
                    "Status: equity=%.2f open=%d trades=%d",
                    self.equity, len(self.positions), len(self.trade_log),
                )
        except asyncio.CancelledError:
            pass
        finally:
            await feed.stop()

    def status(self) -> dict:
        return {
            "equity": self.equity,
            "pnl_today": self.equity - self._initial,
            "open_positions": len(self.positions),
            "total_trades": len(self.trade_log),
        }
