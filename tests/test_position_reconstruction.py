"""Tests for position reconstruction correctness.

These tests verify that fill replay produces positions consistent with
the HL clearinghouse state API. Uses synthetic fill sequences with
known outcomes to validate the math.
"""

from __future__ import annotations

import pytest
import polars as pl
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any


# ---------------------------------------------------------------------------
# Helpers: build synthetic Fill structs
# ---------------------------------------------------------------------------

def _make_fill(side: str, sz: float, px: float, time: int, dir_: str, coin: str = "SOL"):
    """Build a Fill msgspec struct matching HL API schema."""
    from hl_liq_cascade.types import Fill
    return Fill(
        coin=coin,
        px=str(px),
        sz=str(sz),
        side=side,
        time=time,
        startPosition="0",
        dir=dir_,
        closedPnl="0",
        hash=f"0x{time:016x}",
        oid=time,
        crossed=True,
        fee=str(sz * px * 0.0002),
        tid=time,
        feeToken="USDC",
        liquidation=None,
    )


# ---------------------------------------------------------------------------
# Position math tests (pure functions, using module-level _apply_fill)
# ---------------------------------------------------------------------------

class TestPositionMath:
    """Test position size and entry_px computation from fill replay."""

    def _pos(self, size: float = 0.0, entry_px: float = 0.0):
        from hl_liq_cascade.data.position_store import _CoinPosition
        return _CoinPosition(size=size, entry_px=entry_px, realized_pnl=0.0, total_fees=0.0)

    def test_open_long(self) -> None:
        """Open a long position: size = +sz, entry = fill.px."""
        from hl_liq_cascade.data.position_store import _apply_fill

        pos = self._pos()
        fill = _make_fill("B", 10.0, 100.0, 1000, "Open Long")
        snapshots = _apply_fill(pos, fill)

        assert pos.size == pytest.approx(10.0)
        assert pos.entry_px == pytest.approx(100.0)
        assert len(snapshots) == 1

    def test_add_to_long(self) -> None:
        """Add to existing long: VWAP entry price."""
        from hl_liq_cascade.data.position_store import _apply_fill

        pos = self._pos(size=10.0, entry_px=100.0)
        fill = _make_fill("B", 10.0, 120.0, 2000, "Open Long")
        _apply_fill(pos, fill)

        assert pos.size == pytest.approx(20.0)
        # VWAP: (10*100 + 10*120) / 20 = 110
        assert pos.entry_px == pytest.approx(110.0)

    def test_partial_close_long(self) -> None:
        """Close part of long: reduce size, realize PnL, entry_px unchanged."""
        from hl_liq_cascade.data.position_store import _apply_fill

        pos = self._pos(size=10.0, entry_px=100.0)
        fill = _make_fill("A", 5.0, 120.0, 2000, "Close Long")
        _apply_fill(pos, fill)

        assert pos.size == pytest.approx(5.0)
        assert pos.entry_px == pytest.approx(100.0)  # entry unchanged on partial close
        # realized PnL = (120 - 100) * 5 = 100 (accounted in closedPnl=0 + internal calc)
        # Note: _apply_fill uses fill.closedPnl (from API), but our fill has "0" so pnl from internal calc
        assert pos.realized_pnl == pytest.approx(100.0, abs=1e-6)

    def test_full_close_long(self) -> None:
        """Close entire long: size goes to 0."""
        from hl_liq_cascade.data.position_store import _apply_fill

        pos = self._pos(size=10.0, entry_px=100.0)
        fill = _make_fill("A", 10.0, 90.0, 2000, "Close Long")
        _apply_fill(pos, fill)

        assert pos.size == pytest.approx(0.0)
        assert pos.realized_pnl == pytest.approx(-100.0, abs=1e-6)

    def test_flip_long_to_short(self) -> None:
        """Flip from long to short through a larger sell."""
        from hl_liq_cascade.data.position_store import _apply_fill

        pos = self._pos(size=5.0, entry_px=100.0)
        # "Close Long" of 10 when only 5 long: closes 5 + opens 5 short
        fill = _make_fill("A", 10.0, 110.0, 2000, "Close Long")
        snapshots = _apply_fill(pos, fill)

        assert pos.size == pytest.approx(-5.0)
        assert pos.entry_px == pytest.approx(110.0)
        assert pos.realized_pnl == pytest.approx(50.0, abs=1e-6)
        assert len(snapshots) == 2  # close snapshot + new open snapshot

    def test_open_short(self) -> None:
        """Open short: size = -sz."""
        from hl_liq_cascade.data.position_store import _apply_fill

        pos = self._pos()
        fill = _make_fill("A", 10.0, 100.0, 1000, "Open Short")
        _apply_fill(pos, fill)

        assert pos.size == pytest.approx(-10.0)
        assert pos.entry_px == pytest.approx(100.0)

    def test_close_short_profit(self) -> None:
        """Close short at lower price → profit."""
        from hl_liq_cascade.data.position_store import _apply_fill

        pos = self._pos(size=-10.0, entry_px=100.0)
        fill = _make_fill("B", 10.0, 80.0, 2000, "Close Short")
        _apply_fill(pos, fill)

        assert pos.size == pytest.approx(0.0)
        assert pos.realized_pnl == pytest.approx(200.0, abs=1e-6)  # (100-80)*10


# ---------------------------------------------------------------------------
# Liquidation price tests
# ---------------------------------------------------------------------------

class TestLiquidationPriceComputation:

    def _store(self):
        from hl_liq_cascade.data.position_store import PositionStore, MAINTENANCE_MARGIN
        store = PositionStore.__new__(PositionStore)
        return store, MAINTENANCE_MARGIN

    def test_isolated_long_liq_px(self) -> None:
        store, mm = self._store()
        lev = 10
        entry = 100.0
        liq = store.compute_liq_px(size=1.0, entry_px=entry, leverage=lev, lev_type="isolated")
        expected = entry * (1 - 1 / lev + mm)
        assert liq == pytest.approx(expected)

    def test_isolated_short_liq_px(self) -> None:
        store, mm = self._store()
        lev = 10
        entry = 100.0
        liq = store.compute_liq_px(size=-1.0, entry_px=entry, leverage=lev, lev_type="isolated")
        expected = entry * (1 + 1 / lev - mm)
        assert liq == pytest.approx(expected)

    def test_cross_margin_returns_none(self) -> None:
        store, _ = self._store()
        liq = store.compute_liq_px(size=1.0, entry_px=100.0, leverage=5, lev_type="cross")
        assert liq is None

    def test_liq_below_entry_for_long(self) -> None:
        store, _ = self._store()
        liq = store.compute_liq_px(size=10.0, entry_px=100.0, leverage=5, lev_type="isolated")
        assert liq < 100.0

    def test_liq_above_entry_for_short(self) -> None:
        store, _ = self._store()
        liq = store.compute_liq_px(size=-10.0, entry_px=100.0, leverage=5, lev_type="isolated")
        assert liq > 100.0

    def test_higher_leverage_closer_liq(self) -> None:
        store, _ = self._store()
        liq_5x = store.compute_liq_px(size=1.0, entry_px=100.0, leverage=5, lev_type="isolated")
        liq_20x = store.compute_liq_px(size=1.0, entry_px=100.0, leverage=20, lev_type="isolated")
        assert liq_20x > liq_5x  # higher leverage → liq closer to entry (higher for long)


# ---------------------------------------------------------------------------
# Liq map math tests
# ---------------------------------------------------------------------------

class TestLiqMapMath:
    """Validate that liq map buckets are populated correctly."""

    def _make_positions(self) -> list:
        from hl_liq_cascade.types import Position
        return [
            # Long at 100, liq at 85
            Position(address="0xA", coin="SOL", size=100.0, entry_px=100.0,
                     liq_px=85.0, margin_used=1000.0, unrealized_pnl=0.0,
                     leverage=10, leverage_type="isolated", snapshot_ts=1000),
            # Long at 100, liq at 88
            Position(address="0xB", coin="SOL", size=50.0, entry_px=100.0,
                     liq_px=88.0, margin_used=500.0, unrealized_pnl=0.0,
                     leverage=10, leverage_type="isolated", snapshot_ts=1000),
            # Short at 100, liq at 118
            Position(address="0xC", coin="SOL", size=-80.0, entry_px=100.0,
                     liq_px=118.0, margin_used=800.0, unrealized_pnl=0.0,
                     leverage=10, leverage_type="isolated", snapshot_ts=1000),
        ]

    def test_long_liq_in_correct_side(self) -> None:
        from hl_liq_cascade.liq_map.builder import LiqMapBuilder
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        builder = LiqMapBuilder(cfg["liq_map"])
        positions = self._make_positions()
        mids = {"SOL": 100.0}
        liq_maps = builder.build(positions, mids, ts=1000)

        assert "SOL" in liq_maps
        lm = liq_maps["SOL"]

        # All long liq notional should be in buckets BELOW 100 (price < 100)
        long_below = sum(b.long_notional for b in lm.buckets if b.price < 100.0)
        long_above = sum(b.long_notional for b in lm.buckets if b.price >= 100.0)
        assert long_below > 0
        assert long_above == 0.0

    def test_short_liq_in_correct_side(self) -> None:
        from hl_liq_cascade.liq_map.builder import LiqMapBuilder
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        builder = LiqMapBuilder(cfg["liq_map"])
        positions = self._make_positions()
        mids = {"SOL": 100.0}
        liq_maps = builder.build(positions, mids, ts=1000)
        lm = liq_maps["SOL"]

        # All short liq notional should be in buckets ABOVE 100
        short_above = sum(b.short_notional for b in lm.buckets if b.price > 100.0)
        short_below = sum(b.short_notional for b in lm.buckets if b.price <= 100.0)
        assert short_above > 0
        assert short_below == 0.0

    def test_notional_magnitude(self) -> None:
        from hl_liq_cascade.liq_map.builder import LiqMapBuilder
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        builder = LiqMapBuilder(cfg["liq_map"])
        positions = self._make_positions()
        mids = {"SOL": 100.0}
        liq_maps = builder.build(positions, mids, ts=1000)
        lm = liq_maps["SOL"]

        # Total long notional ≈ (100*85 + 50*88)
        expected_long = 100.0 * 85.0 + 50.0 * 88.0
        assert lm.total_long_notional == pytest.approx(expected_long, rel=0.01)

        # Total short notional ≈ 80*118
        expected_short = 80.0 * 118.0
        assert lm.total_short_notional == pytest.approx(expected_short, rel=0.01)


# ---------------------------------------------------------------------------
# No-lookahead enforcement tests
# ---------------------------------------------------------------------------

class TestNoLookahead:
    """Verify the EventBus strictly enforces temporal ordering."""

    def test_cannot_push_past_event(self) -> None:
        from hl_liq_cascade.backtest.event_bus import EventBus
        from hl_liq_cascade.types import Event, EventKind

        bus = EventBus()
        e1 = Event(ts=1000, kind=EventKind.MARKET_TICK, payload={})
        e2 = Event(ts=2000, kind=EventKind.MARKET_TICK, payload={})
        e_past = Event(ts=500, kind=EventKind.MARKET_TICK, payload={})

        bus.push(e1)
        bus.push(e2)
        list(bus.pop_until(1000))  # advance to ts=1000

        with pytest.raises(ValueError, match="retroactive"):
            bus.push(e_past)

    def test_events_emitted_in_order(self) -> None:
        from hl_liq_cascade.backtest.event_bus import EventBus
        from hl_liq_cascade.types import Event, EventKind

        bus = EventBus()
        timestamps = [5000, 1000, 3000, 2000, 4000]
        for ts in timestamps:
            bus.push(Event(ts=ts, kind=EventKind.MARKET_TICK, payload=ts))

        received = []
        while not bus.is_empty():
            events = list(bus.pop_until(bus.peek_next_ts()))  # type: ignore
            received.extend(events)

        assert [e.ts for e in received] == sorted(timestamps)

    def test_pop_until_respects_boundary(self) -> None:
        from hl_liq_cascade.backtest.event_bus import EventBus
        from hl_liq_cascade.types import Event, EventKind

        bus = EventBus()
        for ts in [100, 200, 300, 400, 500]:
            bus.push(Event(ts=ts, kind=EventKind.MARKET_TICK, payload=ts))

        received = list(bus.pop_until(300))
        assert all(e.ts <= 300 for e in received)
        assert len(received) == 3

        remaining = list(bus.pop_until(500))
        assert len(remaining) == 2

    def test_current_ts_monotone(self) -> None:
        from hl_liq_cascade.backtest.event_bus import EventBus
        from hl_liq_cascade.types import Event, EventKind

        bus = EventBus()
        bus.push(Event(ts=1000, kind=EventKind.MARKET_TICK, payload={}))
        list(bus.pop_until(1000))
        assert bus._current_ts == 1000

        bus.push(Event(ts=2000, kind=EventKind.MARKET_TICK, payload={}))
        list(bus.pop_until(2000))
        assert bus._current_ts == 2000


# ---------------------------------------------------------------------------
# PnL math tests
# ---------------------------------------------------------------------------

class TestPnLMath:

    def _make_engine(self):
        from hl_liq_cascade.backtest.engine import BacktestEngine
        from hl_liq_cascade.config import load_config
        cfg = load_config()
        return BacktestEngine(cfg, initial_capital=100_000.0)

    def test_long_profit(self) -> None:
        """Long 100 units at 50, close at 60 → PnL ≈ 1000 minus fees."""
        from hl_liq_cascade.types import BacktestPosition

        engine = self._make_engine()
        pos = BacktestPosition(
            coin="SOL",
            side="long",
            size_usd=5000.0,  # 100 units at $50
            entry_px=50.0,
            entry_ts=1000,
            stop_px=47.0,
            signal_source="cascade_frontrun",
            fill_ts=1000,
        )

        engine._close_position(pos, px=60.0, ts=2000, reason="signal")
        # Gross PnL = (60-50)/50 * 5000 = 1000
        # Net PnL = gross - fees - exit_slippage (should be close to 1000)
        assert pos.pnl == pytest.approx(1000.0, rel=0.05)  # within 5% (fees eat a bit)

    def test_short_profit(self) -> None:
        """Short $5000 at 100, close at 80 → PnL ≈ 1000 minus fees."""
        from hl_liq_cascade.types import BacktestPosition

        engine = self._make_engine()
        pos = BacktestPosition(
            coin="BTC",
            side="short",
            size_usd=5000.0,
            entry_px=100.0,
            entry_ts=1000,
            stop_px=103.0,
            signal_source="squeeze",
            fill_ts=1000,
        )

        engine._close_position(pos, px=80.0, ts=2000, reason="signal")
        # Gross PnL = (100-80)/100 * 5000 = 1000 for short
        assert pos.pnl == pytest.approx(1000.0, rel=0.05)

    def test_stop_loss_respected(self) -> None:
        """Stop loss triggers at correct price."""
        from hl_liq_cascade.risk.stops import StopManager
        from hl_liq_cascade.types import BacktestPosition
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        mgr = StopManager(cfg["risk"])

        pos = BacktestPosition(
            coin="SOL", side="long", size_usd=1000.0, entry_px=100.0,
            entry_ts=0, stop_px=97.0, signal_source="cascade_frontrun", fill_ts=0,
        )

        assert not mgr.check_hard_stop(pos, 98.0)
        assert mgr.check_hard_stop(pos, 97.0)
        assert mgr.check_hard_stop(pos, 95.0)

    def test_time_stop_triggers(self) -> None:
        from hl_liq_cascade.risk.stops import StopManager
        from hl_liq_cascade.types import BacktestPosition
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        mgr = StopManager(cfg["risk"])

        entry_ts = 0
        pos = BacktestPosition(
            coin="SOL", side="long", size_usd=1000.0, entry_px=100.0,
            entry_ts=entry_ts, stop_px=97.0, signal_source="squeeze", fill_ts=entry_ts,
        )

        one_hour_ms = 3_600_000
        assert not mgr.check_time_stop(pos, entry_ts + one_hour_ms - 1)
        assert mgr.check_time_stop(pos, entry_ts + one_hour_ms)


# ---------------------------------------------------------------------------
# Risk / sizing tests
# ---------------------------------------------------------------------------

class TestPositionSizing:

    def test_basic_size_calculation(self) -> None:
        from hl_liq_cascade.risk.sizing import PositionSizer
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        sizer = PositionSizer(cfg["risk"])

        equity = 100_000.0
        # 0.5% risk on $100k = $500 risk
        # entry=100, stop=97 → stop distance = 3
        # coins = 500/3 ≈ 166.67
        # size_usd = 166.67 * 100 = $16,667
        size_usd = sizer.size(equity, 1.0, atr=3.0, entry_px=100.0, stop_px=97.0, coin="SOL")
        expected = (equity * 0.005) / 3.0 * 100.0
        assert size_usd == pytest.approx(expected, rel=0.01)

    def test_size_capped_by_max_exposure(self) -> None:
        from hl_liq_cascade.risk.sizing import PositionSizer
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        sizer = PositionSizer(cfg["risk"])

        # Very tight stop → huge size → should be capped
        equity = 100_000.0
        size_usd = sizer.size(equity, 1.0, atr=0.01, entry_px=100.0, stop_px=99.99, coin="SOL")
        assert size_usd <= equity * 0.30

    def test_score_scales_size(self) -> None:
        from hl_liq_cascade.risk.sizing import PositionSizer
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        sizer = PositionSizer(cfg["risk"])

        equity = 100_000.0
        full = sizer.size(equity, 1.0, atr=3.0, entry_px=100.0, stop_px=97.0, coin="SOL")
        half = sizer.size(equity, 0.5, atr=3.0, entry_px=100.0, stop_px=97.0, coin="SOL")
        assert half == pytest.approx(full * 0.5, rel=0.01)


# ---------------------------------------------------------------------------
# Exposure manager tests
# ---------------------------------------------------------------------------

class TestExposureManager:

    def _open_pos(self, coin: str, side: str, size_usd: float) -> "BacktestPosition":
        from hl_liq_cascade.types import BacktestPosition
        return BacktestPosition(
            coin=coin, side=side, size_usd=size_usd, entry_px=100.0,
            entry_ts=0, stop_px=97.0, signal_source="squeeze", fill_ts=0,
        )

    def test_max_concurrent_enforced(self) -> None:
        from hl_liq_cascade.risk.exposure import ExposureManager
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        mgr = ExposureManager(cfg["risk"])

        positions = [
            self._open_pos("BTC", "long", 10_000),
            self._open_pos("ETH", "long", 10_000),
            self._open_pos("SOL", "long", 10_000),
        ]

        allowed, reason = mgr.can_open("AVAX", "long", 5_000, 100_000, positions)
        assert not allowed
        assert "concurrent" in reason

    def test_correlation_cap_enforced(self) -> None:
        from hl_liq_cascade.risk.exposure import ExposureManager
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        mgr = ExposureManager(cfg["risk"])

        positions = [
            self._open_pos("SOL", "long", 10_000),
            self._open_pos("AVAX", "long", 10_000),
        ]

        # SOL, AVAX, APT are correlated — adding SUI long should be blocked
        allowed, reason = mgr.can_open("SUI", "long", 5_000, 100_000, positions)
        assert not allowed
        assert "correlation" in reason

    def test_allows_opposite_side_correlated(self) -> None:
        from hl_liq_cascade.risk.exposure import ExposureManager
        from hl_liq_cascade.config import load_config

        cfg = load_config()
        mgr = ExposureManager(cfg["risk"])

        # Two longs in alt group — short should be allowed (different direction)
        positions = [
            self._open_pos("SOL", "long", 10_000),
            self._open_pos("AVAX", "long", 10_000),
        ]

        allowed, reason = mgr.can_open("SUI", "short", 5_000, 100_000, positions)
        assert allowed
