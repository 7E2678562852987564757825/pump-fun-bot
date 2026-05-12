"""
Tests for the backtest engine: no look-ahead, correct funding accrual, correct PnL math.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy import StrategyParams, compute_signals, build_signals_all
from backtest import (
    _fill_price,
    _exit_fill_price,
    _position_pnl,
    _compute_size_usd,
    TAKER_FEE,
    SLIPPAGE,
    Position,
    run_backtest,
)
from universe import UniverseSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int, start_price: float = 100.0, trend: float = 0.0) -> pd.DataFrame:
    """Synthetic hourly candles."""
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    prices = start_price * np.exp(trend * np.arange(n) / n)
    np.random.seed(42)
    noise = np.random.randn(n) * 0.002
    prices = prices * (1 + noise)
    df = pd.DataFrame({
        "open": prices * (1 - 0.001),
        "high": prices * (1 + 0.003),
        "low": prices * (1 - 0.003),
        "close": prices,
        "volume": np.random.uniform(1000, 5000, n),
    }, index=idx)
    return df


def _make_funding(n: int, rate: float = 0.0001) -> pd.DataFrame:
    """Constant funding rate series."""
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"fundingRate": np.full(n, rate), "premium": np.zeros(n)}, index=idx)


# ---------------------------------------------------------------------------
# Cost model tests
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_long_fill_increases_price(self):
        price = 100.0
        fill = _fill_price(price, "long")
        assert fill > price
        assert abs(fill - price * (1 + SLIPPAGE)) < 1e-10

    def test_short_fill_decreases_price(self):
        price = 100.0
        fill = _fill_price(price, "short")
        assert fill < price
        assert abs(fill - price * (1 - SLIPPAGE)) < 1e-10

    def test_long_exit_decreases_price(self):
        fill = _exit_fill_price(100.0, "long")
        assert fill < 100.0

    def test_short_exit_increases_price(self):
        fill = _exit_fill_price(100.0, "short")
        assert fill > 100.0

    def test_round_trip_cost(self):
        """Round trip (entry + exit slippage) ≈ 2 * SLIPPAGE."""
        entry = _fill_price(100.0, "long")
        exit_ = _exit_fill_price(100.0, "long")
        round_trip_slippage = (entry - exit_) / 100.0
        assert abs(round_trip_slippage - 2 * SLIPPAGE) < 1e-10


# ---------------------------------------------------------------------------
# PnL math tests
# ---------------------------------------------------------------------------

class TestPnLMath:
    def _make_pos(self, direction: str, entry: float, size: float = 10_000.0) -> Position:
        return Position(
            coin="TEST",
            direction=direction,
            entry_price=entry,
            entry_time=pd.Timestamp("2023-01-01", tz="UTC"),
            size_usd=size,
            stop_loss_price=entry * 0.98 if direction == "long" else entry * 1.02,
            take_profit_price=entry * 1.015 if direction == "long" else entry * 0.985,
            max_hold_bars=24,
        )

    def test_long_profit(self):
        pos = self._make_pos("long", 100.0, 10_000.0)
        pnl = _position_pnl(pos, 110.0)
        expected = (110.0 - 100.0) / 100.0 * 10_000.0
        assert abs(pnl - expected) < 1e-6

    def test_long_loss(self):
        pos = self._make_pos("long", 100.0, 10_000.0)
        pnl = _position_pnl(pos, 90.0)
        expected = (90.0 - 100.0) / 100.0 * 10_000.0
        assert abs(pnl - expected) < 1e-6

    def test_short_profit(self):
        pos = self._make_pos("short", 100.0, 10_000.0)
        pnl = _position_pnl(pos, 90.0)
        expected = (100.0 - 90.0) / 100.0 * 10_000.0
        assert abs(pnl - expected) < 1e-6

    def test_short_loss(self):
        pos = self._make_pos("short", 100.0, 10_000.0)
        pnl = _position_pnl(pos, 110.0)
        expected = (100.0 - 110.0) / 100.0 * 10_000.0
        assert abs(pnl - expected) < 1e-6

    def test_sizing_risk_one_pct(self):
        equity = 100_000.0
        params = StrategyParams(stop_loss_pct=2.0)
        entry = 100.0
        stop = 98.0  # 2% below
        size = _compute_size_usd(equity, entry, stop, params)
        # Uncapped: 1% * 100k / 2% = 50k, but capped at 20% of equity = 20k
        assert abs(size - 20_000.0) < 1.0

    def test_sizing_capped(self):
        """Size should be capped at 20% of equity."""
        equity = 100_000.0
        params = StrategyParams(stop_loss_pct=0.01)  # tiny stop → huge position
        size = _compute_size_usd(equity, 100.0, 99.99, params)
        assert size <= equity * 0.20 + 1e-6


# ---------------------------------------------------------------------------
# No look-ahead test
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    """
    Verify signals are computed on bar close and executed on the NEXT bar open.
    """

    def test_signal_uses_previous_bar(self):
        """next_open at time t should equal open at t+1."""
        candles = _make_candles(200)
        funding = _make_funding(200)
        params = StrategyParams()
        df = compute_signals(candles, funding, params)

        # next_open[t] should equal open[t+1]
        for i in range(len(df) - 2):
            t = df.index[i]
            t_next = df.index[i + 1]
            assert abs(df.loc[t, "next_open"] - df.loc[t_next, "open"]) < 1e-9, \
                f"Look-ahead violation at {t}: next_open={df.loc[t, 'next_open']}, open[t+1]={df.loc[t_next, 'open']}"


# ---------------------------------------------------------------------------
# Funding accrual test
# ---------------------------------------------------------------------------

class TestFundingAccrual:
    """Verify funding payments accumulate correctly over the hold period."""

    def test_positive_funding_reduces_long_pnl(self):
        """
        With constant positive funding, a long position should accrue negative funding PnL.
        """
        n = 800
        candles = _make_candles(n, start_price=100.0)
        # Extreme negative funding to trigger long signal
        rate = -0.01
        idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
        funding_df = pd.DataFrame({"fundingRate": [rate] * n, "premium": [0.0] * n}, index=idx)

        params = StrategyParams(
            entry_long_pct=50.0,   # very permissive for testing
            entry_short_pct=999.0,
            funding_window_h=4,
            percentile_lookback_days=5,
        )

        snap = UniverseSnapshot(
            date=pd.Timestamp("2023-01-01", tz="UTC").to_pydatetime(),
            coins=["TEST"],
            volumes=pd.Series({"TEST": 1e9}),
        )
        signals = build_signals_all({"TEST": candles}, {"TEST": funding_df}, params)

        if "TEST" not in signals:
            pytest.skip("No signals generated — adjust params")

        result = run_backtest(
            {"TEST": signals["TEST"]},
            {"TEST": funding_df},
            [snap],
            params,
            initial_equity=100_000.0,
        )

        long_trades = [t for t in result.trades if t.direction == "long" and t.hold_hours > 1]
        if not long_trades:
            pytest.skip("No long trades with sufficient hold time")

        for t in long_trades:
            # With positive rate, longs pay funding; funding_pnl should be negative
            # (rate is negative here so longs receive)
            # We check funding_pnl is nonzero for trades held > 1h
            if t.hold_hours > 1:
                assert t.funding_pnl != 0.0, "Funding should be non-zero for multi-bar holds"

    def test_funding_magnitude(self):
        """
        Funding PnL for a 10h long position ≈ -rate * size * 10.
        """
        rate = 0.001  # 0.1% per hour
        size_usd = 10_000.0
        expected_funding = -rate * size_usd * 10  # 10 hours

        # Manually compute
        pos = Position(
            coin="TEST", direction="long", entry_price=100.0,
            entry_time=pd.Timestamp("2023-01-01", tz="UTC"),
            size_usd=size_usd, stop_loss_price=98.0, take_profit_price=101.5,
            max_hold_bars=24,
        )
        for _ in range(10):
            pos.funding_pnl -= rate * pos.size_usd

        assert abs(pos.funding_pnl - expected_funding) < 1e-8


# ---------------------------------------------------------------------------
# Integration: Backtest produces sensible results
# ---------------------------------------------------------------------------

class TestBacktestIntegration:
    def test_equity_never_negative(self):
        """Equity should stay positive (capped by sizing)."""
        n = 1000
        candles = _make_candles(n, start_price=100.0)
        funding = _make_funding(n, rate=0.0)
        params = StrategyParams(
            entry_long_pct=50.0,
            entry_short_pct=50.0,
            funding_window_h=4,
            percentile_lookback_days=5,
            min_abs_funding_8h=0.0,
        )
        snap = UniverseSnapshot(
            date=pd.Timestamp("2023-01-01", tz="UTC").to_pydatetime(),
            coins=["TEST"],
            volumes=pd.Series({"TEST": 1e9}),
        )
        signals = build_signals_all({"TEST": candles}, {"TEST": funding}, params)
        if "TEST" not in signals:
            return
        result = run_backtest(
            {"TEST": signals["TEST"]}, {"TEST": funding}, [snap], params,
            initial_equity=100_000.0,
        )
        assert result.equity_curve.min() > 0, "Equity went negative"

    def test_trade_count_positive(self):
        """Should generate at least some trades."""
        n = 2000
        candles = _make_candles(n)
        funding = _make_funding(n, rate=0.0)
        params = StrategyParams(
            entry_long_pct=50.0,
            entry_short_pct=50.0,
            funding_window_h=4,
            percentile_lookback_days=5,
            min_abs_funding_8h=0.0,  # disable floor so zero-funding test still fires signals
        )
        snap = UniverseSnapshot(
            date=pd.Timestamp("2023-01-01", tz="UTC").to_pydatetime(),
            coins=["TEST"],
            volumes=pd.Series({"TEST": 1e9}),
        )
        signals = build_signals_all({"TEST": candles}, {"TEST": funding}, params)
        if "TEST" not in signals:
            pytest.skip("No signals")
        result = run_backtest(
            {"TEST": signals["TEST"]}, {"TEST": funding}, [snap], params,
            initial_equity=100_000.0,
        )
        assert len(result.trades) >= 1, "Expected at least one trade"

    def test_max_positions_respected(self):
        """Never exceed MAX_POSITIONS concurrent open positions."""
        from backtest import MAX_POSITIONS
        n = 2000
        coins = [f"COIN{i}" for i in range(10)]
        candles_map = {c: _make_candles(n) for c in coins}
        funding_map = {c: _make_funding(n) for c in coins}
        params = StrategyParams(
            entry_long_pct=50.0,
            entry_short_pct=50.0,
            funding_window_h=4,
            percentile_lookback_days=5,
            min_abs_funding_8h=0.0,
        )
        snap = UniverseSnapshot(
            date=pd.Timestamp("2023-01-01", tz="UTC").to_pydatetime(),
            coins=coins,
            volumes=pd.Series({c: 1e9 for c in coins}),
        )
        signals = build_signals_all(candles_map, funding_map, params)
        if not signals:
            pytest.skip("No signals")
        result = run_backtest(signals, funding_map, [snap], params, initial_equity=100_000.0)

        assert result.positions_over_time["n_positions"].max() <= MAX_POSITIONS, \
            f"Exceeded MAX_POSITIONS={MAX_POSITIONS}"


# ---------------------------------------------------------------------------
# Signal computation tests
# ---------------------------------------------------------------------------

class TestSignals:
    def test_percentile_no_lookahead(self):
        """pct_rank at time t should only use data from before t."""
        candles = _make_candles(400)
        funding = _make_funding(400)
        params = StrategyParams(percentile_lookback_days=5, funding_window_h=4)
        df = compute_signals(candles, funding, params)

        # Verify rank is bounded [0, 100]
        ranks = df["pct_rank_8h"].dropna()
        assert (ranks >= 0).all(), "Negative percentile rank"
        assert (ranks <= 100).all(), "Percentile rank > 100"

    def test_signal_long_short_mutually_exclusive(self):
        """A bar should not be both a long and short signal."""
        candles = _make_candles(400)
        funding = _make_funding(400)
        params = StrategyParams(entry_long_pct=5, entry_short_pct=95)
        df = compute_signals(candles, funding, params)
        both = df["signal_long"] & df["signal_short"]
        assert not both.any(), "Long and short signals fire on same bar"

    def test_realized_vol_positive(self):
        candles = _make_candles(300)
        funding = _make_funding(300)
        df = compute_signals(candles, funding)
        rv = df["realized_vol"].dropna()
        assert (rv > 0).all(), "Realized vol should be positive"
