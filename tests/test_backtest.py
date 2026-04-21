"""
Unit tests for src/backtest/ — engine and metrics.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest.engine import backtest
from backtest.metrics import compute_metrics
from squeeze.indicators import compute_indicators
from squeeze.signals import generate_base_signal


def _make_ohlcv(n=400, seed=99):
    rng   = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high  = close + rng.uniform(0, 2, n)
    low   = close - rng.uniform(0, 2, n)
    df    = pd.DataFrame({
        "open":   close,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float),
    }, index=pd.date_range("2014-01-01", periods=n, freq="B"))
    df.index.name = "date"
    return df


def _prepared_df():
    df = compute_indicators(_make_ohlcv())
    return generate_base_signal(df)


def test_backtest_returns_required_keys():
    df     = _prepared_df()
    result = backtest(df, "base_signal", "TEST")
    for key in ["total_trades", "win_rate", "profit_factor",
                "max_drawdown_pct", "total_return_pct", "cagr_pct",
                "final_capital", "trades", "equity_curve"]:
        assert key in result, f"Missing key: {key}"


def test_backtest_equity_curve_length():
    df     = _prepared_df()
    result = backtest(df, "base_signal", "TEST")
    assert len(result["equity_curve"]) == len(df)


def test_backtest_win_rate_bounds():
    df     = _prepared_df()
    result = backtest(df, "base_signal", "TEST")
    assert 0.0 <= result["win_rate"] <= 100.0


def test_backtest_no_signals():
    df = _prepared_df()
    df["no_signal"] = False
    result = backtest(df, "no_signal", "TEST")
    assert result["total_trades"] == 0
