"""
Unit tests for src/squeeze/ — indicators, filters, signals.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from squeeze.indicators import wilder_smooth, true_range, compute_adx, compute_indicators
from squeeze.filters import apply_rule_filters
from squeeze.signals import generate_base_signal


def _make_ohlcv(n=300, seed=42):
    """Generate synthetic OHLCV DataFrame."""
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


def test_wilder_smooth_length():
    s      = pd.Series(np.random.rand(100))
    result = wilder_smooth(s, 14)
    assert len(result) == 100


def test_true_range_non_negative():
    df = _make_ohlcv()
    tr = true_range(df)
    assert (tr.dropna() >= 0).all()


def test_compute_indicators_columns():
    df  = _make_ohlcv()
    out = compute_indicators(df)
    required = ["atr", "adx", "sma200", "vol_z", "upperBB", "lowerBB",
                "upperKC", "lowerKC", "squeezed", "squeeze_release",
                "sqz_duration", "base_signal" if "base_signal" in out.columns else "vol_sma"]
    for col in ["atr", "adx", "sma200", "squeezed", "squeeze_release"]:
        assert col in out.columns, f"Missing column: {col}"


def test_apply_rule_filters_columns():
    df  = _make_ohlcv()
    df  = compute_indicators(df)
    df  = apply_rule_filters(df)
    for col in ["filter_sma", "filter_sqz", "filter_adx"]:
        assert col in df.columns


def test_generate_base_signal_boolean():
    df  = _make_ohlcv()
    df  = compute_indicators(df)
    df  = generate_base_signal(df)
    assert df["base_signal"].dtype == bool
