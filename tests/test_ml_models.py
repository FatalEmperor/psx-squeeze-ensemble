"""
Unit tests for src/ml/ — HMM regime and XGBoost scorer.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml.hmm_regime import fit_hmm, hmm_bull_states
from ml.xgboost_scorer import label_signals, train_xgb, score_signals, FEATURE_COLS
from squeeze.indicators import compute_indicators
from squeeze.signals import generate_base_signal


def _make_ohlcv(n=500, seed=7):
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
    }, index=pd.date_range("2013-01-01", periods=n, freq="B"))
    df.index.name = "date"
    return df


def _prepared_df():
    df = compute_indicators(_make_ohlcv())
    df = generate_base_signal(df)
    return df


def test_hmm_fit_predict():
    df     = _prepared_df()
    log_r  = df["log_ret"].values
    model  = fit_hmm(log_r)
    bulls  = hmm_bull_states(model, log_r)
    assert bulls.shape == log_r.shape
    assert bulls.dtype == bool


def test_hmm_two_states():
    log_r = np.random.randn(400)
    model = fit_hmm(log_r, n_states=2)
    assert model.n_components == 2


def test_score_signals_returns_bool_series():
    df     = _prepared_df()
    result = score_signals(df, df["base_signal"])
    assert result.dtype == bool
    assert len(result) == len(df)


def test_label_signals_has_label_col():
    df      = _prepared_df()
    labeled = label_signals(df, df["base_signal"])
    if not labeled.empty:
        assert "label" in labeled.columns
        assert set(labeled["label"].unique()).issubset({0, 1})
