"""
Technical indicator computation for the PSX Squeeze strategy.

All indicators are appended to a copy of the input DataFrame and returned.
Uses Wilder's RMA for ATR and ADX (matches Pine Script behaviour).
"""

import numpy as np
import pandas as pd

from config import (
    LOOKBACK, SQZ_LEN, MULT_BB, MULT_KC,
    ATR_PERIOD, ADX_PERIOD, VOL_SMA_LEN, SMA_TREND,
)


# ── Low-level helpers ──────────────────────────────────────────────────────────

def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA — used for ATR and ADX (matches Pine Script ta.rma)."""
    result = np.full(len(series), np.nan)
    vals   = series.values
    start  = period - 1
    while start < len(vals) and np.isnan(vals[start]):
        start += 1
    if start + period > len(vals):
        return pd.Series(result, index=series.index)
    result[start] = np.nanmean(vals[start - period + 1: start + 1])
    for i in range(start + 1, len(vals)):
        if np.isnan(vals[i]):
            result[i] = result[i - 1]
        else:
            result[i] = result[i - 1] - result[i - 1] / period + vals[i]
    return pd.Series(result, index=series.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    """Standard True Range."""
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """Wilder-smoothed ADX (matches Pine Script ta.adx)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    s_tr       = wilder_smooth(tr, period)
    s_plus_dm  = wilder_smooth(pd.Series(plus_dm,  index=df.index), period)
    s_minus_dm = wilder_smooth(pd.Series(minus_dm, index=df.index), period)

    plus_di  = 100 * s_plus_dm  / s_tr
    minus_di = 100 * s_minus_dm / s_tr
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return wilder_smooth(dx, period)


# ── Main indicator pipeline ────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all technical indicators to df and return a cleaned copy.
    Drops rows with any NaN (warm-up period).

    Added columns
    -------------
    tr, atr, sma200, adx,
    log_ret, mu, sigma, vol_z, stable_bull,
    upperBB, lowerBB, bb_width,
    upperKC, lowerKC,
    squeezed, squeeze_release, sqz_duration,
    vol_sma, inst_flow, vol_ratio,
    atr_ratio, kc_breakout, sma200_dist, day_of_week
    """
    df = df.copy()

    # ── True Range & ATR (Wilder RMA) ─────────────────────────────────────────
    df["tr"]  = true_range(df)
    df["atr"] = wilder_smooth(df["tr"], ATR_PERIOD)

    # ── Trend filter ──────────────────────────────────────────────────────────
    df["sma200"] = df["close"].rolling(SMA_TREND).mean()

    # ── ADX ───────────────────────────────────────────────────────────────────
    df["adx"] = compute_adx(df)

    # ── Volatility regime (Markov-style) ─────────────────────────────────────
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["mu"]      = df["log_ret"].rolling(LOOKBACK).mean()
    df["sigma"]   = df["log_ret"].rolling(LOOKBACK).std()
    sig_sma       = df["sigma"].rolling(50).mean()
    sig_std       = df["sigma"].rolling(50).std()
    df["vol_z"]        = (df["sigma"] - sig_sma) / sig_std
    df["stable_bull"]  = (df["log_ret"] > df["mu"]) & (df["vol_z"] < 0.5)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    basis         = df["close"].rolling(SQZ_LEN).mean()
    bb_std        = df["close"].rolling(SQZ_LEN).std()
    df["upperBB"] = basis + MULT_BB * bb_std
    df["lowerBB"] = basis - MULT_BB * bb_std
    df["bb_width"]= (df["upperBB"] - df["lowerBB"]) / basis

    # ── Keltner Channels (TR-based, matches Pine Script) ──────────────────────
    ma_kc         = df["close"].rolling(SQZ_LEN).mean()
    range_kc      = df["tr"].rolling(SQZ_LEN).mean()
    df["upperKC"] = ma_kc + range_kc * MULT_KC
    df["lowerKC"] = ma_kc - range_kc * MULT_KC

    # ── Squeeze detection ─────────────────────────────────────────────────────
    df["squeezed"]        = (df["lowerBB"] > df["lowerKC"]) & (df["upperBB"] < df["upperKC"])
    df["squeeze_release"] = (df["upperBB"] > df["upperKC"]) & (df["lowerBB"] < df["lowerKC"])

    # Count consecutive bars in squeeze before each release
    sqz_dur, count = [], 0
    for sq in df["squeezed"]:
        count = count + 1 if sq else 0
        sqz_dur.append(count)
    df["sqz_duration"] = sqz_dur

    # ── Volume ────────────────────────────────────────────────────────────────
    df["vol_sma"]   = df["volume"].rolling(VOL_SMA_LEN).mean()
    df["inst_flow"] = df["volume"] > df["vol_sma"]
    df["vol_ratio"] = df["volume"] / df["vol_sma"]

    # ── XGBoost feature columns ───────────────────────────────────────────────
    df["atr_ratio"]   = df["atr"] / df["close"]
    df["kc_breakout"] = (df["close"] - df["upperKC"]) / df["upperKC"]
    df["sma200_dist"] = (df["close"] - df["sma200"]) / df["sma200"]
    df["day_of_week"] = df.index.dayofweek

    return df.dropna()
