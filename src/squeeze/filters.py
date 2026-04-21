"""
Rule-based entry filters for the PSX Squeeze strategy (Layers 1 & 2).

Requires compute_indicators() to have been run on the DataFrame first.
"""

import pandas as pd

from config import SMA_TREND, MIN_SQUEEZE_BARS, ADX_PERIOD


def apply_rule_filters(
    df: pd.DataFrame,
    min_sqz_bars: int = MIN_SQUEEZE_BARS,
    adx_threshold: float = 25.0,
) -> pd.DataFrame:
    """
    Add three boolean filter columns to df:

    filter_sma  — Layer 1a: close > SMA-200 (bull-trend bias)
    filter_sqz  — Layer 1b: squeeze was active for >= min_sqz_bars before release
    filter_adx  — Layer 2 : ADX > adx_threshold (real directional momentum)

    Returns df with the new columns appended (in-place copy).
    """
    df = df.copy()
    df["filter_sma"] = df["close"] > df["sma200"]
    df["filter_sqz"] = df["sqz_duration"].shift(1).fillna(0) >= min_sqz_bars
    df["filter_adx"] = df["adx"] > adx_threshold
    return df
