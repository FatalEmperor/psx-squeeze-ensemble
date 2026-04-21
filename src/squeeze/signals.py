"""
Signal generation for the PSX Squeeze strategy.

Requires compute_indicators() (and optionally apply_rule_filters()) to have
been called on the DataFrame first.
"""

import pandas as pd


def generate_base_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 0: original TTM-squeeze breakout entry.

    Conditions:
      - Stable bull regime (log_ret > rolling mean AND vol_z < 0.5)
      - Squeeze release (BB expands outside KC)
      - Close above upperKC
      - Volume surge (volume > vol_sma)

    Adds column: base_signal (bool)
    """
    df = df.copy()
    df["base_signal"] = (
        df["stable_bull"]      &
        df["squeeze_release"]  &
        (df["close"] > df["upperKC"]) &
        df["inst_flow"]
    )
    return df


def generate_rule_filtered_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Layers 0-2: base signal gated by rule filters.

    Requires filter_sma, filter_sqz, filter_adx columns (from apply_rule_filters).
    Adds column: sig_rules (bool)
    """
    df = df.copy()
    df["sig_rules"] = (
        df["base_signal"] &
        df["filter_sma"]  &
        df["filter_sqz"]  &
        df["filter_adx"]
    )
    return df


def generate_ensemble_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Layers 0-4: full ensemble signal.

    Requires: base_signal, filter_sma, filter_sqz, filter_adx,
              hmm_bull (from hmm_regime), xgb_ok (from xgboost_scorer).
    Adds column: sig_ensemble (bool)
    """
    df = df.copy()
    df["sig_ensemble"] = (
        df["base_signal"] &
        df["filter_sma"]  &
        df["filter_sqz"]  &
        df["filter_adx"]  &
        df["hmm_bull"]    &
        df["xgb_ok"]
    )
    return df
