"""
Layer 3: XGBoost signal scorer.

Walk-forward approach:
  1. Label historical squeeze-release signals (did price hit TP before SL?)
  2. Train XGBoost on the first TRAIN_RATIO fraction of signals.
  3. Score remaining (out-of-sample) signals with predicted TP-hit probability.
  4. Gate entry on probability >= XGB_THRESHOLD.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config import SL_MULT, TP_MULT, XGB_THRESHOLD, TRAIN_RATIO

FEATURE_COLS = [
    "vol_z",          # volatility regime z-score
    "sqz_duration",   # bars squeeze was active before release
    "atr_ratio",      # ATR / close  (normalised vol)
    "vol_ratio",      # volume / vol_sma
    "adx",            # trend strength
    "kc_breakout",    # (close - upperKC) / upperKC
    "sma200_dist",    # (close - sma200) / sma200
    "bb_width",       # (upperBB - lowerBB) / basis
    "day_of_week",    # 0=Mon … 4=Fri
]


def label_signals(df: pd.DataFrame, signal_mask: pd.Series) -> pd.DataFrame:
    """
    Walk forward from each signal bar to determine whether TP or SL was hit first.

    Parameters
    ----------
    df          : indicator DataFrame (index = date)
    signal_mask : boolean Series aligned to df.index

    Returns
    -------
    DataFrame with FEATURE_COLS + 'label' column (1 = TP hit, 0 = SL/timeout)
    """
    rows_list = []
    df_arr    = df.reset_index()

    aligned   = signal_mask.reindex(df_arr["date"]).fillna(False).values
    sig_idxs  = df_arr[aligned].index.tolist()

    for idx in sig_idxs:
        row   = df_arr.iloc[idx]
        entry = row["close"]
        atr_v = row["atr"]
        sl    = entry - atr_v * SL_MULT
        tp    = entry + atr_v * SL_MULT * TP_MULT

        label = 0
        for j in range(idx + 1, min(idx + 60, len(df_arr))):
            future = df_arr.iloc[j]
            if future["low"] <= sl:
                label = 0
                break
            if future["high"] >= tp:
                label = 1
                break

        feat = {col: row[col] for col in FEATURE_COLS if col in df_arr.columns}
        feat["label"] = label
        feat["date"]  = row["date"]
        rows_list.append(feat)

    return pd.DataFrame(rows_list).set_index("date")


def train_xgb(labeled: pd.DataFrame):
    """
    Train XGBoost classifier on labeled signal data.

    Parameters
    ----------
    labeled : DataFrame returned by label_signals()

    Returns
    -------
    (XGBClassifier, StandardScaler)
    """
    X = labeled[FEATURE_COLS].values
    y = labeled["label"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_sc, y)
    return model, scaler


def score_signals(
    df: pd.DataFrame,
    base_signal: pd.Series,
    threshold: float = XGB_THRESHOLD,
    train_ratio: float = TRAIN_RATIO,
) -> pd.Series:
    """
    Walk-forward XGBoost scoring.  Returns a boolean Series (xgb_ok) aligned
    to df.index — True where predicted TP-probability >= threshold.

    Falls back to True (no filter) when there are insufficient signals to train.

    Parameters
    ----------
    df          : indicator DataFrame
    base_signal : boolean Series of Layer-0 signals
    threshold   : minimum predicted probability to allow entry
    train_ratio : fraction of signals used for training

    Returns
    -------
    pd.Series (bool), same index as df
    """
    xgb_ok = pd.Series(False, index=df.index)

    sig_indices = df.index[base_signal]
    n_sig       = len(sig_indices)

    if n_sig < 20:
        return pd.Series(True, index=df.index)

    train_cut = sig_indices[int(n_sig * train_ratio)]
    labeled   = label_signals(df.loc[:train_cut], base_signal.loc[:train_cut])

    if len(labeled) < 10 or labeled["label"].sum() < 3:
        return pd.Series(True, index=df.index)

    model, scaler = train_xgb(labeled)

    X_all = df.loc[sig_indices, FEATURE_COLS].values
    X_sc  = scaler.transform(X_all)
    probs = model.predict_proba(X_sc)[:, 1]

    xgb_pass = pd.Series(probs >= threshold, index=sig_indices)
    xgb_ok.loc[sig_indices] = xgb_pass.values
    return xgb_ok
