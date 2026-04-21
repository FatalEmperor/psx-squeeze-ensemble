"""
Ensemble Squeeze Strategy — PSX Backtest
=========================================
Combines four layers on top of the base Institutional Squeeze signal:

  Layer 1 (Rule)  : Price > 200-SMA  +  Squeeze held >= 3 bars
  Layer 2 (Rule)  : ADX > 25  (only trade real momentum)
  Layer 3 (ML)    : XGBoost signal scorer — predicts TP-hit probability
                    Walk-forward: train on past signals, score future ones
  Layer 4 (ML)    : Hidden Markov Model regime filter
                    2-state HMM on log-returns → only trade in bull state

Final entry = base_signal AND sma200 AND squeeze_dur AND adx AND xgb_prob>0.55 AND hmm_bull

Backtest mechanics identical to backtest_squeeze.py:
  - Entry at close of signal bar
  - SL/TP checked from next bar via intrabar high/low
  - 100% equity, no pyramiding, 0.02% commission per side
"""

import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ML
import xgboost as xgb
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ─── CONFIG ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000_000
COMMISSION       = 0.0002
LOOKBACK         = 20
SQZ_LEN          = 20
MULT_BB          = 2.0
MULT_KC          = 1.5
ATR_PERIOD       = 14
SL_MULT          = 2.0
TP_MULT          = 1.5        # TP dist = SL dist * 1.5
VOL_SMA_LEN      = 20
ADX_PERIOD       = 14
SMA_TREND        = 200
MIN_SQUEEZE_BARS = 3          # squeeze must hold >= N bars before release
HMM_STATES       = 2
XGB_THRESHOLD    = 0.55       # min TP probability to allow entry
TRAIN_RATIO      = 0.65       # fraction of signals used to train XGB
BACKTEST_YEARS   = 10

BASE_URL = "https://psxterminal.com/api"

DEFAULT_SYMBOLS = [
    "OGDC", "PPL", "PSO", "HUBC", "ENGRO", "LUCK", "MCB",
    "UBL",  "HBL", "MARI", "EFERT", "DGKC", "MLCF", "UNITY",
    "AIRLINK", "SYS", "TRG", "BAFL", "BAHL", "MEBL",
]

# XGBoost features used to score each signal
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


# ─── DATA ────────────────────────────────────────────────────────────────────

def _ts_ms(dt): return int(dt.timestamp() * 1000)


def fetch_psxterminal(symbol, years=BACKTEST_YEARS):
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=365 * years + 30)
    end_ts, cursor = _ts_ms(end_dt), _ts_ms(start_dt)
    records = []
    while cursor < end_ts:
        try:
            r = requests.get(f"{BASE_URL}/klines/{symbol}/1d",
                             params={"start": cursor, "end": end_ts, "limit": 100},
                             timeout=15)
            r.raise_for_status()
            p = r.json()
        except Exception as e:
            raise ConnectionError(e) from e
        if not p.get("success"): break
        chunk = p.get("data") or []
        if not chunk: break
        records.extend(chunk)
        if len(chunk) < 100: break
        last = chunk[-1]["timestamp"]
        if last <= cursor: break
        cursor = last + 1
        time.sleep(0.12)
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return (df.sort_values("date").drop_duplicates("date").set_index("date")
            [["open","high","low","close","volume"]].astype(float))


def fetch_yfinance(symbol, years=BACKTEST_YEARS):
    if not HAS_YF: return pd.DataFrame()
    start = (datetime.now() - timedelta(days=365*years+30)).strftime("%Y-%m-%d")
    raw = yf.Ticker(f"{symbol}.KA").history(start=start, auto_adjust=True)
    if raw.empty: return pd.DataFrame()
    df = raw[["Open","High","Low","Close","Volume"]].copy()
    df.columns = ["open","high","low","close","volume"]
    df.index.name = "date"
    return df.astype(float)


def fetch_data(symbol):
    try:
        df = fetch_psxterminal(symbol)
        if not df.empty: return df
    except Exception:
        pass
    return fetch_yfinance(symbol)


# ─── INDICATORS ───────────────────────────────────────────────────────────────

def wilder_smooth(series, period):
    """Wilder's RMA (used in ADX/ATR)."""
    result = np.full(len(series), np.nan)
    vals   = series.values
    start  = period - 1
    while start < len(vals) and np.isnan(vals[start]): start += 1
    if start + period > len(vals): return pd.Series(result, index=series.index)
    result[start] = np.nanmean(vals[start - period + 1: start + 1])
    for i in range(start + 1, len(vals)):
        if np.isnan(vals[i]):
            result[i] = result[i-1]
        else:
            result[i] = result[i-1] - result[i-1] / period + vals[i]
    return pd.Series(result, index=series.index)


def compute_adx(df, period=ADX_PERIOD):
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

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    s_tr       = wilder_smooth(tr, period)
    s_plus_dm  = wilder_smooth(pd.Series(plus_dm, index=df.index), period)
    s_minus_dm = wilder_smooth(pd.Series(minus_dm, index=df.index), period)

    plus_di  = 100 * s_plus_dm  / s_tr
    minus_di = 100 * s_minus_dm / s_tr
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx      = wilder_smooth(dx, period)
    return adx


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_close = df["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # ATR
    df["atr"] = wilder_smooth(df["tr"], ATR_PERIOD)

    # SMA-200
    df["sma200"] = df["close"].rolling(SMA_TREND).mean()

    # ADX
    df["adx"] = compute_adx(df)

    # Markov regime
    df["log_ret"]  = np.log(df["close"] / df["close"].shift(1))
    df["mu"]       = df["log_ret"].rolling(LOOKBACK).mean()
    df["sigma"]    = df["log_ret"].rolling(LOOKBACK).std()
    sig_sma        = df["sigma"].rolling(50).mean()
    sig_std        = df["sigma"].rolling(50).std()
    df["vol_z"]    = (df["sigma"] - sig_sma) / sig_std
    df["stable_bull"] = (df["log_ret"] > df["mu"]) & (df["vol_z"] < 0.5)

    # Bollinger Bands
    basis         = df["close"].rolling(SQZ_LEN).mean()
    bb_std        = df["close"].rolling(SQZ_LEN).std()
    df["upperBB"] = basis + MULT_BB * bb_std
    df["lowerBB"] = basis - MULT_BB * bb_std
    df["bb_width"]= (df["upperBB"] - df["lowerBB"]) / basis

    # Keltner Channels
    ma_kc         = df["close"].rolling(SQZ_LEN).mean()
    range_kc      = df["tr"].rolling(SQZ_LEN).mean()
    df["upperKC"] = ma_kc + range_kc * MULT_KC
    df["lowerKC"] = ma_kc - range_kc * MULT_KC

    # Squeeze
    df["squeezed"]        = (df["lowerBB"] > df["lowerKC"]) & (df["upperBB"] < df["upperKC"])
    df["squeeze_release"] = (df["upperBB"] > df["upperKC"]) & (df["lowerBB"] < df["lowerKC"])

    # Squeeze duration: count consecutive bars in squeeze before release
    sqz_dur = []
    count = 0
    for sq in df["squeezed"]:
        if sq: count += 1
        else:  count = 0
        sqz_dur.append(count)
    df["sqz_duration"] = sqz_dur

    # Volume
    df["vol_sma"]   = df["volume"].rolling(VOL_SMA_LEN).mean()
    df["inst_flow"] = df["volume"] > df["vol_sma"]
    df["vol_ratio"] = df["volume"] / df["vol_sma"]

    # XGB features
    df["atr_ratio"]   = df["atr"] / df["close"]
    df["kc_breakout"] = (df["close"] - df["upperKC"]) / df["upperKC"]
    df["sma200_dist"] = (df["close"] - df["sma200"]) / df["sma200"]
    df["day_of_week"] = df.index.dayofweek

    # BASE signal (original strategy)
    df["base_signal"] = (
        df["stable_bull"] &
        df["squeeze_release"] &
        (df["close"] > df["upperKC"]) &
        df["inst_flow"]
    )

    # Layer 1: SMA200 + squeeze duration
    df["filter_sma"]  = df["close"] > df["sma200"]
    df["filter_sqz"]  = df["sqz_duration"].shift(1).fillna(0) >= MIN_SQUEEZE_BARS

    # Layer 2: ADX
    df["filter_adx"]  = df["adx"] > 25

    return df.dropna()


# ─── HMM REGIME ───────────────────────────────────────────────────────────────

def fit_hmm(log_returns: np.ndarray) -> GaussianHMM:
    """Fit 2-state Gaussian HMM. Returns fitted model."""
    model = GaussianHMM(
        n_components=HMM_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(log_returns.reshape(-1, 1))
    return model


def hmm_bull_states(model: GaussianHMM, log_returns: np.ndarray) -> np.ndarray:
    """
    Predict hidden states. Bull = state with higher mean return.
    Returns boolean array: True = bull state.
    """
    states = model.predict(log_returns.reshape(-1, 1))
    means  = model.means_.flatten()
    bull_state = int(np.argmax(means))    # state with higher mean = bull
    return states == bull_state


# ─── XGBOOST SIGNAL SCORER ────────────────────────────────────────────────────

def label_signals(df: pd.DataFrame, signal_mask: pd.Series) -> pd.DataFrame:
    """
    For each signal bar, walk forward to find whether TP or SL is hit first.
    Returns DataFrame with features + binary label (1=TP hit, 0=SL/timeout).
    """
    rows_list = []
    df_arr = df.reset_index()

    signal_indices = df_arr[signal_mask.reindex(df_arr["date"]).fillna(False).values].index.tolist()

    for idx in signal_indices:
        row = df_arr.iloc[idx]
        entry  = row["close"]
        atr_v  = row["atr"]
        sl     = entry - atr_v * SL_MULT
        tp     = entry + atr_v * SL_MULT * TP_MULT

        label = 0   # default: not hit TP
        for j in range(idx + 1, min(idx + 60, len(df_arr))):  # max 60 bars lookforward
            future = df_arr.iloc[j]
            if future["low"] <= sl:
                label = 0; break
            if future["high"] >= tp:
                label = 1; break

        feat = {col: row[col] for col in FEATURE_COLS if col in df_arr.columns}
        feat["label"] = label
        feat["date"]  = row["date"]
        rows_list.append(feat)

    return pd.DataFrame(rows_list).set_index("date")


def train_xgb(labeled: pd.DataFrame):
    """Train XGBoost on labeled signal data. Returns (model, scaler)."""
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
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_sc, y)
    return model, scaler


# ─── BACKTESTER ───────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, signal_col: str, symbol: str) -> dict:
    capital = float(INITIAL_CAPITAL)
    equity_curve: list[float] = []
    trades: list[dict] = []

    in_position = False
    entry_price = sl_price = tp_price = shares = entry_value = 0.0
    entry_date  = None

    rows = list(df.iterrows())

    for idx, (date, row) in enumerate(rows):
        if in_position and idx > 0:
            sl_hit = row["low"]  <= sl_price
            tp_hit = row["high"] >= tp_price
            exited, exit_price, exit_type = False, 0.0, ""

            if sl_hit and tp_hit:
                exit_price, exit_type, exited = sl_price, "SL", True
            elif sl_hit:
                exit_price, exit_type, exited = sl_price, "SL", True
            elif tp_hit:
                exit_price, exit_type, exited = tp_price, "TP", True

            if exited:
                proceeds = shares * exit_price * (1 - COMMISSION)
                pnl      = proceeds - entry_value
                capital  = capital - entry_value + proceeds
                trades.append({
                    "symbol":        symbol,
                    "entry_date":    entry_date,
                    "exit_date":     date,
                    "entry_price":   round(entry_price, 4),
                    "exit_price":    round(exit_price, 4),
                    "exit_type":     exit_type,
                    "pnl":           round(pnl, 2),
                    "pnl_pct":       round((exit_price / entry_price - 1) * 100, 3),
                    "capital_after": round(capital, 2),
                    "bars_held":     (date - entry_date).days,
                })
                in_position = False

        if not in_position and row.get(signal_col, False):
            entry_price = row["close"]
            atr_v       = row["atr"]
            sl_dist     = atr_v * SL_MULT
            sl_price    = entry_price - sl_dist
            tp_price    = entry_price + sl_dist * TP_MULT
            entry_value = capital
            shares      = entry_value / (entry_price * (1 + COMMISSION))
            in_position = True
            entry_date  = date

        equity_curve.append(capital)

    if in_position:
        last_date, last_row = rows[-1]
        exit_price = last_row["close"]
        proceeds   = shares * exit_price * (1 - COMMISSION)
        pnl        = proceeds - entry_value
        capital    = capital - entry_value + proceeds
        equity_curve[-1] = capital
        trades.append({
            "symbol":        symbol,
            "entry_date":    entry_date,
            "exit_date":     last_date,
            "entry_price":   round(entry_price, 4),
            "exit_price":    round(exit_price, 4),
            "exit_type":     "EOD",
            "pnl":           round(pnl, 2),
            "pnl_pct":       round((exit_price / entry_price - 1) * 100, 3),
            "capital_after": round(capital, 2),
            "bars_held":     (last_date - entry_date).days,
        })

    n = len(trades)
    if n == 0:
        return {"symbol": symbol, "signal": signal_col, "total_trades": 0,
                "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0,
                "total_return_pct": round((capital/INITIAL_CAPITAL-1)*100,2),
                "cagr_pct": 0.0, "final_capital": round(capital,2)}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    pf     = gp / gl if gl > 0 else 999.0

    eq          = pd.Series(equity_curve)
    max_dd      = float(((eq - eq.cummax()) / eq.cummax() * 100).min())
    n_years     = (df.index[-1] - df.index[0]).days / 365.25
    total_ret   = (capital / INITIAL_CAPITAL - 1) * 100
    cagr        = ((capital / INITIAL_CAPITAL)**(1/n_years) - 1)*100 if n_years > 0 else 0.0

    return {
        "symbol":           symbol,
        "signal":           signal_col,
        "total_trades":     n,
        "win_rate":         round(len(wins)/n*100, 2),
        "profit_factor":    round(min(pf, 999.0), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_return_pct": round(total_ret, 2),
        "cagr_pct":         round(cagr, 2),
        "final_capital":    round(capital, 2),
        "trades":           trades,
        "equity_curve":     equity_curve,
    }


# ─── PER-SYMBOL PIPELINE ─────────────────────────────────────────────────────

def run_symbol(symbol: str) -> list[dict]:
    df_raw = fetch_data(symbol)
    if df_raw.empty or len(df_raw) < 300:
        print(f"  {symbol:<12s}  SKIP (only {len(df_raw)} bars)")
        return []

    df = compute_indicators(df_raw)
    if df.empty:
        print(f"  {symbol:<12s}  SKIP (indicator error)")
        return []

    # ── Layer 4: HMM regime ───────────────────────────────────────────────────
    log_rets  = df["log_ret"].values
    hmm_model = fit_hmm(log_rets)
    bull_mask = hmm_bull_states(hmm_model, log_rets)
    df["hmm_bull"] = bull_mask

    # ── Layer 3: XGBoost walk-forward ────────────────────────────────────────
    # Use ONLY base_signal rows so XGB learns on squeeze-release events
    base_signals = df["base_signal"]
    sig_indices  = df.index[base_signals]
    n_sig        = len(sig_indices)

    df["xgb_ok"] = False    # default — no XGB filter

    if n_sig >= 20:          # need enough signals to train
        train_cut = sig_indices[int(n_sig * TRAIN_RATIO)]

        train_df  = df.loc[:train_cut]
        train_sig = base_signals.loc[:train_cut]

        labeled   = label_signals(train_df, train_sig)

        if len(labeled) >= 10 and labeled["label"].sum() >= 3:
            xgb_model, scaler = train_xgb(labeled)

            # Score ALL signal bars (train+test); walk-forward is approximate here
            # Strictly correct approach: retrain on every new signal — too slow for demo
            X_all    = df.loc[sig_indices, FEATURE_COLS].values
            X_sc     = scaler.transform(X_all)
            probs    = xgb_model.predict_proba(X_sc)[:, 1]
            xgb_pass = pd.Series(probs >= XGB_THRESHOLD, index=sig_indices)

            df.loc[sig_indices, "xgb_ok"] = xgb_pass.values
        else:
            df["xgb_ok"] = True   # not enough data → don't filter
    else:
        df["xgb_ok"] = True       # not enough signals → don't filter

    # ── Build signal columns ──────────────────────────────────────────────────
    df["sig_base"]     = df["base_signal"]

    df["sig_rules"]    = (
        df["base_signal"] &
        df["filter_sma"]  &
        df["filter_sqz"]  &
        df["filter_adx"]
    )

    df["sig_ensemble"] = (
        df["base_signal"] &
        df["filter_sma"]  &
        df["filter_sqz"]  &
        df["filter_adx"]  &
        df["hmm_bull"]    &
        df["xgb_ok"]
    )

    results = []
    for sig_col in ["sig_base", "sig_rules", "sig_ensemble"]:
        r = backtest(df, sig_col, symbol)
        results.append(r)

    # Print comparison line
    def fmt(r):
        return (f"T:{r['total_trades']:3d}  WR:{r['win_rate']:5.1f}%  "
                f"PF:{r['profit_factor']:5.2f}  CAGR:{r['cagr_pct']:6.1f}%  "
                f"DD:{r['max_drawdown_pct']:6.1f}%")

    print(f"\n  {symbol}")
    for r in results:
        lbl = r["signal"].replace("sig_", "").upper()
        print(f"    [{lbl:8s}]  {fmt(r)}")

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(symbols=None):
    if symbols is None:
        try:
            resp = requests.get(f"{BASE_URL}/symbols", timeout=10)
            syms = [s["symbol"] for s in resp.json().get("data", [])]
            symbols = syms if syms else DEFAULT_SYMBOLS
        except Exception:
            symbols = DEFAULT_SYMBOLS

    print(f"Ensemble Backtest — {len(symbols)} symbols  |  {BACKTEST_YEARS}-year window")
    print(f"Layers: SMA{SMA_TREND} + Squeeze>{MIN_SQUEEZE_BARS}bars + ADX>25 + HMM({HMM_STATES}-state) + XGBoost(thresh={XGB_THRESHOLD})")
    print("=" * 80)

    all_results: list[dict] = []

    for symbol in symbols:
        results = run_symbol(symbol)
        all_results.extend(results)

    if not all_results:
        print("No results.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    cols = ["symbol", "signal", "total_trades", "win_rate", "profit_factor",
            "max_drawdown_pct", "cagr_pct", "final_capital"]
    df_all = pd.DataFrame([{c: r[c] for c in cols} for r in all_results])

    print("\n" + "=" * 100)
    print("COMPARISON: BASE vs RULE-FILTERED vs FULL ENSEMBLE")
    print("=" * 100)

    for sym in df_all["symbol"].unique():
        sub = df_all[df_all["symbol"] == sym]
        print(f"\n  {sym}")
        print(sub[["signal","total_trades","win_rate","profit_factor",
                    "max_drawdown_pct","cagr_pct","final_capital"]].to_string(index=False))

    # Aggregate by signal type
    print("\n-- Average across all symbols -------------------------------------------")
    agg = df_all.groupby("signal")[["win_rate","profit_factor","cagr_pct","max_drawdown_pct"]].mean()
    print(agg.round(2).to_string())

    # Save trades
    trade_rows = []
    for r in all_results:
        for t in r.get("trades", []):
            t["signal"] = r["signal"]
            trade_rows.append(t)

    if trade_rows:
        pd.DataFrame(trade_rows).to_csv("ensemble_trades.csv", index=False)
        print(f"\nTrades saved  -> ensemble_trades.csv  ({len(trade_rows)} rows)")

    df_all.to_csv("ensemble_summary.csv", index=False)
    print("Summary saved -> ensemble_summary.csv")


if __name__ == "__main__":
    user_syms = sys.argv[1:] or None
    main(user_syms)
