"""
PSX Squeeze Ensemble — main strategy runner (Layers 0-4).

Usage
-----
    # Full symbol list (fetched from psxterminal API)
    python strategies/ensemble_squeeze.py

    # Specific symbols
    python strategies/ensemble_squeeze.py OGDC PPL PSO
"""

import sys
import os
import warnings
import pandas as pd

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from config import (
    INITIAL_CAPITAL, BACKTEST_YEARS, SMA_TREND,
    MIN_SQUEEZE_BARS, HMM_STATES, XGB_THRESHOLD,
)
from data.fetcher import fetch_data, get_symbols
from squeeze.indicators import compute_indicators
from squeeze.filters import apply_rule_filters
from squeeze.signals import (
    generate_base_signal,
    generate_rule_filtered_signal,
    generate_ensemble_signal,
)
from ml.hmm_regime import fit_hmm, hmm_bull_states
from ml.xgboost_scorer import score_signals
from backtest.engine import backtest
from backtest.metrics import print_summary

# Output directory (relative to project root)
_ROOT      = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_ROOT, "data", "backtest_results")


def run_symbol(symbol: str) -> list:
    """Run full ensemble pipeline for a single symbol. Returns list of result dicts."""
    df_raw = fetch_data(symbol)
    if df_raw.empty or len(df_raw) < 300:
        print(f"  {symbol:<12s}  SKIP ({len(df_raw)} bars)")
        return []

    df = compute_indicators(df_raw)
    if df.empty:
        print(f"  {symbol:<12s}  SKIP (indicator error)")
        return []

    # Layer 1 & 2: rule filters
    df = apply_rule_filters(df)

    # Layer 0: base signal
    df = generate_base_signal(df)

    # Layer 3: XGBoost
    df["xgb_ok"] = score_signals(df, df["base_signal"])

    # Layer 4: HMM regime
    hmm_model    = fit_hmm(df["log_ret"].values)
    df["hmm_bull"] = hmm_bull_states(hmm_model, df["log_ret"].values)

    # Build all signal variants
    df = generate_rule_filtered_signal(df)
    df = generate_ensemble_signal(df)

    results = []
    for sig_col in ["base_signal", "sig_rules", "sig_ensemble"]:
        r = backtest(df, sig_col, symbol)
        results.append(r)

    def _fmt(r):
        return (f"T:{r['total_trades']:3d}  WR:{r['win_rate']:5.1f}%  "
                f"PF:{r['profit_factor']:5.2f}  CAGR:{r['cagr_pct']:6.1f}%  "
                f"DD:{r['max_drawdown_pct']:6.1f}%")

    print(f"\n  {symbol}")
    for r in results:
        lbl = r["signal"].replace("base_signal", "BASE").replace("sig_", "").upper()
        print(f"    [{lbl:8s}]  {_fmt(r)}")

    return results


def main(symbols=None):
    if symbols is None:
        symbols = get_symbols()

    print(f"Ensemble Backtest — {len(symbols)} symbols  |  {BACKTEST_YEARS}-year window")
    print(f"Layers: SMA{SMA_TREND} + Squeeze>{MIN_SQUEEZE_BARS}bars + ADX>25 "
          f"+ HMM({HMM_STATES}-state) + XGBoost(thresh={XGB_THRESHOLD})")
    print("=" * 80)

    all_results = []
    for symbol in symbols:
        all_results.extend(run_symbol(symbol))

    if not all_results:
        print("No results.")
        return

    # ── Summary ────────────────────────────────────────────────────────────────
    df_all = print_summary(all_results)

    print("\n-- Average across all symbols -------------------------------------------")
    agg = df_all.groupby("signal")[
        ["win_rate", "profit_factor", "cagr_pct", "max_drawdown_pct"]
    ].mean()
    print(agg.round(2).to_string())

    # ── Save outputs ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trade_rows = []
    for r in all_results:
        for t in r.get("trades", []):
            t["signal"] = r["signal"]
            trade_rows.append(t)

    if trade_rows:
        trades_path = os.path.join(OUTPUT_DIR, "ensemble_trades.csv")
        pd.DataFrame(trade_rows).to_csv(trades_path, index=False)
        print(f"\nTrades saved  -> {trades_path}  ({len(trade_rows)} rows)")

    summary_path = os.path.join(OUTPUT_DIR, "ensemble_summary.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
