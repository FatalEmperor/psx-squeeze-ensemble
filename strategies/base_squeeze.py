"""
PSX Institutional Squeeze — base strategy runner (Layer 0 only).

Translated from InstitutionalSqueeze_Improved.pine by Haseeb Zahid.

Usage
-----
    # Full symbol list
    python strategies/base_squeeze.py

    # Specific symbols
    python strategies/base_squeeze.py OGDC PPL PSO HUBC
"""

import sys
import os
import warnings
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from config import INITIAL_CAPITAL, COMMISSION, BACKTEST_YEARS
from data.fetcher import fetch_data, get_symbols
from squeeze.indicators import compute_indicators
from squeeze.signals import generate_base_signal
from backtest.engine import backtest
from backtest.metrics import print_summary

_ROOT      = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_ROOT, "data", "backtest_results")


def main(symbols=None):
    if symbols is None:
        symbols = get_symbols()
        print(f"Fetching {len(symbols)} PSX symbols...")

    all_results = []
    all_trades  = []

    for symbol in symbols:
        sys.stdout.write(f"\r  {symbol:<12s}  fetching...")
        sys.stdout.flush()

        df_raw = fetch_data(symbol)
        if df_raw.empty or len(df_raw) < 120:
            sys.stdout.write(f"\r  {symbol:<12s}  SKIP ({len(df_raw)} bars)\n")
            continue

        df = compute_indicators(df_raw)
        if df.empty:
            sys.stdout.write(f"\r  {symbol:<12s}  SKIP (indicator error)\n")
            continue

        df = generate_base_signal(df)
        result = backtest(df, "base_signal", symbol)
        all_results.append(result)
        all_trades.extend(result.get("trades", []))

        sys.stdout.write(
            f"\r  {symbol:<12s}  "
            f"Trades: {result['total_trades']:3d}  "
            f"WR: {result['win_rate']:5.1f}%  "
            f"PF: {result['profit_factor']:5.2f}  "
            f"CAGR: {result['cagr_pct']:6.1f}%  "
            f"MaxDD: {result['max_drawdown_pct']:6.1f}%\n"
        )

    if not all_results:
        print("No results — check network / API availability.")
        return [], []

    summary_df = print_summary(all_results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if all_trades:
        trades_path = os.path.join(OUTPUT_DIR, "backtest_trades.csv")
        pd.DataFrame(all_trades).to_csv(trades_path, index=False)
        print(f"\nTrades saved  -> {trades_path}  ({len(all_trades)} rows)")

    summary_path = os.path.join(OUTPUT_DIR, "backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved -> {summary_path}")

    return all_results, all_trades


if __name__ == "__main__":
    results, trades = main(sys.argv[1:] or None)
