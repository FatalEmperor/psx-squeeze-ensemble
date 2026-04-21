"""
Bar-by-bar backtest engine.

Pine Script behaviour replicated:
  - Entry executes at the CLOSE of the signal bar.
  - SL / TP checked from the NEXT bar onwards using intrabar high/low.
  - Both SL and TP triggered same bar → SL wins (conservative).
  - 100% of equity per trade, no pyramiding.
  - Commission applied on both entry and exit.
"""

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL, COMMISSION, SL_MULT, TP_MULT
from backtest.metrics import compute_metrics


def backtest(
    df: pd.DataFrame,
    signal_col: str,
    symbol: str,
    initial_capital: float = INITIAL_CAPITAL,
    commission: float = COMMISSION,
    sl_mult: float = SL_MULT,
    tp_mult: float = TP_MULT,
) -> dict:
    """
    Simulate trades bar-by-bar.

    Parameters
    ----------
    df            : indicator DataFrame with 'atr' column and signal_col
    signal_col    : boolean column name for entry signals
    symbol        : ticker label (included in trade records)
    initial_capital, commission, sl_mult, tp_mult : override config defaults

    Returns
    -------
    dict with keys: symbol, signal, total_trades, win_rate, profit_factor,
                    max_drawdown_pct, total_return_pct, cagr_pct,
                    final_capital, trades, equity_curve
    """
    capital      = float(initial_capital)
    equity_curve = []
    trades       = []

    in_position  = False
    entry_price  = sl_price = tp_price = shares = entry_value = 0.0
    entry_date   = None

    rows = list(df.iterrows())

    for idx, (date, row) in enumerate(rows):

        # ── Manage open position ──────────────────────────────────────────────
        if in_position and idx > 0:
            sl_hit = row["low"]  <= sl_price
            tp_hit = row["high"] >= tp_price
            exited, exit_price, exit_type = False, 0.0, ""

            if sl_hit:
                exit_price, exit_type, exited = sl_price, "SL", True
            elif tp_hit:
                exit_price, exit_type, exited = tp_price, "TP", True

            if exited:
                proceeds = shares * exit_price * (1.0 - commission)
                pnl      = proceeds - entry_value
                capital  = capital - entry_value + proceeds
                trades.append({
                    "symbol":        symbol,
                    "entry_date":    entry_date,
                    "exit_date":     date,
                    "entry_price":   round(entry_price, 4),
                    "exit_price":    round(exit_price, 4),
                    "sl_price":      round(sl_price, 4),
                    "tp_price":      round(tp_price, 4),
                    "exit_type":     exit_type,
                    "shares":        round(shares, 2),
                    "pnl":           round(pnl, 2),
                    "pnl_pct":       round((exit_price / entry_price - 1) * 100, 3),
                    "capital_after": round(capital, 2),
                    "bars_held":     (date - entry_date).days,
                })
                in_position = False

        # ── Check entry signal ────────────────────────────────────────────────
        if not in_position and row.get(signal_col, False):
            entry_price = row["close"]
            sl_dist     = row["atr"] * sl_mult
            sl_price    = entry_price - sl_dist
            tp_price    = entry_price + sl_dist * tp_mult
            entry_value = capital
            shares      = entry_value / (entry_price * (1.0 + commission))
            in_position = True
            entry_date  = date

        equity_curve.append(capital)

    # ── Force-close open position at end of data ──────────────────────────────
    if in_position:
        last_date, last_row = rows[-1]
        exit_price = last_row["close"]
        proceeds   = shares * exit_price * (1.0 - commission)
        pnl        = proceeds - entry_value
        capital    = capital - entry_value + proceeds
        equity_curve[-1] = capital
        trades.append({
            "symbol":        symbol,
            "entry_date":    entry_date,
            "exit_date":     last_date,
            "entry_price":   round(entry_price, 4),
            "exit_price":    round(exit_price, 4),
            "sl_price":      round(sl_price, 4),
            "tp_price":      round(tp_price, 4),
            "exit_type":     "EOD",
            "shares":        round(shares, 2),
            "pnl":           round(pnl, 2),
            "pnl_pct":       round((exit_price / entry_price - 1) * 100, 3),
            "capital_after": round(capital, 2),
            "bars_held":     (last_date - entry_date).days,
        })

    metrics = compute_metrics(
        trades, equity_curve, initial_capital,
        df.index[0], df.index[-1], symbol, signal_col,
    )
    metrics["trades"]       = trades
    metrics["equity_curve"] = equity_curve
    return metrics
