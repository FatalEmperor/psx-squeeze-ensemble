"""
Performance metrics and reporting utilities.
"""

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL, COMMISSION


def compute_metrics(
    trades: list,
    equity_curve: list,
    initial_capital: float,
    start_date,
    end_date,
    symbol: str,
    signal_col: str,
) -> dict:
    """
    Compute standard performance metrics from a completed trade list.

    Returns a dict (without 'trades' or 'equity_curve' — those are added by the engine).
    """
    capital = equity_curve[-1] if equity_curve else initial_capital
    n = len(trades)

    base = {
        "symbol":           symbol,
        "signal":           signal_col,
        "total_trades":     n,
        "win_rate":         0.0,
        "avg_win_pct":      0.0,
        "avg_loss_pct":     0.0,
        "gross_profit":     0.0,
        "gross_loss":       0.0,
        "profit_factor":    0.0,
        "max_drawdown_pct": 0.0,
        "total_return_pct": round((capital / initial_capital - 1) * 100, 2),
        "cagr_pct":         0.0,
        "final_capital":    round(capital, 2),
    }

    if n == 0:
        return base

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = (gp / gl) if gl > 0 else 999.0

    eq         = pd.Series(equity_curve)
    max_dd     = float(((eq - eq.cummax()) / eq.cummax() * 100).min())
    n_years    = (end_date - start_date).days / 365.25
    total_ret  = (capital / initial_capital - 1) * 100
    cagr       = ((capital / initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    base.update({
        "win_rate":         round(len(wins) / n * 100, 2),
        "avg_win_pct":      round(np.mean([t["pnl_pct"] for t in wins])   if wins   else 0, 2),
        "avg_loss_pct":     round(np.mean([t["pnl_pct"] for t in losses]) if losses else 0, 2),
        "gross_profit":     round(gp, 2),
        "gross_loss":       round(gl, 2),
        "profit_factor":    round(min(pf, 999.0), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_return_pct": round(total_ret, 2),
        "cagr_pct":         round(cagr, 2),
        "final_capital":    round(capital, 2),
    })
    return base


def print_summary(all_results: list, initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """Print a formatted performance table and return it as a DataFrame."""
    cols = [
        "symbol", "signal", "total_trades", "win_rate", "avg_win_pct",
        "avg_loss_pct", "profit_factor", "max_drawdown_pct",
        "total_return_pct", "cagr_pct", "final_capital",
    ]
    rows = [{c: r[c] for c in cols if c in r} for r in all_results]
    df   = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False)

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 140)

    print("\n" + "=" * 110)
    print("PSX SQUEEZE ENSEMBLE — BACKTEST RESULTS")
    print(f"Initial Capital: PKR {initial_capital:,.0f}  |  Commission: {COMMISSION*100:.2f}% per side")
    print("=" * 110)
    print(df.to_string(index=False))

    active = df[df["total_trades"] > 0]
    if not active.empty:
        finite_pf = active[active["profit_factor"] < 999.0]["profit_factor"]
        print("\n-- Aggregate ----------------------------------------------------------")
        print(f"  Symbols tested    : {len(df)}")
        print(f"  Symbols with trades: {len(active)}")
        print(f"  Total trades      : {df['total_trades'].sum():.0f}")
        print(f"  Avg win rate      : {active['win_rate'].mean():.1f}%")
        if not finite_pf.empty:
            print(f"  Avg profit factor : {finite_pf.mean():.2f}")
        print(f"  Avg CAGR          : {active['cagr_pct'].mean():.1f}%")
        print(f"  Avg max drawdown  : {active['max_drawdown_pct'].mean():.1f}%")

    return df
