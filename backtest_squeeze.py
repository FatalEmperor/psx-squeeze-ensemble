"""
Institutional Squeeze Strategy — PSX Backtest
Translated from Pine Script by Haseeb Zahid

Strategy Logic:
  Entry : Stable bull regime + TTM squeeze release + close > upperKC + volume surge
  Exit  : ATR-based stop-loss (2x ATR) and take-profit (3x ATR = sl * 1.5)
  Size  : 100% of equity per trade (no pyramiding)
  Fee   : 0.02% per side (commission_value=0.02 in Pine = 0.02% in backtest)

Data Source: psxterminal.com REST API (paginated, 100 bars/request)
Fallback   : yfinance with .KA suffix if API unavailable
"""

import time
import json
import sys
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Try optional deps ────────────────────────────────────────────────────────
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ─── STRATEGY CONFIG ─────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000_000   # PKR
COMMISSION       = 0.0002        # 0.02% per side
LOOKBACK         = 20            # Markov lookback
SQZ_LEN          = 20            # Squeeze / BB / KC length
MULT_BB          = 2.0           # Bollinger Band multiplier
MULT_KC          = 1.5           # Keltner Channel multiplier
ATR_PERIOD       = 14
SL_MULT          = 2.0           # SL = entry - ATR * 2.0
TP_MULT          = 1.5           # TP dist = SL dist * 1.5  => TP = entry + ATR * 3.0
VOL_SMA_LEN      = 20
BACKTEST_YEARS   = 10

BASE_URL = "https://psxterminal.com/api"

# Common PSX blue-chip symbols (fallback if /api/symbols fails)
DEFAULT_SYMBOLS = [
    "OGDC", "PPL", "PSO", "HUBC", "ENGRO", "LUCK", "MCB",
    "UBL",  "HBL", "MARI", "EFERT", "DGKC", "MLCF", "UNITY",
    "AIRLINK", "SYS", "TRG", "BAFL", "BAHL", "MEBL",
    "SEARL", "COLG", "NESTLE", "FFC", "FATIMA", "POL", "SNGP",
    "NBP", "ABL", "ATRL",
]


# ─── DATA FETCHING ────────────────────────────────────────────────────────────

def _ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_psxterminal(symbol: str, years: int = BACKTEST_YEARS) -> pd.DataFrame:
    """Paginate psxterminal API to collect `years` of daily OHLCV data."""
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=365 * years + 30)   # extra buffer

    end_ts   = _ts_ms(end_dt)
    start_ts = _ts_ms(start_dt)

    all_records: list[dict] = []
    cursor = start_ts

    while cursor < end_ts:
        url    = f"{BASE_URL}/klines/{symbol}/1d"
        params = {"start": cursor, "end": end_ts, "limit": 100}

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise ConnectionError(f"API request failed: {exc}") from exc

        if not payload.get("success"):
            break

        records = payload.get("data") or []
        if not records:
            break

        all_records.extend(records)

        if len(records) < 100:
            break                               # reached the end

        last_ts = records[-1]["timestamp"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1                    # next page starts after last bar

        time.sleep(0.12)                        # ~8 req/s — well inside 100/min

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = (df
          .sort_values("date")
          .drop_duplicates("date")
          .set_index("date")
          [["open", "high", "low", "close", "volume"]]
          .astype(float))
    return df


def fetch_yfinance(symbol: str, years: int = BACKTEST_YEARS) -> pd.DataFrame:
    """Fallback: Yahoo Finance with .KA suffix."""
    if not HAS_YF:
        return pd.DataFrame()

    start = (datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")
    ticker = yf.Ticker(f"{symbol}.KA")
    raw    = ticker.history(start=start, auto_adjust=True)

    if raw.empty:
        return pd.DataFrame()

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.astype(float)
    return df


def fetch_data(symbol: str) -> pd.DataFrame:
    """Try psxterminal first, fall back to yfinance."""
    try:
        df = fetch_psxterminal(symbol)
        if not df.empty:
            return df
    except Exception:
        pass

    return fetch_yfinance(symbol)


def get_symbols() -> list[str]:
    """Fetch all PSX symbols from API, fall back to DEFAULT_SYMBOLS."""
    try:
        resp = requests.get(f"{BASE_URL}/symbols", timeout=10)
        data = resp.json()
        if data.get("success") and data.get("data"):
            syms = [s["symbol"] for s in data["data"] if isinstance(s, dict) and s.get("symbol")]
            if syms:
                return syms
    except Exception:
        pass
    return DEFAULT_SYMBOLS


# ─── INDICATORS ───────────────────────────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── True Range & ATR ─────────────────────────────────────────────────────
    df["tr"]  = true_range(df)
    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()

    # ── Markov / Volatility Regime ───────────────────────────────────────────
    df["log_ret"]     = np.log(df["close"] / df["close"].shift(1))
    df["mu"]          = df["log_ret"].rolling(LOOKBACK).mean()
    df["sigma"]       = df["log_ret"].rolling(LOOKBACK).std()
    sigma_sma50       = df["sigma"].rolling(50).mean()
    sigma_std50       = df["sigma"].rolling(50).std()
    df["vol_z"]       = (df["sigma"] - sigma_sma50) / sigma_std50
    df["stable_bull"] = (df["log_ret"] > df["mu"]) & (df["vol_z"] < 0.5)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    basis          = df["close"].rolling(SQZ_LEN).mean()
    bb_std         = df["close"].rolling(SQZ_LEN).std()
    df["upperBB"]  = basis + MULT_BB * bb_std
    df["lowerBB"]  = basis - MULT_BB * bb_std

    # ── Keltner Channels ──────────────────────────────────────────────────────
    ma_kc          = df["close"].rolling(SQZ_LEN).mean()
    range_kc       = df["tr"].rolling(SQZ_LEN).mean()        # SMA of TR (Pine: ta.sma(ta.tr, sqz_len))
    df["upperKC"]  = ma_kc + range_kc * MULT_KC
    df["lowerKC"]  = ma_kc - range_kc * MULT_KC

    # ── Squeeze ───────────────────────────────────────────────────────────────
    df["squeezed"]         = (df["lowerBB"] > df["lowerKC"]) & (df["upperBB"] < df["upperKC"])
    df["squeeze_release"]  = (df["upperBB"] > df["upperKC"]) & (df["lowerBB"] < df["lowerKC"])

    # ── Volume Filter ─────────────────────────────────────────────────────────
    df["vol_sma"]   = df["volume"].rolling(VOL_SMA_LEN).mean()
    df["inst_flow"] = df["volume"] > df["vol_sma"]

    # ── Entry Signal ──────────────────────────────────────────────────────────
    df["long_entry"] = (
        df["stable_bull"] &
        df["squeeze_release"] &
        (df["close"] > df["upperKC"]) &
        df["inst_flow"]
    )

    return df.dropna()


# ─── BACKTESTER ───────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, symbol: str) -> dict:
    """
    Bar-by-bar simulation.

    Pine Script behaviour replicated:
      - Entry executes at the CLOSE of the signal bar.
      - SL / TP checked from the NEXT bar onwards using intrabar high/low.
      - No pyramiding (position_size == 0 guard).
      - 100% of equity allocated per trade.
    """
    capital    = float(INITIAL_CAPITAL)
    equity_curve: list[float] = []
    trades:       list[dict]  = []

    in_position  = False
    entry_price  = 0.0
    sl_price     = 0.0
    tp_price     = 0.0
    entry_date   = None
    shares       = 0.0
    entry_value  = 0.0   # capital locked (including commission)

    rows = list(df.iterrows())

    for idx, (date, row) in enumerate(rows):

        # ── Manage open position (checked from bar AFTER entry) ──────────────
        if in_position and idx > 0:
            # Pessimistic fill: SL assumed at sl_price, TP at tp_price
            sl_hit = row["low"]  <= sl_price
            tp_hit = row["high"] >= tp_price

            exited     = False
            exit_price = 0.0
            exit_type  = ""

            if sl_hit and tp_hit:
                # Both triggered same bar — use whichever is worse (conservative)
                exit_price = sl_price
                exit_type  = "SL"
                exited     = True
            elif sl_hit:
                exit_price = sl_price
                exit_type  = "SL"
                exited     = True
            elif tp_hit:
                exit_price = tp_price
                exit_type  = "TP"
                exited     = True

            if exited:
                proceeds = shares * exit_price * (1.0 - COMMISSION)
                pnl      = proceeds - entry_value
                capital  = capital - entry_value + proceeds   # update equity
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

        # ── Check entry signal (only when flat) ──────────────────────────────
        if not in_position and row["long_entry"]:
            entry_price = row["close"]
            atr_val     = row["atr"]
            sl_dist     = atr_val * SL_MULT
            sl_price    = entry_price - sl_dist
            tp_price    = entry_price + sl_dist * TP_MULT   # sl_dist * 1.5

            # 100% of equity, commission on entry
            entry_value = capital                           # full equity in
            shares      = entry_value / (entry_price * (1.0 + COMMISSION))
            in_position = True
            entry_date  = date

        equity_curve.append(capital)

    # ── Force-close any open position at final bar ────────────────────────────
    if in_position:
        last_date, last_row = rows[-1]
        exit_price = last_row["close"]
        proceeds   = shares * exit_price * (1.0 - COMMISSION)
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

    # ── Performance Metrics ───────────────────────────────────────────────────
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "symbol":          symbol,
            "total_trades":    0,
            "win_rate":        0.0,
            "profit_factor":   0.0,
            "max_drawdown_pct":0.0,
            "total_return_pct":round((capital / INITIAL_CAPITAL - 1) * 100, 2),
            "cagr_pct":        0.0,
            "final_capital":   round(capital, 2),
            "gross_profit":    0.0,
            "gross_loss":      0.0,
            "avg_win_pct":     0.0,
            "avg_loss_pct":    0.0,
        }

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    eq           = pd.Series(equity_curve)
    rolling_max  = eq.cummax()
    max_dd       = float(((eq - rolling_max) / rolling_max * 100).min())

    n_years      = (df.index[-1] - df.index[0]).days / 365.25
    total_return = (capital / INITIAL_CAPITAL - 1) * 100
    cagr         = ((capital / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    return {
        "symbol":           symbol,
        "total_trades":     n_trades,
        "win_rate":         round(len(wins) / n_trades * 100, 2),
        "avg_win_pct":      round(np.mean([t["pnl_pct"] for t in wins])   if wins   else 0, 2),
        "avg_loss_pct":     round(np.mean([t["pnl_pct"] for t in losses]) if losses else 0, 2),
        "gross_profit":     round(gross_profit, 2),
        "gross_loss":       round(gross_loss, 2),
        "profit_factor":    round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        "max_drawdown_pct": round(max_dd, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct":         round(cagr, 2),
        "final_capital":    round(capital, 2),
        "trades":           trades,
        "equity_curve":     equity_curve,
    }


# ─── REPORTING ────────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]) -> pd.DataFrame:
    cols = [
        "symbol", "total_trades", "win_rate", "avg_win_pct", "avg_loss_pct",
        "profit_factor", "max_drawdown_pct", "total_return_pct", "cagr_pct",
        "final_capital",
    ]

    rows = [{c: r[c] for c in cols} for r in all_results]
    df   = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False)

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)

    print("\n" + "=" * 100)
    print("INSTITUTIONAL SQUEEZE STRATEGY — 10-YEAR PSX BACKTEST")
    print(f"Initial Capital: PKR {INITIAL_CAPITAL:,.0f} | Commission: {COMMISSION*100:.2f}% per side")
    print("=" * 100)
    print(df.to_string(index=False))

    active = df[df["total_trades"] > 0]
    if not active.empty:
        finite_pf = active[active["profit_factor"] < 999.0]["profit_factor"]
        print("\n-- Aggregate Stats --------------------------------------------------")
        print(f"  Symbols tested    : {len(df)}")
        print(f"  Symbols with trades: {len(active)}")
        print(f"  Total trades      : {df['total_trades'].sum():.0f}")
        print(f"  Avg win rate      : {active['win_rate'].mean():.1f}%")
        print(f"  Avg profit factor : {finite_pf.mean():.2f}" if not finite_pf.empty else "  Avg profit factor : N/A")
        print(f"  Avg CAGR          : {active['cagr_pct'].mean():.1f}%")
        print(f"  Avg max drawdown  : {active['max_drawdown_pct'].mean():.1f}%")

        print("\n-- Top 5 by CAGR ---------------------------------------------------")
        for _, row in active.head(5).iterrows():
            print(f"  {row['symbol']:12s}  CAGR: {row['cagr_pct']:6.1f}%  "
                  f"WR: {row['win_rate']:5.1f}%  PF: {row['profit_factor']:5.2f}  "
                  f"MaxDD: {row['max_drawdown_pct']:6.1f}%  Trades: {row['total_trades']:.0f}")

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(symbols: list[str] | None = None):
    print("Fetching PSX symbol list...")
    if symbols is None:
        symbols = get_symbols()
        if symbols == DEFAULT_SYMBOLS:
            print(f"API unavailable — using {len(symbols)} default blue-chip symbols")
        else:
            print(f"Found {len(symbols)} symbols via API")

    all_results: list[dict] = []
    all_trades:  list[dict] = []

    for symbol in symbols:
        sys.stdout.write(f"\r  {symbol:<12s}  fetching...")
        sys.stdout.flush()

        df_raw = fetch_data(symbol)
        if df_raw.empty or len(df_raw) < 120:
            sys.stdout.write(f"\r  {symbol:<12s}  SKIP (only {len(df_raw)} bars)\n")
            continue

        df = compute_indicators(df_raw)
        if df.empty:
            sys.stdout.write(f"\r  {symbol:<12s}  SKIP (indicator error)\n")
            continue

        result = backtest(df, symbol)
        all_results.append(result)
        all_trades.extend(result.get("trades", []))

        t  = result["total_trades"]
        wr = result.get("win_rate", 0.0)
        pf = result.get("profit_factor", 0.0)
        ca = result.get("cagr_pct", 0.0)
        dd = result.get("max_drawdown_pct", 0.0)

        sys.stdout.write(
            f"\r  {symbol:<12s}  "
            f"Trades: {t:3d}  WR: {wr:5.1f}%  PF: {pf:5.2f}  "
            f"CAGR: {ca:6.1f}%  MaxDD: {dd:6.1f}%\n"
        )

    if not all_results:
        print("No results — check network / API availability.")
        return [], []

    summary_df = print_summary(all_results)

    # ── Save output ───────────────────────────────────────────────────────────
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("backtest_trades.csv", index=False)
        print(f"\nTrades saved  -> backtest_trades.csv  ({len(all_trades)} rows)")

    summary_df.to_csv("backtest_summary.csv", index=False)
    print("Summary saved → backtest_summary.csv")

    return all_results, all_trades


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pass specific symbols as CLI args, e.g.:
    #   python backtest_squeeze.py OGDC PPL HUBC
    # Or run with no args for full symbol list.
    user_syms = sys.argv[1:] or None
    results, trades = main(user_syms)
