"""
PSX Institutional Squeeze — parameter optimisation via backtesting.py.

Sweeps sqz_len, mult_kc, atr_mult, rr_ratio to maximise Win Rate.

Usage
-----
    python strategies/optimization.py
    python strategies/optimization.py PSO.KA OGDC.KA

Dependencies
------------
    pip install backtesting pandas-ta
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed: pip install yfinance")
    sys.exit(1)


class InstitutionalSqueeze(Strategy):
    """
    Institutional Squeeze Strategy [Haseeb Zahid]
    Enhanced with SMA trend filter and optimisable parameters.
    """
    lookback  = 20
    sqz_len   = 20
    mult_bb   = 2.0
    mult_kc   = 1.5
    atr_mult  = 2.0   # stop-loss multiplier
    rr_ratio  = 1.5   # take-profit / stop-loss ratio
    trend_len = 200   # SMA trend filter period

    def init(self):
        close  = pd.Series(self.data.Close)
        high   = pd.Series(self.data.High)
        low    = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Volatility regime
        log_ret        = np.log(close / close.shift(1))
        mu             = ta.sma(log_ret, length=self.lookback)
        sigma          = ta.stdev(log_ret, length=self.lookback)
        sigma_sma_50   = ta.sma(sigma, length=50)
        sigma_stdev_50 = ta.stdev(sigma, length=50)
        vol_z          = (sigma - sigma_sma_50) / sigma_stdev_50
        is_stable_bull = (log_ret > mu) & (vol_z < 0.5)

        # Bollinger Bands
        bbands = ta.bbands(close, length=self.sqz_len, std=self.mult_bb)
        if bbands is not None and not bbands.empty:
            lowerBB = bbands.iloc[:, 0]
            upperBB = bbands.iloc[:, 2]
        else:
            lowerBB = pd.Series(np.nan, index=close.index)
            upperBB = pd.Series(np.nan, index=close.index)

        # Keltner Channels
        kc = ta.kc(high, low, close, length=self.sqz_len, scalar=self.mult_kc)
        if kc is not None and not kc.empty:
            lowerKC = kc.iloc[:, 0]
            upperKC = kc.iloc[:, 2]
        else:
            lowerKC = pd.Series(np.nan, index=close.index)
            upperKC = pd.Series(np.nan, index=close.index)

        squeeze_release = (upperBB > upperKC) & (lowerBB < lowerKC)

        vol_sma   = ta.sma(volume, length=20)
        inst_flow = volume > vol_sma
        sma_trend = ta.sma(close, length=self.trend_len)
        atr_obj   = ta.atr(high, low, close, length=14)
        if atr_obj is None:
            atr_obj = pd.Series(np.nan, index=close.index)

        self.is_stable_bull  = self.I(lambda x: x, is_stable_bull)
        self.squeeze_release = self.I(lambda x: x, squeeze_release)
        self.upperKC         = self.I(lambda x: x, upperKC)
        self.inst_flow       = self.I(lambda x: x, inst_flow)
        self.sma_trend       = self.I(lambda x: x, sma_trend)
        self.atr_val         = self.I(lambda x: x, atr_obj)

    def next(self):
        if (np.isnan(self.upperKC[-1]) or
                np.isnan(self.atr_val[-1]) or
                np.isnan(self.sma_trend[-1])):
            return

        long_entry = (
            self.is_stable_bull[-1]  and
            self.squeeze_release[-1] and
            self.data.Close[-1] > self.upperKC[-1]   and
            self.inst_flow[-1]       and
            self.data.Close[-1] > self.sma_trend[-1]
        )

        if long_entry and not self.position.is_long:
            sl_dist = self.atr_val[-1] * self.atr_mult
            tp_dist = sl_dist * self.rr_ratio
            entry   = self.data.Close[-1]
            self.buy(size=0.99, sl=entry - sl_dist, tp=entry + tp_dist)


def run_optimization(tickers: list, commission_rate: float = 0.0015):
    print("--------------------------------------------------")
    print("PSX Institutional Squeeze — Optimisation Runner")
    print("--------------------------------------------------")
    print("Objective: Maximise Win Rate [%]")

    for ticker_sym in tickers:
        print(f"\n==========================================")
        print(f"Analysing {ticker_sym} (10yr data)...")

        df = yf.Ticker(ticker_sym).history(period="10y")
        if df.empty:
            print(f"Error: no data for {ticker_sym}.")
            continue

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        bt       = Backtest(df, InstitutionalSqueeze,
                            cash=10_000_000, commission=commission_rate,
                            exclusive_orders=True)
        baseline = bt.run()

        print(f"Optimising parameters for {ticker_sym}...")
        stats = bt.optimize(
            sqz_len=range(10, 31, 10),
            mult_kc=[1.3, 1.5, 1.7],
            atr_mult=[2.0, 2.5, 3.0],
            rr_ratio=[1.0, 1.2, 1.5],
            maximize="Win Rate [%]",
            constraint=lambda p: p.sqz_len >= 10,
        )

        print(f"\n--- Results for {ticker_sym} ---")
        print(f"  sqz_len   : {stats['_strategy'].sqz_len}")
        print(f"  mult_kc   : {stats['_strategy'].mult_kc}")
        print(f"  atr_mult  : {stats['_strategy'].atr_mult}")
        print(f"  rr_ratio  : {stats['_strategy'].rr_ratio}")
        print(f"\n  Original Win Rate : {baseline['Win Rate [%]']:.2f}%")
        print(f"  Optimised Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"  Total Return      : {stats['Return [%]']:.2f}%")
        print(f"  Max Drawdown      : {stats['Max. Drawdown [%]']:.2f}%")
        print(f"  Total Trades      : {stats['# Trades']}")

    print("\n--------------------------------------------------")
    print("Optimisation complete.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else [
        "PSO.KA", "HUBC.KA", "OGDC.KA", "NCPL.KA"
    ]
    run_optimization(tickers)
