import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest

class InstitutionalSqueeze(Strategy):
    """
    Institutional Squeeze Strategy [Haseeb Zahid]
    Enhanced with SMA Trend Filter and Optimization Parameters
    """
    # --- [INPUTS / OPTIMIZABLE] ---
    lookback = 20
    sqz_len  = 20
    mult_bb  = 2.0
    mult_kc  = 1.5
    
    # --- [NEW OPTIMIZABLE INPUTS] ---
    atr_mult = 2.0  # SL multiplier
    rr_ratio = 1.5  # TP/SL ratio
    trend_len = 200 # SMA Trend filter length

    def init(self):
        # Convert backtesting data arrays to pandas Series for ta calculations
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # --- [SECTION 1: MARKOV & VOLATILITY] ---
        log_ret = np.log(close / close.shift(1))
        mu = ta.sma(log_ret, length=self.lookback)
        sigma = ta.stdev(log_ret, length=self.lookback)
        
        sigma_sma_50 = ta.sma(sigma, length=50)
        sigma_stdev_50 = ta.stdev(sigma, length=50)
        vol_z = (sigma - sigma_sma_50) / sigma_stdev_50
        
        # Stable, non-panic bull
        is_stable_bull = (log_ret > mu) & (vol_z < 0.5)

        # --- [SECTION 2: THE SQUEEZE (TTM Style)] ---
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

        # Squeeze release triggers when BB expand outside KC
        squeeze_release = (upperBB > upperKC) & (lowerBB < lowerKC)

        # --- [SECTION 3: ENTRY & FLOW] ---
        vol_sma = ta.sma(volume, length=20)
        inst_flow = volume > vol_sma
        
        # --- [SECTION 5: TREND FILTER] ---
        sma_trend = ta.sma(close, length=self.trend_len)

        # Wrapping indicators for use in next()
        self.is_stable_bull = self.I(lambda x: x, is_stable_bull)
        self.squeeze_release = self.I(lambda x: x, squeeze_release)
        self.upperKC = self.I(lambda x: x, upperKC)
        self.inst_flow = self.I(lambda x: x, inst_flow)
        self.sma_trend = self.I(lambda x: x, sma_trend)
        
        # ATR for risk
        atr_obj = ta.atr(high, low, close, length=14)
        if atr_obj is None:
            atr_obj = pd.Series(np.nan, index=close.index)
        self.atr_val = self.I(lambda x: x, atr_obj)

    def next(self):
        # Basic checks to avoid errors
        if np.isnan(self.upperKC[-1]) or np.isnan(self.atr_val[-1]) or np.isnan(self.sma_trend[-1]):
            return

        # Trigger logic
        long_entry = (self.is_stable_bull[-1] and 
                      self.squeeze_release[-1] and 
                      self.data.Close[-1] > self.upperKC[-1] and 
                      self.inst_flow[-1] and 
                      self.data.Close[-1] > self.sma_trend[-1])

        # Execution
        if long_entry and not self.position.is_long:
            sl_dist = self.atr_val[-1] * self.atr_mult
            tp_dist = sl_dist * self.rr_ratio
            
            entry_price = self.data.Close[-1]
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
            
            self.buy(size=0.99, sl=sl, tp=tp)


if __name__ == "__main__":
    import yfinance as yf
    
    print("--------------------------------------------------")
    print("PSX Institutional Squeeze - Optimization Runner")
    print("--------------------------------------------------")
    print("Objective: Maximize Win Rate [%] across PSX pairs")
    
    tickers = ["PSO.KA", "HUBC.KA", "OGDC.KA", "NCPL.KA"]
    commission_rate = 0.0015 # 0.15% per trade
    
    for ticker_sym in tickers:
        print(f"\n==========================================")
        print(f"Analyzing {ticker_sym} (10yr Data)...")
        
        # Download data
        df = yf.Ticker(ticker_sym).history(period="10y")
        if df.empty:
            print(f"Error: Could not retrieve {ticker_sym} data.")
            continue
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # Initialize Backtest
        bt = Backtest(df, InstitutionalSqueeze, cash=10000000, commission=commission_rate, exclusive_orders=True)
        
        # First check baseline
        baseline = bt.run()
        
        print(f"Optimizing parameters for {ticker_sym}...")
        
        # OPTIMIZATION LOOP
        # We optimize sqz_len, mult_kc, atr_mult, and rr_ratio to find better Win Rates
        optimized_stats = bt.optimize(
            sqz_len=range(10, 31, 10),
            mult_kc=[1.3, 1.5, 1.7],
            atr_mult=[2.0, 2.5, 3.0],
            rr_ratio=[1.0, 1.2, 1.5],
            maximize='Win Rate [%]',
            constraint=lambda p: p.sqz_len >= 10
        )
        
        print(f"\n--- Results for {ticker_sym} ---")
        print(f"Optimal Parameters: ")
        print(f"  Squeeze Len:  {optimized_stats['_strategy'].sqz_len}")
        print(f"  KC Mult:       {optimized_stats['_strategy'].mult_kc}")
        print(f"  ATR SL Mult:   {optimized_stats['_strategy'].atr_mult}")
        print(f"  RR Ratio:      {optimized_stats['_strategy'].rr_ratio}")
        print(f"\nPerformance Improvement:")
        print(f"  Original Win Rate:  {baseline['Win Rate [%]']:.2f}%")
        print(f"  NEW Win Rate:       {optimized_stats['Win Rate [%]']:.2f}%")
        print(f"  Total Return:       {optimized_stats['Return [%]']:.2f}%")
        print(f"  Max Drawdown:       {optimized_stats['Max. Drawdown [%]']:.2f}%")
        print(f"  Total Trades:       {optimized_stats['# Trades']}")
        
    print("\n--------------------------------------------------")
    print("Optimization Tasks Finished.")
    print("--------------------------------------------------")
