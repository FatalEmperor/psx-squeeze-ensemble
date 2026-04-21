# Institutional Squeeze Trading Strategy - PSX

![Strategy Banner](https://img.shields.io/badge/Strategy-Institutional_Squeeze-blue) 
![Market](https://img.shields.io/badge/Market-PSX-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![PineScript](https://img.shields.io/badge/PineScript-v6-orange)

A sophisticated multi-layered institutional squeeze trading strategy designed specifically for the Pakistan Stock Exchange (PSX). This project combines classical technical analysis with Machine Learning (XGBoost) and Statistical Modeling (Hidden Markov Models) to filter high-probability trading signals.

## 🚀 Key Strategy Layers

The strategy evolves through four distinct layers to ensure only the highest quality "institutional" moves are captured:

1.  **The Base Squeeze**: Based on the TTM Squeeze principle (Bollinger Bands vs. Keltner Channels), identifying periods of extreme price compression.
2.  **Layer 1 (Momentum & Trend)**: 
    *   **SMA 200 Filter**: Prevails only in long-term bullish trends.
    *   **Squeeze Maturity**: Requires the squeeze to hold for at least 3-5 bars before a release signal is valid.
3.  **Layer 2 (Trend Strength)**: **ADX > 25** filter to ensure the breakout has sufficient directional strength.
4.  **Layer 3 (Machine Learning)**: **XGBoost Scorer** trained on past signal performance to predict the probability of hitting Take-Profit vs. Stop-Loss targets based on volatility and volume z-scores.
5.  **Layer 4 (Statistical Regime)**: **Hidden Markov Model (HMM)** regime detection to identify "Bull" and "Bear" market states, automatically pausing trading during high-risk regimes.

## 📊 Performance Highlights (PSO.KA Optimization)

| Metric | Base Strategy | Optimized Ensemble |
| :--- | :--- | :--- |
| **Win Rate** | 44.74% | **66.67%** |
| **Total Return** | - | **106.74%** |
| **Max Drawdown** | - | **-22.46%** |
| **Trade Count** | - | 36 |

*Backtested over a 10-year window in the PSX market.*

## 📁 Repository Structure

```text
├── backtest_ensemble.py     # Advanced ensemble backtest (Python/ML/HMM)
├── backtest_squeeze.py      # Core squeeze logic and signal processing
├── psx_backtest.py          # Legacy backtesting and optimization suite
├── InstitutionalSqueeze_Improved.pine # TradingView Strategic Script (v6)
├── requirements.txt         # Python dependencies
├── ensemble_summary.csv     # Compiled backtest results
└── ensemble_trades.csv      # Detailed trade-by-trade log
```

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/psx-institutional-squeeze.git
cd psx-institutional-squeeze
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Ensemble Backtest
To run the full multi-layered backtest on a default set of PSX tickers:
```bash
python backtest_ensemble.py
```

To run on specific symbols:
```bash
python backtest_ensemble.py OGDC PPL PSO
```

### 4. TradingView Implementation
Copy the contents of `InstitutionalSqueeze_Improved.pine` into the TradingView Pine Editor and click "Add to Chart".

## ⚠️ Disclaimer
Trading financial instruments involves significant risk. This repository is for educational and research purposes only. Past performance is not indicative of future results. Always perform your own due diligence before deploying any capital.

---
*Created for PSX Market Intelligence.*
