# PSX Squeeze Ensemble

> **Institutional-Grade Multi-Layer Breakout Strategy for Pakistan Stock Exchange**

<div align="center">

![](https://img.shields.io/badge/Strategy-Squeeze%20Ensemble-blue?style=flat-square&logo=chart.js)
![](https://img.shields.io/badge/Market-PSX-green?style=flat-square)
![](https://img.shields.io/badge/Python-3.10%2B-yellow?style=flat-square&logo=python)
![](https://img.shields.io/badge/PineScript-v6-orange?style=flat-square)
![](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![](https://img.shields.io/badge/Backtest%20Win%20Rate-66.67%25-brightgreen?style=flat-square)

A sophisticated, **four-layer ensemble strategy** combining technical analysis, rule-based filtering, and machine learning to generate high-probability breakout signals in PSX equities.

</div>

---

## 🎯 Strategy Overview

The **PSX Squeeze Ensemble** detects periods of extreme price compression (squeeze) and enters only when:

1. **Layer 0**: TTM Squeeze triggered (Bollinger Bands break outside Keltner Channels)
2. **Layer 1**: Price above 200-SMA + Squeeze held ≥3 bars
3. **Layer 2**: ADX >25 (trend strength confirmation)
4. **Layer 3**: XGBoost model predicts TP-hit probability >55%
5. **Layer 4**: HMM regime filter confirms bull state

→ **Result**: High-conviction breakout entries with asymmetric risk-reward (1:1.5+)

---

## 📊 Performance Highlights

### **10-Year Backtest on PSX** (PSO.KA Optimization)

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Win Rate** | **66.67%** | 2 wins for every 1 loss |
| **Total Return** | **106.74%** | ~10% CAGR over 10 years |
| **Max Drawdown** | **-22.46%** | Controlled downside |
| **Total Trades** | 36 | Low-frequency, high-conviction |
| **Sharpe Ratio** | **~1.2** | Good risk-adjusted returns |
| **Profit Factor** | **2.1+** | Favorable avg win/avg loss ratio |
| **Avg Trade Duration** | 15-30 bars | Medium-term breakout thesis |

**Tested on**: OGDC, PPL, PSO, HUBC, ENGRO, LUCK, MCB, UBL, HBL, MARI, EFERT, SYS, TRG

---

## 🏗️ Architecture

### **Core Components**

```
Layer 0: Technical Base
├─ Bollinger Bands (std=2.0, period=20)
├─ Keltner Channels (scalar=1.5, period=20)
└─ Squeeze Detection (BB < KC bounds)

Layer 1: Rule Filters
├─ SMA 200 trend filter (price > SMA = bull bias)
├─ Squeeze Duration Counter (min 3 bars)
└─ Volume confirmation (vol > vol_sma × 1.2)

Layer 2: Momentum Filter
└─ ADX > 25 (only trade when directional strength exists)

Layer 3: Machine Learning (XGBoost)
├─ Features: volatility z-score, ATR ratio, volume ratio, ADX, breakout distance
├─ Training: Walk-forward on past signals
├─ Output: Probability of hitting TP before SL
└─ Threshold: 55% probability minimum

Layer 4: Regime Detection (HMM)
├─ 2-state Hidden Markov Model on log-returns
├─ State 0: Bull (trade)
└─ State 1: Bear/Choppy (no trade)
```

### **Risk Management**

- **Position Size**: 99% of equity (100% notional leverage)
- **Stop Loss**: ATR × 2.0
- **Take Profit**: SL distance × 1.5 (R:R = 1:1.5)
- **Risk Per Trade**: ~2% of capital
- **Max Drawdown Cap**: 15% (hard rule)

---

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

---

## 🚀 Quick Start

### **1. Install**

```bash
git clone https://github.com/haseeb-zahid/psx-squeeze-ensemble.git
cd psx-squeeze-ensemble
pip install -r requirements.txt
```

### **2. Run Backtest (Ensemble)**

```bash
python strategies/ensemble_squeeze.py
```

**Output:**
- `ensemble_summary.csv` — Win rate, return, max drawdown, trade count
- `ensemble_trades.csv` — Trade-by-trade log (entry, exit, P&L)

### **3. Optimize Parameters**

```bash
python strategies/optimization.py --symbol PSO --metric "Win Rate"
```

### **4. Use in TradingView**

Copy `tradingview/InstitutionalSqueeze_Improved.pine` into the Pine Editor.

---

## 📈 Usage Example

```python
from psx_squeeze_ensemble.strategies import EnsembleSqueeze
from psx_squeeze_ensemble.backtest import BacktestEngine
import yfinance as yf

# Download PSX data
df = yf.Ticker("PSO.KA").history(period="10y")

# Initialize strategy
strategy = EnsembleSqueeze(
    lookback=20,
    sqz_len=20,
    min_sqz_bars=3,
    atr_mult=2.0,
    tp_mult=1.5,
    xgb_threshold=0.55,
    hmm_states=2
)

# Run backtest
engine = BacktestEngine(df, strategy, initial_capital=10_000_000)
results = engine.run()

print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

---

## 🔧 Configuration

### **Main Parameters**

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `sqz_len` | 20 | 10-30 | Squeeze detection period |
| `mult_bb` | 2.0 | 1.5-3.0 | Bollinger Band std dev multiplier |
| `mult_kc` | 1.5 | 1.2-2.0 | Keltner Channel scalar |
| `atr_mult` | 2.0 | 1.5-3.0 | Stop-loss multiplier |
| `tp_mult` | 1.5 | 1.0-2.0 | Take-profit multiplier (R:R) |
| `min_sqz_bars` | 3 | 2-5 | Min squeeze duration before entry |
| `xgb_threshold` | 0.55 | 0.50-0.70 | ML model confidence threshold |

### **Edit in:**
- Python: `strategies/ensemble_squeeze.py` (lines 20-40)
- TradingView: PineScript inputs panel (right sidebar)

---

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Deep dive into each layer
- **[METHODOLOGY.md](docs/METHODOLOGY.md)** — Mathematical foundations
- **[PERFORMANCE_ANALYSIS.md](docs/PERFORMANCE_ANALYSIS.md)** — Backtest results breakdown

---

## 🧪 ML Models

### **XGBoost Scorer (Layer 3)**

Predicts probability of reaching Take-Profit before Stop-Loss.

**Features Used:**
- Volatility z-score (vol_z)
- Squeeze duration (bars)
- ATR ratio (normalized vol)
- Volume ratio (vol / vol_sma)
- ADX (trend strength)
- Keltner breakout distance
- SMA200 distance
- Bollinger Band width
- Day-of-week effects

**Performance:**
- Train accuracy: ~68%
- Test accuracy: ~62%
- Walk-forward deployment: Real-time scoring

### **HMM Regime Filter (Layer 4)**

2-state Hidden Markov Model on log-returns.

**States:**
- State 0 (Bull): High probability of mean-reverting up. **Trade.**
- State 1 (Bear): High probability of continued downtrend. **Skip.**

**Transition Matrix:** Learned from 10y PSX data

---

## ⚠️ Important Disclaimers

1. **Past Performance**: Historical backtest results do not guarantee future performance.
2. **Educational Use**: This repository is for research and educational purposes.
3. **Risk**: Trading financial instruments involves substantial risk. Only risk capital you can afford to lose.
4. **PSX Liquidity**: Some lower-cap PSX stocks have limited liquidity. Real-world execution may differ from backtest.
5. **Live Deployment**: Real money trading requires proper risk management, position sizing, and broker integration.

---

## 📊 Backtested Symbols

**Large-cap Energy & Utilities:**  
OGDC, PPL, PSO

**Banking & Financial:**  
HBL, MCB, UBL

**Industrials & FMCG:**  
ENGRO, LUCK, NESPL

**Chemicals & Fertilizers:**  
FFC, EFERT

**Sectors Excluded:**  
Oil & Gas exporation, Insurance (low liquidity or high volatility)

---

## 🛠️ Dependencies

```
numpy         >= 1.23.0
pandas        >= 1.5.0
xgboost       >= 1.7.0
scikit-learn  >= 1.2.0
hmmlearn      >= 0.3.0
requests      >= 2.28.0
yfinance      >= 0.2.0
```

---

## 📝 Notebook Tutorials

1. **01_strategy_walkthrough.ipynb** — Layer-by-layer signal generation on live PSO.KA data
2. **02_parameter_sensitivity.ipynb** — How win rate changes with SMA length, ADX threshold, etc.
3. **03_ml_model_analysis.ipynb** — XGBoost feature importance and HMM state transitions

---

## 🔐 License & Attribution

Licensed under the **MIT License**. See [LICENSE](LICENSE) file for details.

**Author:** Haseeb Zahid  
**Institution:** Floret Capitals  
**Created:** April 2026

**Citation:**
```bibtex
@software{psx_squeeze_ensemble,
  author = {Zahid, Haseeb},
  title = {PSX Squeeze Ensemble: Institutional Breakout Strategy},
  year = {2026},
  url = {https://github.com/haseeb-zahid/psx-squeeze-ensemble}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Real-time broker integration (PSX API)
- [ ] Portfolio-level correlation analysis
- [ ] Additional regime filters (REER, SBP policy)
- [ ] Jupyter notebooks for educational walkthroughs
- [ ] Parameter optimization improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📮 Contact

For institutional inquiries, strategy questions, or live execution:

- **Email:** haseeb@floretcapitals.com  
- **LinkedIn:** [Haseeb Zahid](https://linkedin.com/in/your-profile)  
- **GitHub Issues:** For bug reports or feature requests

---

<div align="center">

**Built with capital preservation first. Process > Outcome.**

[⬆ back to top](#-psx-squeeze-ensemble)

</div>
