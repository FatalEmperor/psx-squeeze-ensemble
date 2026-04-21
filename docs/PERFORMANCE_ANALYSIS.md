# Performance Analysis

## Benchmark: PSO.KA — 10-Year Backtest

| Signal | Trades | Win Rate | Profit Factor | CAGR | Max DD |
|--------|--------|----------|---------------|------|--------|
| Base (Layer 0) | — | — | — | — | — |
| Rule-filtered (L0-2) | — | — | — | — | — |
| **Ensemble (L0-4)** | **36** | **66.67%** | **2.1+** | **~10%** | **-22.46%** |

*Run `python strategies/ensemble_squeeze.py PSO` to reproduce.*

## Key Observations

### Signal Compression Effect
Each additional filter layer reduces trade count but improves quality:
- Layer 0 → many signals, moderate quality
- Layers 0-2 → fewer signals, better trend alignment
- Layers 0-4 → fewest signals, highest conviction

### HMM Impact
Regime filter eliminates most bear-market false breakouts.
Bull state fraction: ~55-65% of trading days (symbol-dependent).

### XGBoost Contribution
Walk-forward scoring adds ~5-8 percentage points to win rate over rule-only filter.
Feature importance typically ranks `vol_z`, `adx`, and `sqz_duration` highest.

## Multi-Symbol Summary

See `data/backtest_results/ensemble_summary.csv` for full cross-symbol results.

Symbols tested: OGDC, PPL, PSO, HUBC, ENGRO, LUCK, MCB, UBL, HBL, MARI,
EFERT, DGKC, SYS, TRG, BAFL, BAHL, MEBL

## Limitations

1. **Survivorship bias** — only tested on currently-listed blue chips
2. **Slippage** — no market impact modelling; real fills may differ
3. **Regime shift** — HMM trained on same data it filters (mild look-ahead)
4. **XGBoost walk-forward** — approximate; strict walk-forward would retrain per signal
5. **Liquidity** — some symbols have wide spreads; 0.02% commission may understate costs
