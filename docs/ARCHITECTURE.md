# Architecture

See the main [README](../README.md) for the full architecture diagram.

## Data Flow

```
psxterminal API / yfinance
        │
        ▼
src/data/fetcher.py          — OHLCV DataFrame (date index, lowercase cols)
        │
        ▼
src/squeeze/indicators.py    — append ATR, ADX, BB, KC, squeeze, vol_z …
        │
        ▼
src/squeeze/filters.py       — append filter_sma, filter_sqz, filter_adx
        │
        ├──► src/squeeze/signals.py      — base_signal (Layer 0)
        │
        ├──► src/ml/xgboost_scorer.py    — xgb_ok (Layer 3)
        │
        ├──► src/ml/hmm_regime.py        — hmm_bull (Layer 4)
        │
        ▼
src/squeeze/signals.py       — sig_rules, sig_ensemble
        │
        ▼
src/backtest/engine.py       — bar-by-bar simulation → trades, equity_curve
        │
        ▼
src/backtest/metrics.py      — win_rate, CAGR, max_drawdown, profit_factor
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/config.py` | Single source of truth for all hyperparameters |
| `src/data/fetcher.py` | Network I/O; returns clean OHLCV DataFrames |
| `src/squeeze/indicators.py` | Pure technical computations; no side effects |
| `src/squeeze/filters.py` | Rule-based gate columns (Layers 1 & 2) |
| `src/squeeze/signals.py` | Combine indicator + filter columns into signal booleans |
| `src/ml/hmm_regime.py` | Fit and decode 2-state Gaussian HMM (Layer 4) |
| `src/ml/xgboost_scorer.py` | Walk-forward XGBoost TP-probability scoring (Layer 3) |
| `src/backtest/engine.py` | Stateful trade simulator; Pine Script semantics |
| `src/backtest/metrics.py` | Stateless performance metric computation |
| `strategies/ensemble_squeeze.py` | CLI entry point — full 4-layer ensemble |
| `strategies/base_squeeze.py` | CLI entry point — Layer 0 only (benchmark) |
| `strategies/optimization.py` | Grid-search via backtesting.py library |
