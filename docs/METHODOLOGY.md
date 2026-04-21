# Methodology

## Layer 0 — TTM Squeeze Base Signal

Entry requires all four conditions simultaneously:

1. **Stable bull regime** — `log_return > rolling_mean(20)` AND `vol_z < 0.5`
   - Volatility z-score: `(sigma_20 - SMA_50(sigma_20)) / STD_50(sigma_20)`
   - Rejects panic rallies and high-volatility environments

2. **Squeeze release** — `upperBB > upperKC` AND `lowerBB < lowerKC`
   - Bollinger Bands: SMA-20 ± 2σ
   - Keltner Channels: SMA-20 ± 1.5 × ATR (Wilder RMA)
   - Compressed range breaks out = energy release

3. **Price above Keltner upper** — `close > upperKC` (directional confirmation)

4. **Institutional volume** — `volume > SMA_20(volume)` (smart-money participation)

## Layer 1 — Rule Filters (Trend + Duration)

- **SMA-200 trend filter**: only trade when `close > SMA(200)` (bull bias)
- **Squeeze duration**: squeeze must have been active for ≥ 3 consecutive bars before release
  - Prevents triggering on trivially short compressions

## Layer 2 — ADX Momentum Filter

- **ADX > 25**: only trade when directional strength is confirmed
- Uses Wilder's smoothed DI+ / DI− (matches Pine Script `ta.adx`)

## Layer 3 — XGBoost TP Scorer

Walk-forward binary classifier predicting whether price hits TP before SL.

**Features** (9 total):
| Feature | Description |
|---------|-------------|
| `vol_z` | Volatility regime |
| `sqz_duration` | Bars in squeeze before release |
| `atr_ratio` | ATR / close (normalised vol) |
| `vol_ratio` | volume / vol_sma |
| `adx` | Trend strength |
| `kc_breakout` | (close − upperKC) / upperKC |
| `sma200_dist` | (close − SMA200) / SMA200 |
| `bb_width` | (upperBB − lowerBB) / basis |
| `day_of_week` | Calendar effect |

**Training**: first 65% of signals → label using walk-forward SL/TP simulation  
**Threshold**: predicted probability ≥ 0.55 required for entry

## Layer 4 — HMM Regime Filter

2-state Gaussian HMM fitted on `log_returns`.

- State with higher mean log-return = **bull** → trade
- State with lower mean = **bear/choppy** → skip
- Transition matrix learned from 10 years of PSX data

## Risk Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Stop-loss | `entry − ATR × 2.0` | Wilder ATR, matches Pine Script |
| Take-profit | `SL_dist × 1.5` | 1:1.5 R:R minimum |
| Position size | 100% equity | Single concentrated bet per signal |
| Commission | 0.02% per side | Approximate PSX brokerage |
