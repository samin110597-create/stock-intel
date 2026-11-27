# Initial Laya Stock Analyzer backtest

Exploratory results from real Yahoo Finance daily history. These results do not establish a reliable trading edge.

The strategy parameters and model combinations were fixed before these runs. No combination is automatically promoted.

| Combination | NVDA net return | MU net return | SPY net return |
|---|---:|---:|---:|
| technicals | -32.21% | +17.82% | -3.25% |
| technicals+linear | -32.16% | +41.94% | -3.60% |
| technicals+chronos | -25.48% | +1.41% | -5.33% |
| technicals+laya | -51.05% | -19.34% | -17.11% |
| technicals+chronos+laya | -22.95% | -9.15% | -4.91% |
| all | -15.10% | +51.07% | -2.82% |
| Buy and hold (price return, less 15 bps) | +24.64% | +500.46% | +14.99% |

Buy-and-hold is continuously invested and has different exposure and risk. Dividends are excluded for both.

## What improved?

- **NVDA:** highest observed return was **all** (-15.10%), +17.11 percentage points versus technicals. Forecasts use historical prefixes, but selecting a winner using these same test results does not establish future superiority.
- **MU:** highest observed return was **all** (+51.07%), +33.24 percentage points versus technicals. Forecasts use historical prefixes, but selecting a winner using these same test results does not establish future superiority.
- **SPY:** highest observed return was **all** (-2.82%), +0.43 percentage points versus technicals. Forecasts use historical prefixes, but selecting a winner using these same test results does not establish future superiority.

## Per-ticker detail

### NVDA

Test: **2025-09-19 through 2026-09-17**. Input history starts 2018-01-02; 2191 daily bars. Source: yahoo_finance.

| Combination | Trades | Net return | Daily close max drawdown | Trade win rate | Direction accuracy |
|---|---:|---:|---:|---:|---:|
| technicals | 26 | -32.21% | -37.00% | 30.8% | 34.6% |
| technicals+linear | 20 | -32.16% | -36.95% | 30.0% | 35.0% |
| technicals+chronos | 17 | -25.48% | -32.73% | 35.3% | 41.2% |
| technicals+laya | 42 | -51.05% | -53.42% | 31.0% | 35.7% |
| technicals+chronos+laya | 19 | -22.95% | -24.19% | 31.6% | 31.6% |
| all | 10 | -15.10% | -16.47% | 30.0% | 30.0% |

Data SHA-256: `8ce52e0d1eb1a44fe53c2030183b79979bb35d0f8100269fdc03d4efa0d752b0`

### MU

Test: **2025-09-19 through 2026-09-17**. Input history starts 2018-01-02; 2191 daily bars. Source: yahoo_finance.

| Combination | Trades | Net return | Daily close max drawdown | Trade win rate | Direction accuracy |
|---|---:|---:|---:|---:|---:|
| technicals | 39 | +17.82% | -36.19% | 51.3% | 64.1% |
| technicals+linear | 34 | +41.94% | -27.69% | 52.9% | 67.6% |
| technicals+chronos | 25 | +1.41% | -41.75% | 52.0% | 64.0% |
| technicals+laya | 35 | -19.34% | -48.95% | 48.6% | 54.3% |
| technicals+chronos+laya | 24 | -9.15% | -47.05% | 50.0% | 54.2% |
| all | 18 | +51.07% | -27.25% | 61.1% | 66.7% |

Data SHA-256: `29b831bc3e2a1baac4779fc1c39ebfce6747b49d751d4ec5781bfa7d64cd4d8b`

### SPY

Test: **2025-09-19 through 2026-09-17**. Input history starts 2018-01-02; 2191 daily bars. Source: yahoo_finance.

| Combination | Trades | Net return | Daily close max drawdown | Trade win rate | Direction accuracy |
|---|---:|---:|---:|---:|---:|
| technicals | 31 | -3.25% | -9.17% | 54.8% | 58.1% |
| technicals+linear | 28 | -3.60% | -10.94% | 53.6% | 57.1% |
| technicals+chronos | 20 | -5.33% | -9.07% | 50.0% | 55.0% |
| technicals+laya | 40 | -17.11% | -19.20% | 40.0% | 42.5% |
| technicals+chronos+laya | 22 | -4.91% | -11.40% | 50.0% | 50.0% |
| all | 20 | -2.82% | -9.45% | 50.0% | 50.0% |

Data SHA-256: `391e8dbf70365a093325d88e6d2aacdc271ce46cd3e8dfab2d377656f204df26`

## Method and caveats

Each ticker has 50 non-overlapping five-session opportunities. Signals use historical prefixes, with the small classifier refit using only mature labels. Entries use the following open. Stops are 1.5 ATR (at least 1%); full exits use stop, 2R target, or the fifth session close. Stop wins ambiguous bars; adverse gaps fill at open. Round-trip costs are 15 bps; short borrow is assumed at 3% annually. Cash earns zero.

Win rate means profitable executed trades after modeled costs. Direction accuracy tests the raw five-session price direction on executed signals, regardless of whether a stop closed the trade earlier. They are different metrics.

The backtest evaluates raw score strategies. The live screen can additionally refuse stale or mismatched user-entered prices, so this is not an exact simulation of every live UI guard.

- Exploratory historical comparison, not proof of a reliable or profitable edge.
- User-selected surviving US tickers; survivorship and selection bias; no delisted universe.
- Foundation-model pretraining data may overlap test history; contamination cannot be excluded.
- No stock-domain probability calibration or model selection on an independent final holdout.
- Current vendor-adjusted price history, not archived point-in-time data; dividends and short dividend payments excluded.
- Daily OHLC cannot reconstruct intraday barrier ordering; conservative stop-first assumption.
- Fixed per-trade costs and borrow assumptions omit liquidity, locate availability, variable spreads and taxes.
- Drawdown uses daily closing marks; intraday drawdown can be worse. Entry uses next open, not a live quote.
- Full exit at target 2 or stop/horizon; target 1 and target 3 are reference levels, not partial fills.
- Lightweight/AI scores are experimental agreement scores. Legacy qualified-probability model is unchanged.

## Validation and reproduction

59 offline tests passed on Windows, including the two-input screen, invalid input, causal features, historical-prefix isolation, stop ordering, gap fills, stale-data refusal, and missing-model handling. Dependency consistency checks passed.

Run from the repository folder after installing the AI dependencies:

```powershell
.\.venv\Scripts\python.exe -m mi.analyzer.backtest --tickers NVDA MU SPY --end 2026-09-21 --models linear,chronos,laya --test-sessions 252 --output data/backtests/ensemble.json
```

See `ANALYZER.md` for setup, assumptions, and the lightweight run. `initial-backtest.json` contains every decision and the measured package versions.

Model references: [Laya model card](https://huggingface.co/convaiinnovations/laya-typed-decisions), [Chronos documentation](https://github.com/amazon-science/chronos-forecasting). Laya has not been validated or calibrated for stocks in this prototype.
