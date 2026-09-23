# Laya Stock Analyzer prototype

Enter **ticker + current price**, then press **Analyze**. This is a local Windows
application with a five-session research outlook. It does not place orders.
The existing dashboard (`app.py`) and GitHub Pages site remain separate.

## Windows: first run

1. Download this branch as a ZIP from GitHub and extract it, or open the existing
   project folder supplied with this build.
2. Double-click **Start Analyzer.cmd**. It prepares an isolated environment on
   first use and opens the analyzer in your browser. Leave its window open.
3. Type a US ticker and its current dollar price. Press **Analyze**.

If Python is missing, install **Python 3.12** from https://www.python.org/downloads/windows/,
including the Python launcher, then double-click again. Python 3.11 is also
supported. The local build already has its environment prepared.

**Start Analyzer with AI.cmd** installs/runs Laya typed decisions and Chronos-2
as well as the lightweight model. First use downloads model weights to
`data/models/`; it takes longer and needs several GB of disk space. CPU inference
is supported; no paid inference service is used. Both launchers bind only to
your own computer. Close the console to stop the server.

The light launcher defaults to technicals plus a regularized logistic model.
The AI launcher uses technicals, logistic, Chronos-2, and Laya with equal weights.
This is an experimental comparison configuration, not an automatically selected
or statistically qualified winner. The screen lists the components actually used.

## Data and secrets

The analyzer reuses `mi/providers` with the existing names `TWELVEDATA_KEY`,
`POLYGON_KEY`, `FMP_KEY`, `FINNHUB_KEY`, and `ALPHAVANTAGE_KEY`. Locally these
are optional environment variables. **GitHub Actions secrets cannot be read back
onto the PC**; the new manual workflow can use them inside GitHub. No secret
values are written into source, results, or browser code. A `.env` file is not
automatically loaded.

When keys are absent or keyed providers fail validation, Yahoo Finance through
`yfinance` provides a keyless fallback. It can throttle or change; all providers
failing produces a visible error, never synthetic market history. Data access
and reuse remain subject to the vendor's terms. Default history starts in 2018.
Only completed bars **before today's New York date** are used, conservatively
excluding today even after close. Recent prices are cached for four hours (the
UI caches the result for one hour). The observation date and provider are shown.
Daily history older than five calendar days or a large quote/history discrepancy
forces WAIT and suppresses levels.

## What the result means

- **Direction/action:** average signed evidence; LONG above +0.35, SHORT below
  -0.35, otherwise WAIT. These are research signals.
- **Setup quality:** inspectable agreement thresholds, not a predicted win rate.
- **Reversal:** possible structure change or liquidity sweep from existing indicators.
- **Breakout:** completed close outside the prior 20-bar range with relative
  volume at least 1.5; an entered quote outside the range is only developing.
- **Entry:** the price you entered, not a verified executable quote. Stop distance
  is the larger of 1.5 ATR or 1% of entry; targets are 1R, 2R, 3R. WAIT has no levels.
- **Laya details:** genuine `choice`, `score`, and `noul` outputs in the evidence
  panel. Its uncalibrated raw scores do not represent stock success probabilities.

The logistic model fits only mature historical five-session labels. Laya is an
independent structured second opinion on compact technical facts. Chronos uses
256 observed closes and past volume, on an ordinal trading-session clock.
Models are loaded lazily and retained across requests. Failure is reported and
the remaining components are explicitly listed. No fallback is called Laya.

The original `mi/ml/DirectionalModel` qualification gate remains unchanged.
This prototype does not publish a qualified probability or claim that the
experimental ensemble passed that gate.

## Reproduce the backtest

In PowerShell opened in the repository folder:

```powershell
.\.venv\Scripts\python.exe -m mi.analyzer.backtest --tickers NVDA MU SPY --test-sessions 252
```

For all model combinations after the AI installation:

```powershell
$env:USE_TF = '0'
$env:HF_HOME = "$PWD\data\models"
$env:OMP_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -m mi.analyzer.backtest --tickers NVDA MU SPY --models linear,chronos,laya --test-sessions 252
```

Results go to `data/backtests/latest.json`. Each ticker includes data hash,
source/fetch time, package versions, test dates, every signal/trade, returns,
daily closing drawdown, exposure, win rate, and explicit missing-model statuses.
Use `--end YYYY-MM-DD`, `--start YYYY-MM-DD`, `--cost-bps 30`, and `--output PATH`
to control reproducibility and test cost sensitivity. Fetching again may produce
revised vendor history; preserve the local cache and report hashes.

`python -m pytest tests -q` runs offline verification, including the two-input UI.
`requirements-tested-windows.txt` records the exact installed dependency versions
used for the initial Windows checks. The smaller requirement files remain the
normal installation entry points.
The manual GitHub Action **Analyzer backtest (artifact only)** can run on this
branch using existing repository secrets. It uploads results and has read-only
repository permissions; it does not commit data or deploy the site. GitHub may
not show a newly added manual workflow in its menu until it exists on the default
branch; the local commands do not depend on that.

## Method and limits

Signals are generated from expanding historical prefixes; the logistic model
is refit on each observation with matured labels only. Positions enter at the
**next open**. Every strategy uses the same non-overlapping five-session slots,
with a full exit at the stop, target 2, or final close. After an early exit it
holds cash through the rest of the slot. If stop and target touch in one bar,
stop is assumed first. Adverse gaps fill at the opening price. Assumed round-trip
commission/slippage is 15 basis points and short borrow is 3% annually. The
reported drawdown marks open positions at daily close, not only after exits.
The backtest evaluates the raw score strategies. The live screen additionally
suppresses actions for stale history or a large entered-price discrepancy; those
user-quote checks are not represented by this daily-bar backtest.

No strategy is promoted automatically. A higher return in one exploratory test
does not establish predictive accuracy, and comparing several combinations
increases selection bias. These are selected surviving tickers, not a complete
point-in-time universe. Pretrained model training overlap cannot be ruled out.
Dividends, taxes, variable borrow/locate availability, liquidity and actual
intraday price paths are omitted. Splits/adjustments depend on the vendor.
Historical results need independent future validation before use as a trading aid.

Primary model documentation:
- https://huggingface.co/convaiinnovations/laya-typed-decisions
- https://github.com/NandhaKishorM/laya
- https://github.com/amazon-science/chronos-forecasting

Laya's model card explicitly warns that its specialist checkpoint was trained
on four synthetic business workflows and needs domain-specific validation and
calibration. Chronos' general forecasting benchmarks do not establish a stock
trading edge.
