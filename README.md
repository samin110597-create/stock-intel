# stock-intel

A market analysis site built around one principle: **the system is not allowed to show you a number it cannot defend.**

**You never run Python.** A scheduled GitHub Action fetches the data, writes plain JSON into `docs/data/`, and commits it. GitHub Pages serves `docs/` as a static site. You add the repo, add your keys as secrets, turn on Pages, and visit the URL.

Most retail quant tooling fails silently. A provider returns an empty payload, a field gets forward-filled, an indicator computes cleanly on corrupted data, and the dashboard renders a confident chart built on nothing. This repo is structured so that cannot happen — every failure surfaces as a failure.

---

## What is here

| Module | What it does |
|---|---|
| `mi/providers` | Five-provider chain with ordered fallback and a full attempt log |
| `mi/contracts.py` | Schema validation that treats bad data as provider failure, not data |
| `mi/provenance.py` | Per-field source, fetch time, and observation date on every column |
| `mi/indicators.py` | Causal indicators + market structure (BOS/CHoCH, FVG, liquidity sweeps) |
| `mi/macro` | FRED catalogue, gold/silver panel, rule-based regime classifier |
| `mi/ml` | Directional model with walk-forward validation and a hard skill gate |
| `mi/etf` | Holdings, weight-based overlap, look-through exposure |
| `app.py` | Streamlit dashboard, Data Health tab first |

---

## How it runs

```
config/watchlist.json     you edit this
        |
        v
.github/workflows/refresh.yml     weekdays 11:10 UTC, or on demand
        |
        +-- pytest tests -q            43 offline tests gate the build
        +-- scripts/build_site.py      fetches, analyses, writes JSON
        +-- scripts/check_no_secrets.py   refuses to commit a leaked key
        |
        v
docs/data/*.json          committed to the repo
        |
        v
docs/index.html           GitHub Pages serves this. No server, no keys in the browser.
```

Your keys live in **Settings → Secrets and variables → Actions**. They are visible only to the workflow. The published page contains no credentials — only numbers and the provenance of those numbers.

**The watchlist is the only file you normally edit.** Change `config/watchlist.json`, commit, and the push triggers a rebuild.

A local Streamlit version (`app.py`) is still in the repo if you ever want to type an arbitrary ticker, but it is optional and nothing depends on it.

---

## The four design decisions that matter

### 1. Failure is loud

When every provider fails, you get this:

```
DataUnavailable: No provider could satisfy: ohlcv:MU:2015-01-01:2026-08-07
  [FAIL] twelvedata (312ms): RuntimeError: API credits exhausted
  [FAIL] polygon (89ms): RateLimited: polygon HTTP 429
  [FAIL] fmp (204ms): ContractViolation: MU: 41 NaN closes
  [FAIL] finnhub (0ms): no API key configured
```

Not an empty chart. There is no code path that returns partial, forward-filled, or synthesised prices.

### 2. Contract violations are provider failures

`mi/contracts.py` rejects frames with NaN closes, non-monotonic indices, constant prices, `high < low`, and unadjusted splits (a >45% single-day gap with no volume spike). A rejected frame moves the router to the next provider. It never reaches an indicator.

Holdings tables are checked to sum to ~1.0. Providers frequently return only the top 10 holdings; weights then sum to 0.4 and every overlap number computed from them is wrong by a factor of two, with nothing to indicate it.

### 3. Nothing is lookahead-contaminated

`tests/test_no_lookahead.py` asserts that indicator values computed on data through day 500 are bit-identical to the same values computed on the full 600-day series. A lookahead bug does not crash — it produces a beautiful backtest. This test is the only thing that catches it.

Swing points are shifted by their confirmation lag, so structure signals appear on the bar they became *known*, not the bar they occurred. The walk-forward splitter purges the label horizon and applies an embargo, because a 5-day forward label makes adjacent rows share outcome information.

### 4. The model must earn the right to speak

`DirectionalModel` runs expanding-window walk-forward validation, then checks four gates:

- Brier **skill** (not accuracy) beats the base rate known at train time
- Skill is positive in ≥60% of folds — not one lucky regime
- Block permutation test p < 0.05 — skill is distinguishable from noise
- ≥300 out-of-sample rows

Fail any one and `predict_proba()` **raises** instead of returning a probability:

```
ModelNotQualified: model failed its out-of-sample gate and will not emit probabilities:
  - Brier skill -0.0405 < 0.005 (no better than predicting the base rate)
  - skill positive in only 0/6 folds (unstable across time, likely one lucky regime)
  - permutation p=0.972 > 0.05 (skill indistinguishable from noise)
```

**Expect most tickers to fail this gate.** That is the system working. Verified behaviour: on synthetic random walks the model refuses across every seed tested; on a synthetic series with an injected momentum edge it qualifies with Brier skill +0.023, p=0.0005, and positive skill in 5 of 6 folds. The gate is strict but it is passable.

Note on the benchmark: fold skill is measured against the base rate **knowable before the fold started**, not the fold's own realised base rate. Grading against the latter benchmarks the model against a reference that saw the future, and on any series whose base rate drifts it makes a genuinely skillful model look useless.

---

## The macro module

FRED-sourced, because there is no second source for TIPS real yields and pretending otherwise would be dishonest. Each series declares its own release cadence and staleness tolerance — CPI at 40 days old is normal, the broad dollar index at 40 days old is broken. A global freshness threshold cannot tell those apart.

The regime classifier is **rule-based, not an HMM**. An HMM gives you "state 3" and then you rationalise what state 3 means, and the states are unstable across refits. These rules are wrong in a way you can inspect and argue with, which is more useful.

`framework_status()` reports the rolling sensitivity of gold to real-yield *changes*. When that beta drifts toward zero, the real-yield framework has temporarily stopped being the driver — and the dashboard says so rather than continuing to reason as if it were.

---

## Honest limitations

Stated plainly, because the alternative is you finding out later.

**No semiconductor cycle module.** DRAM/NAND spot pricing, HBM demand, wafer utilization, and memory inventory are TrendForce and DRAMeXchange products. There is no free API. Any module claiming to automate this is either scraping something fragile or making it up. Doing it properly means manual quarterly entry from 10-Qs.

**No dark pool or options flow.** Requires paid tier data (Unusual Whales, CBOE). Free option chains are delayed and patchy enough that any signal built on them is measuring the data vendor.

**Thematic ETF holdings are unreliable.** Broad funds (SPY, QQQ, SMH, SOXX, XLK, GDX) are well covered. Low-AUM thematic funds frequently are not — `coverage_note()` warns before you rely on them, and failures are reported per fund rather than producing a silently incomplete overlap matrix. The real fix for these is the issuer's own holdings CSV, not another API key.

**No deep learning.** LSTMs and Transformers on ~2000 daily bars of a single ticker memorise. Out of sample they do not beat a drift baseline. If the gradient-boosted model finds nothing, a neural net will only find something that isn't there.

**The site shows only what the build precomputed.** Pages serves static files, so you cannot type an arbitrary ticker and have it fetch. Everything comes from `config/watchlist.json`. Adding a ticker means editing that file and waiting for the next run — which is the trade you make for never running Python locally.

**The repo grows a little every day.** Each build commits roughly 200 KB of JSON. That is about 50 MB a year of git history. Reduce `chart_points` in the watchlist if that ever bothers you.

**If the repo is public, your watchlist is public.** The tickers you follow, and the ETFs you hold, are visible to anyone with the URL. Keys are not — they stay in Actions secrets, and `check_no_secrets.py` refuses to commit a build containing one. Make the repo private if the watchlist itself is sensitive; Pages on a private repo requires a paid plan.

**Nothing here is a trading system.** There is no position sizing, no risk management, no execution. A qualified model gives you a calibrated probability, which is an input to a decision, not a decision.

---

## Adding a provider

Subclass `Provider`, declare `capabilities`, reshape the vendor payload into the contract's column names, and add it to `ALL_PROVIDERS`. Do not catch your own errors and do not clean the data — the router handles fallback and `mi/contracts.py` decides what is usable. An adapter that quietly repairs a bad response defeats the entire design.

---

## Test suite

43 tests, all offline. They gate the scheduled build: if they fail, no data is fetched and yesterday's good data is left untouched.

```
tests/test_contracts.py      contract validation, including unadjusted-split detection
tests/test_router.py         fallback order, loud failure, provenance completeness
tests/test_no_lookahead.py   causality of indicators, features, and splits
tests/test_model_gate.py     refuses random walks, passes a real injected edge
tests/test_etf_and_macro.py  overlap maths, look-through risk, regime classification
tests/test_redaction.py      API keys never reach a committed file
```
