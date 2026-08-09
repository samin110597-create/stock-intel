"""Feature and label construction.

Two rules, both non-negotiable:
  1. Every feature at row t uses only information available at the close of t.
  2. The label at row t looks FORWARD h days, which means rows within h days
     of each other share information. That overlap is handled in the splitter,
     not here — but it is the reason the splitter needs an embargo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr, indicator_pack

# Round-trip cost assumption in basis points. The label asks "did the move
# clear costs", not "did price go up by a hair". A model trained on the
# latter looks accurate and loses money.
DEFAULT_COST_BPS = 15.0


def build_features(price: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    ind = indicator_pack(price)
    c = price["close"]
    a = atr(price, 14)

    X = pd.DataFrame(index=price.index)
    for n in (1, 5, 10, 21, 63):
        X[f"ret_{n}d"] = c.pct_change(n)
    X["rsi14"] = ind["rsi14"] / 100.0
    X["macd_hist_atr"] = ind["macd_hist"] / a          # scale-free across tickers
    X["dist_ema21"] = ind["dist_ema21"]
    X["dist_sma200"] = ind["dist_sma200"]
    X["ema_stack"] = np.sign(ind["ema21"] - ind["ema50"])
    X["rvol20"] = np.log1p(ind["rvol20"])
    X["vol20"] = ind["vol20"]
    X["vol_ratio"] = ind["vol20"] / ind["vol20"].rolling(120, min_periods=60).mean()
    X["obv_slope"] = ind["obv"].diff(20) / price["volume"].rolling(20).mean().replace(0, np.nan)
    X["trend"] = ind["trend"]
    X["bos"] = ind["bos"].astype(float)
    X["choch"] = ind["choch"].astype(float)
    X["sweep_low"] = ind["sweep_low"].astype(float)
    X["sweep_high"] = ind["sweep_high"].astype(float)
    X["atr_pct"] = a / c

    if macro is not None and len(macro):
        m = macro.reindex(price.index.union(macro.index)).sort_index().ffill().reindex(price.index)
        for col, lag in (("real_yield_10y", 20), ("dxy_broad", 20), ("hy_spread", 20),
                         ("breakeven_10y", 20), ("yield_curve", 20)):
            if col in m:
                X[f"{col}_chg{lag}"] = m[col].diff(lag)
        if "real_yield_10y" in m:
            X["real_yield_level"] = m["real_yield_10y"]

    return X.replace([np.inf, -np.inf], np.nan)


def build_label(
    price: pd.DataFrame, horizon: int = 5, cost_bps: float = DEFAULT_COST_BPS
) -> pd.Series:
    """1 if the forward `horizon`-day return clears round-trip costs."""
    fwd = price["close"].shift(-horizon) / price["close"] - 1.0
    return (fwd > cost_bps / 10_000.0).astype(float).where(fwd.notna()).rename("y")


def assemble(
    price: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    horizon: int = 5,
    cost_bps: float = DEFAULT_COST_BPS,
    min_rows: int = 400,
) -> tuple[pd.DataFrame, pd.Series]:
    X = build_features(price, macro)
    y = build_label(price, horizon, cost_bps)
    df = X.join(y).dropna()
    if len(df) < min_rows:
        raise ValueError(
            f"only {len(df)} usable rows after feature warm-up; need >= {min_rows}. "
            "Fetch more history rather than lowering this."
        )
    return df.drop(columns=["y"]), df["y"]
