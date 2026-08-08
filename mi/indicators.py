"""Indicators. Pure functions on price series, no I/O, no state.

All of these are causal: a value at time t uses only data up to t. That is
enforced by convention here and checked by tests/test_no_lookahead.py, which
matters more than it sounds — a single centered rolling window silently turns
a backtest into a fortune teller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return (sign * df["volume"]).cumsum()


def anchored_vwap(df: pd.DataFrame, anchor: str | pd.Timestamp) -> pd.Series:
    d = df.loc[pd.Timestamp(anchor):]
    tp = (d["high"] + d["low"] + d["close"]) / 3
    return (tp * d["volume"]).cumsum() / d["volume"].cumsum().replace(0, np.nan)


def realized_vol(s: pd.Series, n: int = 20) -> pd.Series:
    return s.pct_change().rolling(n, min_periods=n).std() * np.sqrt(252)


def zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std()
    return (s - m) / sd.replace(0, np.nan)


# -- market structure -------------------------------------------------------
def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Confirmed swing highs/lows.

    A swing is only confirmed `right` bars after it prints. The returned
    frame is shifted so that a swing appears on the bar where it became
    KNOWN, not on the bar where it happened. Without that shift, every
    structure signal downstream is lookahead-contaminated.
    """
    h, l = df["high"], df["low"]
    is_high = (h == h.rolling(left + right + 1, center=True).max()) & h.notna()
    is_low = (l == l.rolling(left + right + 1, center=True).min()) & l.notna()
    out = pd.DataFrame(
        {
            "swing_high": h.where(is_high),
            "swing_low": l.where(is_low),
        }
    )
    return out.shift(right)  # known only after confirmation


def structure(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Break of Structure / Change of Character on confirmed swings.

    BOS  = close through the most recent confirmed swing in the trend direction.
    CHoCH= close through the most recent confirmed swing against the trend.
    """
    sw = swing_points(df, left, right)
    last_high = sw["swing_high"].ffill()
    last_low = sw["swing_low"].ffill()
    c = df["close"]

    up_break = c > last_high
    dn_break = c < last_low

    trend = pd.Series(0, index=df.index, dtype=int)
    state = 0
    bos, choch = [], []
    for i in range(len(df)):
        u, d = bool(up_break.iloc[i]), bool(dn_break.iloc[i])
        b = ch = False
        if u and state >= 0:
            b, state = True, 1
        elif u and state < 0:
            ch, state = True, 1
        elif d and state <= 0:
            b, state = True, -1
        elif d and state > 0:
            ch, state = True, -1
        trend.iloc[i] = state
        bos.append(b)
        choch.append(ch)

    return pd.DataFrame(
        {
            "trend": trend,
            "bos": pd.Series(bos, index=df.index),
            "choch": pd.Series(choch, index=df.index),
            "last_swing_high": last_high,
            "last_swing_low": last_low,
        }
    )


def fair_value_gaps(df: pd.DataFrame, min_atr_frac: float = 0.25) -> pd.DataFrame:
    """Three-bar imbalances, filtered by size relative to ATR.

    Unfiltered FVGs fire on every other bar and are noise. The ATR filter is
    what makes them worth looking at.
    """
    a = atr(df, 14)
    up_gap = df["low"] - df["high"].shift(2)
    dn_gap = df["low"].shift(2) - df["high"]
    thr = a * min_atr_frac
    return pd.DataFrame(
        {
            "fvg_up": up_gap.where(up_gap > thr),
            "fvg_down": dn_gap.where(dn_gap > thr),
        }
    )


def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Wick through a prior extreme that closes back inside it."""
    hh = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    ll = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    return pd.DataFrame(
        {
            "sweep_high": (df["high"] > hh) & (df["close"] < hh),
            "sweep_low": (df["low"] < ll) & (df["close"] > ll),
        }
    )


def indicator_pack(df: pd.DataFrame) -> pd.DataFrame:
    """Everything at once, aligned to the price index."""
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["ema21"] = ema(c, 21)
    out["ema50"] = ema(c, 50)
    out["sma200"] = sma(c, 200)
    out["rsi14"] = rsi(c, 14)
    out = out.join(macd(c))
    out["atr14"] = atr(df, 14)
    out["obv"] = obv(df)
    out["rvol20"] = df["volume"] / df["volume"].rolling(20, min_periods=20).median()
    out["vol20"] = realized_vol(c, 20)
    out["dist_ema21"] = c / out["ema21"] - 1
    out["dist_sma200"] = c / out["sma200"] - 1
    out = out.join(structure(df)).join(fair_value_gaps(df)).join(liquidity_sweep(df))
    return out
