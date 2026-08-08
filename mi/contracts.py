"""Schema contracts.

A provider response is not trusted until it passes these. Failing a contract
counts as a provider failure, which means the router moves to the next
provider rather than handing you a broken frame.
"""

from __future__ import annotations

import pandas as pd

from .errors import ContractViolation

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame, symbol: str, min_rows: int = 60) -> pd.DataFrame:
    """Return a normalised OHLCV frame or raise ContractViolation."""
    if df is None or len(df) == 0:
        raise ContractViolation(f"{symbol}: empty frame")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ContractViolation(f"{symbol}: missing columns {missing}")

    out = df[OHLCV_COLUMNS].copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ContractViolation(f"{symbol}: index is {type(out.index).__name__}, need DatetimeIndex")

    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    for c in OHLCV_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if len(out) < min_rows:
        raise ContractViolation(f"{symbol}: {len(out)} rows, need >= {min_rows}")

    if out["close"].isna().any():
        n = int(out["close"].isna().sum())
        raise ContractViolation(f"{symbol}: {n} NaN closes")

    if (out["close"] <= 0).any():
        raise ContractViolation(f"{symbol}: non-positive close prices")

    bad_hl = out["high"] < out["low"]
    if bad_hl.any():
        raise ContractViolation(f"{symbol}: high < low on {int(bad_hl.sum())} bars")

    tol = 1e-6
    outside = (out["close"] > out["high"] * (1 + tol)) | (out["close"] < out["low"] * (1 - tol))
    if outside.any():
        raise ContractViolation(f"{symbol}: close outside high/low on {int(outside.sum())} bars")

    if (out["volume"].fillna(0) < 0).any():
        raise ContractViolation(f"{symbol}: negative volume")

    # A flat series usually means the provider returned a placeholder.
    if out["close"].nunique() == 1:
        raise ContractViolation(f"{symbol}: close is constant across {len(out)} bars")

    # Detect unadjusted-split contamination: a >45% single-day gap with no
    # corresponding volume spike is almost always a data error, not a move.
    ret = out["close"].pct_change()
    volr = out["volume"] / out["volume"].rolling(20).median()
    suspicious = (ret.abs() > 0.45) & (volr < 1.5)
    if suspicious.any():
        dates = out.index[suspicious.fillna(False)].strftime("%Y-%m-%d").tolist()
        raise ContractViolation(
            f"{symbol}: probable unadjusted split/bad tick on {dates[:3]}"
        )

    return out


def validate_series(s: pd.Series, name: str, min_rows: int = 12) -> pd.Series:
    """Contract for a single macro time series."""
    if s is None or len(s) == 0:
        raise ContractViolation(f"{name}: empty series")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ContractViolation(f"{name}: index is not DatetimeIndex")
    out = pd.to_numeric(s, errors="coerce").dropna()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if len(out) < min_rows:
        raise ContractViolation(f"{name}: {len(out)} observations, need >= {min_rows}")
    return out.rename(name)


def validate_holdings(df: pd.DataFrame, etf: str, tolerance: float = 0.15) -> pd.DataFrame:
    """Contract for an ETF holdings table.

    Weights must be a fraction (not a percent) and must sum to roughly 1.
    A holdings table summing to 0.30 means the provider truncated the list,
    and every overlap number computed from it would be silently wrong.
    """
    if df is None or len(df) == 0:
        raise ContractViolation(f"{etf}: empty holdings")
    for c in ("symbol", "weight"):
        if c not in df.columns:
            raise ContractViolation(f"{etf}: holdings missing '{c}'")

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["weight"])
    out = out[out["symbol"].str.len() > 0]

    if out["weight"].max() > 1.5:  # provider gave percents
        out["weight"] = out["weight"] / 100.0

    out = out.groupby("symbol", as_index=False).agg(
        {"weight": "sum", **({"name": "first"} if "name" in out.columns else {})}
    )

    total = float(out["weight"].sum())
    if abs(total - 1.0) > tolerance:
        raise ContractViolation(
            f"{etf}: holdings weights sum to {total:.3f}, expected 1.0 "
            f"(+/-{tolerance}). Provider likely truncated the list."
        )
    if len(out) < 5:
        raise ContractViolation(f"{etf}: only {len(out)} holdings returned")

    return out.sort_values("weight", ascending=False).reset_index(drop=True)
