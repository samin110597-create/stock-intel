"""Validated daily history using existing keyed providers, then Yahoo Finance."""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from mi.contracts import validate_ohlcv
from mi.providers.base import Provider
from mi.providers.router import DataRouter
from mi.providers.vendors import ALL_PROVIDERS


def ticker(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-^]{0,14}", value):
        raise ValueError("Enter a US stock or ETF ticker, such as NVDA or BRK-B.")
    return value


def validated(frame: pd.DataFrame, symbol: str, min_rows: int = 220) -> pd.DataFrame:
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Daily history contains duplicate or unordered dates.")
    out = validate_ohlcv(frame, symbol, min_rows=min_rows)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Daily history contains missing or infinite OHLCV values.")
    if (out[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    if ((out.open > out.high * 1.000001) | (out.open < out.low * .999999)).any():
        raise ValueError("Open falls outside the daily high/low.")
    if (out.volume <= 0).any():
        raise ValueError("Daily history must contain positive volume.")
    return out


class YahooDaily(Provider):
    name = "yahoo_finance"
    capabilities = {"ohlcv"}

    def daily_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        # Split-adjusted OHLC, without dividend adjustment. Avoid mixing raw
        # highs/lows with dividend-adjusted close. This is a price-return test.
        frame = yf.Ticker(symbol).history(
            start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=False, actions=False, raise_errors=True, timeout=self.timeout,
        )
        frame.columns = frame.columns.str.lower()
        frame.index = frame.index.tz_localize(None).normalize()
        return frame[['open', 'high', 'low', 'close', 'volume']]


def load_history(symbol: str, start: str = "2018-01-01", end: str | None = None):
    symbol = ticker(symbol)
    # Conservatively exclude today's bar, even after the close, so a quote
    # entered during the session never becomes a fabricated daily OHLC bar.
    cutoff = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date()) - pd.Timedelta(days=1)
    if end:
        cutoff = min(cutoff, pd.Timestamp(end))
    router = DataRouter([cls() for cls in ALL_PROVIDERS] + [YahooDaily()])
    def fetch(provider):
        raw = provider.daily_ohlcv(symbol, start, cutoff.date().isoformat())
        raw = raw.sort_index()
        return raw.loc[pd.Timestamp(start):cutoff]
    # Distinct cache namespace: the legacy cache has a weaker validation contract.
    request = f"analyzer-v1:{symbol}:{start}:{cutoff.date()}"
    result = router._resolve(request, "ohlcv", fetch,
                             lambda raw, _: validated(raw, symbol), 4)
    from mi import cache
    cache.write(result)
    result.data = validated(result.data, symbol)
    return result
