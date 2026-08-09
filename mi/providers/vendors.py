"""Concrete vendor adapters.

Each one does exactly two things: call the endpoint, and reshape the response
into the contract's column names. Any judgement about whether the result is
usable belongs in mi.contracts, not here.
"""

from __future__ import annotations

import pandas as pd

from .base import Provider


class TwelveData(Provider):
    name = "twelvedata"
    env_key = "TWELVEDATA_KEY"
    capabilities = {"ohlcv"}
    calls_per_minute = 8  # free tier

    BASE = "https://api.twelvedata.com"

    def daily_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        payload = self._get(
            f"{self.BASE}/time_series",
            {
                "symbol": symbol,
                "interval": "1day",
                "start_date": start,
                "end_date": end,
                "outputsize": 5000,
                "apikey": self.api_key,
            },
        )
        values = payload.get("values") if isinstance(payload, dict) else None
        if not values:
            raise RuntimeError("no 'values' in response")
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df["volume"] = df.get("volume", 0)
        return df[["open", "high", "low", "close", "volume"]]


class Polygon(Provider):
    name = "polygon"
    env_key = "POLYGON_KEY"
    capabilities = {"ohlcv"}
    calls_per_minute = 5

    BASE = "https://api.polygon.io"

    def daily_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        payload = self._get(
            f"{self.BASE}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
            {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key},
        )
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            raise RuntimeError("no 'results' in response")
        df = pd.DataFrame(results)
        df.index = pd.to_datetime(df["t"], unit="ms")
        return df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )[["open", "high", "low", "close", "volume"]]


class FMP(Provider):
    name = "fmp"
    env_key = "FMP_KEY"
    capabilities = {"ohlcv", "fundamentals", "etf_holdings"}
    calls_per_minute = 30

    BASE = "https://financialmodelingprep.com/api/v3"

    def daily_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        payload = self._get(
            f"{self.BASE}/historical-price-full/{symbol}",
            {"from": start, "to": end, "apikey": self.api_key},
        )
        hist = payload.get("historical") if isinstance(payload, dict) else None
        if not hist:
            raise RuntimeError("no 'historical' in response")
        df = pd.DataFrame(hist)
        df.index = pd.to_datetime(df["date"])
        return df[["open", "high", "low", "close", "volume"]]

    def fundamentals(self, symbol: str) -> dict:
        km = self._get(f"{self.BASE}/key-metrics-ttm/{symbol}", {"apikey": self.api_key})
        profile = self._get(f"{self.BASE}/profile/{symbol}", {"apikey": self.api_key})
        if not km or not profile:
            raise RuntimeError("empty fundamentals payload")
        out = dict(km[0])
        out.update({k: profile[0].get(k) for k in ("companyName", "sector", "industry", "beta", "mktCap")})
        return out

    def etf_holdings(self, symbol: str) -> pd.DataFrame:
        payload = self._get(f"{self.BASE}/etf-holder/{symbol}", {"apikey": self.api_key})
        if not payload:
            raise RuntimeError("empty holdings payload")
        df = pd.DataFrame(payload)
        cols = {}
        for src, dst in (("asset", "symbol"), ("name", "name"), ("weightPercentage", "weight")):
            if src in df.columns:
                cols[src] = dst
        df = df.rename(columns=cols)
        if "symbol" not in df.columns or "weight" not in df.columns:
            raise RuntimeError(f"unexpected holdings schema: {list(df.columns)[:8]}")
        keep = ["symbol", "weight"] + (["name"] if "name" in df.columns else [])
        return df[keep]


class Finnhub(Provider):
    name = "finnhub"
    env_key = "FINNHUB_KEY"
    capabilities = {"ohlcv", "fundamentals"}
    calls_per_minute = 30

    BASE = "https://finnhub.io/api/v1"

    def daily_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        s = int(pd.Timestamp(start).timestamp())
        e = int(pd.Timestamp(end).timestamp())
        payload = self._get(
            f"{self.BASE}/stock/candle",
            {"symbol": symbol, "resolution": "D", "from": s, "to": e, "token": self.api_key},
        )
        if not isinstance(payload, dict) or payload.get("s") != "ok":
            raise RuntimeError(f"status={payload.get('s') if isinstance(payload, dict) else '?'}")
        df = pd.DataFrame(
            {
                "open": payload["o"],
                "high": payload["h"],
                "low": payload["l"],
                "close": payload["c"],
                "volume": payload["v"],
            },
            index=pd.to_datetime(payload["t"], unit="s"),
        )
        return df

    def fundamentals(self, symbol: str) -> dict:
        payload = self._get(
            f"{self.BASE}/stock/metric", {"symbol": symbol, "metric": "all", "token": self.api_key}
        )
        metric = payload.get("metric") if isinstance(payload, dict) else None
        if not metric:
            raise RuntimeError("no 'metric' in response")
        return dict(metric)


class AlphaVantage(Provider):
    name = "alphavantage"
    env_key = "ALPHAVANTAGE_KEY"
    capabilities = {"ohlcv"}
    calls_per_minute = 5

    BASE = "https://www.alphavantage.co/query"

    def daily_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        payload = self._get(
            self.BASE,
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.api_key,
            },
        )
        series = payload.get("Time Series (Daily)") if isinstance(payload, dict) else None
        if not series:
            raise RuntimeError("no daily time series in response")
        df = pd.DataFrame(series).T
        df.index = pd.to_datetime(df.index)
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "5. adjusted close": "close",
                "6. volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df.loc[(df.index >= start) & (df.index <= end)]


class Fred(Provider):
    """FRED is free, has no meaningful rate limit, and is the only sane source
    for real yields, breakevens, CPI and M2. It has no fallback because there
    is no equivalent second source — if FRED is down, the macro tab says so."""

    name = "fred"
    env_key = "FRED_API_KEY"
    capabilities = {"macro"}
    calls_per_minute = 100

    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def daily_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("FRED does not serve OHLCV")

    def macro_series(self, series_id: str, start: str) -> pd.Series:
        payload = self._get(
            self.BASE,
            {
                "series_id": series_id,
                "observation_start": start,
                "file_type": "json",
                "api_key": self.api_key,
            },
        )
        obs = payload.get("observations") if isinstance(payload, dict) else None
        if not obs:
            raise RuntimeError(f"no observations for {series_id}")
        df = pd.DataFrame(obs)
        s = pd.Series(
            pd.to_numeric(df["value"].replace(".", None), errors="coerce").values,
            index=pd.to_datetime(df["date"]),
            name=series_id,
        )
        return s.dropna()


ALL_PROVIDERS = [TwelveData, Polygon, FMP, Finnhub, AlphaVantage, Fred]
