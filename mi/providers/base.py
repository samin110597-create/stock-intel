"""Provider interface.

Adapters translate a vendor's JSON into the contract shape. They do not
retry across vendors, they do not fill gaps, and they do not catch their own
errors — the router owns all of that.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


class RateLimited(Exception):
    """Vendor said slow down. The router treats this differently from a hard
    failure: the provider is not broken, it is busy."""


class Provider(ABC):
    name: str = "base"
    capabilities: set[str] = set()
    env_key: str | None = None
    calls_per_minute: int = 60

    def __init__(self, api_key: str | None = None, timeout: int = 20):
        self.api_key = api_key or (os.getenv(self.env_key) if self.env_key else None)
        self.timeout = timeout
        self._call_times: list[float] = []

    # -- plumbing -------------------------------------------------------
    @property
    def configured(self) -> bool:
        return self.env_key is None or bool(self.api_key)

    def _throttle(self) -> None:
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        if len(self._call_times) >= self.calls_per_minute:
            sleep_for = 60 - (now - self._call_times[0]) + 0.25
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._call_times.append(time.time())

    def _get(self, url: str, params: dict | None = None) -> dict | list:
        if requests is None:  # pragma: no cover
            raise RuntimeError("requests is not installed")
        self._throttle()
        r = requests.get(url, params=params or {}, timeout=self.timeout)
        if r.status_code in (429, 503):
            raise RateLimited(f"{self.name} HTTP {r.status_code}")
        r.raise_for_status()
        payload = r.json()
        self._check_payload_error(payload)
        return payload

    def _check_payload_error(self, payload) -> None:
        """Vendors love returning HTTP 200 with an error body. Catch that here
        so it never reaches the contract validator as an 'empty frame'."""
        if isinstance(payload, dict):
            for k in ("Error Message", "Note", "error", "message", "Information"):
                if k in payload and payload[k]:
                    raise RuntimeError(f"{self.name}: {payload[k]}")
            if payload.get("status") == "error":
                raise RuntimeError(f"{self.name}: {payload.get('message', 'error')}")

    # -- capability surface ---------------------------------------------
    @abstractmethod
    def daily_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        ...

    def fundamentals(self, symbol: str) -> dict:
        raise NotImplementedError

    def etf_holdings(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    def macro_series(self, series_id: str, start: str) -> pd.Series:
        raise NotImplementedError
