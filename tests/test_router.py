"""Fallback behaviour, and the guarantee that failure is never silent."""
import pandas as pd
import pytest

from mi.errors import DataUnavailable
from mi.providers.base import Provider, RateLimited
from mi.providers.router import DataRouter
from tests.conftest import make_ohlcv


class Down(Provider):
    name = "down"; env_key = None; capabilities = {"ohlcv"}
    def daily_ohlcv(self, s, a, b): raise RuntimeError("502")


class Limited(Provider):
    name = "limited"; env_key = None; capabilities = {"ohlcv"}
    def daily_ohlcv(self, s, a, b): raise RateLimited("429")


class Truncating(Provider):
    name = "trunc"; env_key = None; capabilities = {"ohlcv"}
    def daily_ohlcv(self, s, a, b): return make_ohlcv(n=15)


class Good(Provider):
    name = "good"; env_key = None; capabilities = {"ohlcv"}
    def daily_ohlcv(self, s, a, b): return make_ohlcv()


class NeedsKey(Provider):
    name = "needskey"; env_key = "DEFINITELY_NOT_SET_12345"; capabilities = {"ohlcv"}
    def daily_ohlcv(self, s, a, b): return make_ohlcv()


def test_falls_through_to_working_provider():
    r = DataRouter([Down(), Limited(), Truncating(), Good()], use_cache=False)
    p = r.ohlcv("T", "2021-01-01", "2024-01-01")
    assert p.sources()["close"] == "good"


def test_contract_failure_counts_as_provider_failure():
    r = DataRouter([Truncating(), Good()], use_cache=False)
    r.ohlcv("T", "2021-01-01", "2024-01-01")
    tbl = r.attempt_table()
    assert not tbl[tbl.provider == "trunc"]["ok"].iloc[0]


def test_raises_rather_than_returning_partial():
    r = DataRouter([Down(), Truncating()], use_cache=False)
    with pytest.raises(DataUnavailable) as e:
        r.ohlcv("T", "2021-01-01", "2024-01-01")
    assert "502" in str(e.value) and "rows" in str(e.value)


def test_unconfigured_provider_is_reported_not_hidden():
    r = DataRouter([NeedsKey(), Good()], use_cache=False)
    r.ohlcv("T", "2021-01-01", "2024-01-01")
    tbl = r.attempt_table()
    assert "no API key configured" in " ".join(tbl["detail"])


def test_every_column_has_provenance():
    r = DataRouter([Good()], use_cache=False)
    p = r.ohlcv("T", "2021-01-01", "2024-01-01")
    assert set(p.provenance) == set(p.data.columns)
    assert not p.audit().empty
