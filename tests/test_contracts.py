"""The contract suite is the whole point of the data layer. If these pass,
a broken provider response cannot reach an indicator."""
import numpy as np
import pandas as pd
import pytest

from mi.contracts import validate_holdings, validate_ohlcv, validate_series
from mi.errors import ContractViolation
from tests.conftest import make_ohlcv


def test_accepts_clean_frame(ohlcv):
    assert len(validate_ohlcv(ohlcv, "T")) == len(ohlcv)


def test_rejects_empty():
    with pytest.raises(ContractViolation):
        validate_ohlcv(pd.DataFrame(), "T")


def test_rejects_nan_close(ohlcv):
    bad = ohlcv.copy()
    bad.iloc[300, bad.columns.get_loc("close")] = np.nan
    with pytest.raises(ContractViolation, match="NaN"):
        validate_ohlcv(bad, "T")


def test_rejects_too_few_rows(ohlcv):
    with pytest.raises(ContractViolation, match="rows"):
        validate_ohlcv(ohlcv.head(20), "T")


def test_rejects_constant_close(ohlcv):
    bad = ohlcv.copy()
    for c in ("open", "high", "low", "close"):
        bad[c] = 50.0
    with pytest.raises(ContractViolation, match="constant"):
        validate_ohlcv(bad, "T")


def test_rejects_high_below_low(ohlcv):
    bad = ohlcv.copy()
    bad.iloc[100, bad.columns.get_loc("high")] = 1.0
    with pytest.raises(ContractViolation):
        validate_ohlcv(bad, "T")


def test_rejects_unadjusted_split(ohlcv):
    """A 50% overnight drop with normal volume is a split the provider forgot
    to adjust. Silently accepting it corrupts every return in the series."""
    bad = ohlcv.copy()
    bad.iloc[400:, bad.columns.get_loc("close")] /= 2
    bad.iloc[400:, bad.columns.get_loc("high")] /= 2
    bad.iloc[400:, bad.columns.get_loc("low")] /= 2
    with pytest.raises(ContractViolation, match="split|bad tick"):
        validate_ohlcv(bad, "T")


def test_deduplicates_and_sorts(ohlcv):
    dup = pd.concat([ohlcv, ohlcv.tail(5)]).sample(frac=1, random_state=1)
    out = validate_ohlcv(dup, "T")
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates


def test_holdings_percent_normalisation():
    df = pd.DataFrame({"symbol": list("ABCDEF"), "weight": [30, 25, 15, 12, 10, 8]})
    out = validate_holdings(df, "X")
    assert abs(out["weight"].sum() - 1.0) < 1e-9


def test_holdings_rejects_truncated_list():
    """Providers often return only the top 10. Weights then sum to ~0.4 and
    every overlap number downstream is wrong by a factor of two."""
    df = pd.DataFrame({"symbol": list("ABCDEF"), "weight": [8, 7, 6, 5, 4, 3]})
    with pytest.raises(ContractViolation, match="sum"):
        validate_holdings(df, "X")


def test_series_contract():
    s = pd.Series([1.0, 2.0, 3.0] * 10, index=pd.bdate_range("2024-01-01", periods=30))
    assert len(validate_series(s, "m")) == 30
    with pytest.raises(ContractViolation):
        validate_series(s.head(3), "m")
