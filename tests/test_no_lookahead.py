"""The most valuable tests in the repo.

A lookahead bug does not crash. It produces a beautiful backtest. The only
way to catch it is to assert that today's indicator value does not change
when tomorrow's data arrives.
"""
import numpy as np
import pandas as pd

from mi.indicators import indicator_pack, structure, swing_points
from mi.ml.features import build_features, build_label
from mi.ml.walkforward import walk_forward_splits
from tests.conftest import make_ohlcv

CAUSAL_COLS = [
    "ema21", "ema50", "sma200", "rsi14", "macd", "macd_signal", "macd_hist",
    "atr14", "rvol20", "vol20", "dist_ema21", "dist_sma200",
    "trend", "last_swing_high", "last_swing_low",
]


def test_indicators_are_causal():
    df = make_ohlcv(n=600, seed=5)
    cut = 500
    full = indicator_pack(df)
    trunc = indicator_pack(df.iloc[:cut])
    a = full[CAUSAL_COLS].iloc[:cut].tail(50)
    b = trunc[CAUSAL_COLS].tail(50)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_swings_only_confirmed_after_the_fact():
    df = make_ohlcv(n=400, seed=6)
    sw = swing_points(df, 3, 3)
    # a swing high recorded at row i must equal an actual high at or before i-3
    rows = sw["swing_high"].dropna()
    for ts, val in rows.tail(20).items():
        i = df.index.get_loc(ts)
        assert np.isclose(df["high"].iloc[max(0, i - 3)], val)


def test_features_are_causal():
    df = make_ohlcv(n=700, seed=7)
    cut = 600
    a = build_features(df).iloc[:cut].tail(30)
    b = build_features(df.iloc[:cut]).tail(30)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_label_looks_forward_and_last_rows_are_unknown():
    df = make_ohlcv(n=300, seed=8)
    y = build_label(df, horizon=5)
    assert y.tail(5).isna().all()
    assert y.iloc[:-5].notna().all()


def test_walkforward_purges_and_embargoes():
    idx = pd.bdate_range("2020-01-01", periods=2000)
    for f in walk_forward_splits(idx, n_folds=5, horizon=5):
        assert f.train.max() < f.test.min()
        gap = f.test.min() - f.train.max()
        assert gap > 5, f"gap {gap} does not cover the 5-day label horizon"
