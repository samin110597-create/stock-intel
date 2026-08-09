import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n=800, seed=0, start="2021-01-01"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.013, n))), index=idx)
    return pd.DataFrame(
        {
            "open": c.shift(1).bfill(),
            "high": c * (1 + abs(rng.normal(0, 0.007, n))),
            "low": c * (1 - abs(rng.normal(0, 0.007, n))),
            "close": c,
            "volume": rng.lognormal(15, 0.4, n),
        },
        index=idx,
    )


@pytest.fixture
def ohlcv():
    return make_ohlcv()
