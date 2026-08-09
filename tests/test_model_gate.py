"""The gate must refuse noise. This is the test that stops the repo from
becoming a confident random number generator."""
import warnings

import numpy as np
import pandas as pd
import pytest

from mi.errors import ModelNotQualified
from mi.ml import DirectionalModel, assemble
from mi.ml.calibration import block_permutation_pvalue, brier_skill, reliability_table

warnings.filterwarnings("ignore")


def synth(seed: int, k: float = 0.0, n: int = 2000) -> pd.DataFrame:
    """k=0 is a random walk. k>0 injects a genuine, causal momentum edge."""
    idx = pd.bdate_range("2016-01-01", periods=n)
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.012, n)
    lp = np.zeros(n)
    for i in range(n):
        mom = lp[i - 1] - lp[i - 22] if i > 22 else 0.0
        lp[i] = (lp[i - 1] if i else 0.0) + 0.0002 + k * np.tanh(mom * 4) + e[i]
    c = pd.Series(100 * np.exp(lp), index=idx)
    return pd.DataFrame(
        {
            "open": c.shift(1).bfill(),
            "high": c * (1 + abs(rng.normal(0, 0.006, n))),
            "low": c * (1 - abs(rng.normal(0, 0.006, n))),
            "close": c,
            "volume": rng.lognormal(15, 0.35, n),
        },
        index=idx,
    )


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_refuses_random_walk(seed):
    X, y = assemble(synth(seed))
    m = DirectionalModel().fit(X, y)
    assert not m.qualified_, "model claimed skill on a random walk"
    assert m.reasons_
    with pytest.raises(ModelNotQualified):
        m.predict_proba(X.tail(1))


def test_gate_is_passable_when_edge_is_real():
    """A gate that never passes is just a refusal with extra steps."""
    X, y = assemble(synth(21, k=0.008))
    m = DirectionalModel().fit(X, y)
    assert m.qualified_, f"gate rejected a real edge: {m.reasons_}"
    p = m.predict_proba(X.tail(5))
    assert p.between(0, 1).all()


def test_verdict_is_always_safe_to_call():
    m = DirectionalModel()
    assert m.verdict()["status"] == "not fitted"


def test_brier_skill_zero_for_base_rate_forecast():
    y = np.array([1.0] * 60 + [0.0] * 40)
    p = np.full(100, y.mean())
    assert abs(brier_skill(y, p)) < 1e-12


def test_permutation_test_finds_no_skill_in_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000).astype(float)
    p = rng.uniform(0.4, 0.6, 1000)
    assert block_permutation_pvalue(y, p, n_iter=500) > 0.05


def test_reliability_table_flags_overconfidence():
    y = np.zeros(400)
    p = np.full(400, 0.9)
    t = reliability_table(y, p)
    assert t["gap"].abs().max() > 0.5
