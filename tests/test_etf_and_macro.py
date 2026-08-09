import numpy as np
import pandas as pd

from mi.etf import concentration_table, effective_n, overlap_matrix, shared_exposure, weight_overlap
from mi.macro.regime import classify


def mk(d):
    return pd.DataFrame({"symbol": list(d), "weight": list(d.values())}).sort_values(
        "weight", ascending=False
    ).reset_index(drop=True)


def test_identical_funds_overlap_fully():
    a = mk({"A": 0.5, "B": 0.3, "C": 0.2})
    assert abs(weight_overlap(a, a) - 1.0) < 1e-9


def test_disjoint_funds_do_not_overlap():
    a = mk({"A": 0.6, "B": 0.4})
    b = mk({"C": 0.7, "D": 0.3})
    assert weight_overlap(a, b) == 0.0


def test_shared_names_at_tiny_weight_do_not_count_as_overlap():
    """The failure mode this whole module exists to prevent."""
    a = mk({"BIG": 0.9, "X": 0.05, "Y": 0.05})
    b = mk({"OTHER": 0.9, "X": 0.05, "Y": 0.05})
    assert weight_overlap(a, b) < 0.15


def test_effective_n_penalises_concentration():
    conc = mk({"A": 0.9, "B": 0.05, "C": 0.05})
    even = mk({str(i): 0.1 for i in range(10)})
    assert effective_n(conc) < 2 < effective_n(even)


def test_look_through_reveals_hidden_single_name_risk():
    h = {"E1": mk({"NVDA": 0.3, "A": 0.7}), "E2": mk({"NVDA": 0.25, "B": 0.75})}
    top = shared_exposure(h, top=3)
    assert top.iloc[0]["symbol"] == "NVDA" or top[top.symbol == "NVDA"]["in_n_etfs"].iloc[0] == 2


def test_overlap_matrix_is_symmetric():
    h = {"A": mk({"X": 0.6, "Y": 0.4}), "B": mk({"X": 0.3, "Z": 0.7})}
    m = overlap_matrix(h)
    assert np.allclose(m.to_numpy(), m.to_numpy().T)


def test_regime_reports_missing_inputs_instead_of_guessing():
    idx = pd.bdate_range("2024-01-01", periods=200)
    partial = pd.DataFrame({"real_yield_10y": np.linspace(1.0, 2.0, 200)}, index=idx)
    call = classify(partial)
    assert call.confidence == "low"
    assert "dxy_broad" in call.missing


def test_regime_flags_tightening():
    idx = pd.bdate_range("2024-01-01", periods=200)
    m = pd.DataFrame(
        {
            "real_yield_10y": np.linspace(1.0, 2.5, 200),
            "breakeven_10y": np.linspace(2.5, 2.0, 200),
            "dxy_broad": np.linspace(100, 108, 200),
            "hy_spread": np.linspace(3.0, 3.2, 200),
        },
        index=idx,
    )
    assert "tightening" in classify(m).regime
