"""Overlap and concentration maths.

The number most people quote — "these two ETFs share 40 holdings" — is close
to meaningless. What matters is shared *weight*, because 40 shared names at
0.1% each is not overlap in any sense that affects your portfolio.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def weight_overlap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Sum of min(weight) across the union. This is the fraction of capital
    that is genuinely duplicated if you hold both at equal dollar size."""
    m = a.set_index("symbol")["weight"].add(0).to_frame("a").join(
        b.set_index("symbol")["weight"].to_frame("b"), how="outer"
    ).fillna(0.0)
    return float(np.minimum(m["a"], m["b"]).sum())


def name_jaccard(a: pd.DataFrame, b: pd.DataFrame) -> float:
    sa, sb = set(a["symbol"]), set(b["symbol"])
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def hhi(df: pd.DataFrame) -> float:
    """Herfindahl index of weights. 1/HHI is the effective number of holdings."""
    return float((df["weight"] ** 2).sum())


def effective_n(df: pd.DataFrame) -> float:
    h = hhi(df)
    return float(1 / h) if h > 0 else float("nan")


def top_n_weight(df: pd.DataFrame, n: int = 10) -> float:
    return float(df.nlargest(n, "weight")["weight"].sum())


def concentration_table(holdings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for etf, df in holdings.items():
        rows.append(
            {
                "etf": etf,
                "holdings": len(df),
                "top10_weight": round(top_n_weight(df, 10), 4),
                "hhi": round(hhi(df), 4),
                "effective_n": round(effective_n(df), 1),
                "largest": df.iloc[0]["symbol"] if len(df) else None,
                "largest_weight": round(float(df.iloc[0]["weight"]), 4) if len(df) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("effective_n")


def overlap_matrix(holdings: dict[str, pd.DataFrame], metric: str = "weight") -> pd.DataFrame:
    keys = list(holdings)
    fn = weight_overlap if metric == "weight" else name_jaccard
    m = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)
    for a, b in itertools.combinations(keys, 2):
        v = fn(holdings[a], holdings[b])
        m.loc[a, b] = m.loc[b, a] = v
    return m.round(4)


def shared_exposure(holdings: dict[str, pd.DataFrame], top: int = 25) -> pd.DataFrame:
    """Look-through: what do you actually own across the whole ETF sleeve?

    Assumes equal dollar allocation to each ETF. This is the table that shows
    a 'diversified' four-ETF sleeve is 22% one stock.
    """
    n = len(holdings)
    acc: dict[str, float] = {}
    appear: dict[str, int] = {}
    for df in holdings.values():
        for sym, w in zip(df["symbol"], df["weight"]):
            acc[sym] = acc.get(sym, 0.0) + w / n
            appear[sym] = appear.get(sym, 0) + 1
    out = pd.DataFrame(
        {"symbol": list(acc), "effective_weight": list(acc.values())}
    )
    out["in_n_etfs"] = out["symbol"].map(appear)
    return out.sort_values("effective_weight", ascending=False).head(top).reset_index(drop=True)
