"""Macro regime classification.

Deliberately rule-based, not an HMM. An HMM on macro data gives you states
that are unstable and unlabelled — you get "state 3" and then you rationalise
what state 3 means. These rules are wrong in a way you can inspect and argue
with, which is more useful than being wrong in a way you cannot.

Four axes, each scored -1 / 0 / +1, then mapped to a named regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RegimeCall:
    regime: str
    confidence: str
    axes: dict[str, int]
    detail: dict[str, float]
    missing: list[str]

    def as_dict(self) -> dict:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            **{f"axis_{k}": v for k, v in self.axes.items()},
            "missing_inputs": ", ".join(self.missing) or "none",
        }


def _slope(s: pd.Series, n: int) -> float | None:
    s = s.dropna()
    if len(s) < n + 1:
        return None
    return float(s.iloc[-1] - s.iloc[-1 - n])


def _pct_change(s: pd.Series, n: int) -> float | None:
    s = s.dropna()
    if len(s) < n + 1:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1)


def classify(macro: pd.DataFrame, lookback_d: int = 60) -> RegimeCall:
    axes: dict[str, int] = {}
    detail: dict[str, float] = {}
    missing: list[str] = []

    # Axis 1: real rate direction. Rising real rates tighten everything.
    ry = _slope(macro.get("real_yield_10y", pd.Series(dtype=float)), lookback_d)
    if ry is None:
        missing.append("real_yield_10y")
    else:
        detail["real_yield_chg_60d_bp"] = round(ry * 100, 1)
        axes["real_rates"] = 1 if ry > 0.15 else (-1 if ry < -0.15 else 0)

    # Axis 2: inflation expectations.
    be = _slope(macro.get("breakeven_10y", pd.Series(dtype=float)), lookback_d)
    if be is None:
        missing.append("breakeven_10y")
    else:
        detail["breakeven_chg_60d_bp"] = round(be * 100, 1)
        axes["inflation_exp"] = 1 if be > 0.10 else (-1 if be < -0.10 else 0)

    # Axis 3: dollar. A strong dollar is a global liquidity drain.
    dxy = _pct_change(macro.get("dxy_broad", pd.Series(dtype=float)), lookback_d)
    if dxy is None:
        missing.append("dxy_broad")
    else:
        detail["dxy_chg_60d_pct"] = round(dxy * 100, 2)
        axes["dollar"] = 1 if dxy > 0.015 else (-1 if dxy < -0.015 else 0)

    # Axis 4: credit / growth stress.
    hy = _slope(macro.get("hy_spread", pd.Series(dtype=float)), lookback_d)
    curve = macro.get("yield_curve", pd.Series(dtype=float)).dropna()
    if hy is None:
        missing.append("hy_spread")
    else:
        detail["hy_oas_chg_60d_bp"] = round(hy * 100, 1)
        stress = 1 if hy > 0.40 else (-1 if hy < -0.30 else 0)
        if len(curve) and float(curve.iloc[-1]) < 0:
            detail["curve_10y2y"] = round(float(curve.iloc[-1]), 2)
            stress = max(stress, 0)
        axes["credit_stress"] = stress

    regime = _map(axes)
    n = len(axes)
    confidence = "high" if n == 4 else ("medium" if n == 3 else "low")
    return RegimeCall(regime, confidence, axes, detail, missing)


def _map(a: dict[str, int]) -> str:
    rr = a.get("real_rates", 0)
    ie = a.get("inflation_exp", 0)
    dx = a.get("dollar", 0)
    cs = a.get("credit_stress", 0)

    if cs > 0 and dx > 0:
        return "recessionary / risk-off"
    if rr > 0 and ie <= 0:
        return "disinflationary tightening"
    if rr > 0 and ie > 0:
        return "inflationary tightening"
    if rr < 0 and ie > 0:
        return "inflationary easing"
    if rr < 0 and dx < 0:
        return "liquidity expansion"
    if rr < 0:
        return "easing / mixed"
    return "neutral / transitional"


GOLD_PLAYBOOK = {
    "liquidity expansion": "Most supportive. Falling real rates plus a weakening dollar is gold's best combination.",
    "inflationary easing": "Supportive. Negative real rates are the driver; watch for breakevens rolling over.",
    "inflationary tightening": "Mixed. Breakevens help, rising real rates hurt. Historically the real rate wins.",
    "disinflationary tightening": "Hostile. This is the regime that produces multi-quarter gold drawdowns.",
    "recessionary / risk-off": "Two-stage. Initial liquidation with everything else, then a strong bid once policy responds.",
    "easing / mixed": "Mildly supportive but low conviction.",
    "neutral / transitional": "No macro edge. Trade structure, not the thesis.",
}
