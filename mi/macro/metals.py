"""Gold and silver against the macro drivers that actually move them.

The core claim this module encodes: over any horizon that matters, gold is
priced off the 10Y TIPS real yield and the dollar, and the residual is
positioning. Everything here measures those relationships rather than
asserting them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import zscore


def align(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Daily grid, forward-filled macro, no forward-filled prices.

    Macro is forward-filled because a monthly CPI print is genuinely the
    prevailing value until the next one. Prices are not, because a stale
    price is a hole.
    """
    grid = prices.index
    m = macro.reindex(grid.union(macro.index)).sort_index().ffill().reindex(grid)
    return prices.join(m)


def real_yield_beta(gold: pd.Series, real_yield: pd.Series, window: int = 120) -> pd.Series:
    """Rolling sensitivity of gold returns to CHANGES in the real yield.

    Expressed as % gold move per 100bp real-yield move. Historically this
    sits meaningfully negative; when it drifts toward zero or positive, the
    real-yield framework has temporarily stopped being the driver and you
    should not lean on it.
    """
    g = gold.pct_change()
    dy = real_yield.diff() / 100.0  # bp -> decimal
    cov = g.rolling(window, min_periods=window // 2).cov(dy)
    var = dy.rolling(window, min_periods=window // 2).var()
    return (cov / var.replace(0, np.nan)) / 100.0


def gold_silver_ratio(gold: pd.Series, silver: pd.Series, z_window: int = 250) -> pd.DataFrame:
    ratio = gold / silver
    return pd.DataFrame(
        {
            "gs_ratio": ratio,
            "gs_z": zscore(ratio, z_window),
            "gs_pctile_5y": ratio.rolling(1250, min_periods=250).rank(pct=True),
        }
    )


def in_cad(usd_series: pd.Series, usdcad: pd.Series) -> pd.Series:
    """USD-denominated metal expressed in CAD.

    For a Canadian holder this is the only price that matters. Gold can fall
    in USD and rise in CAD, which is exactly what happens when the dollar
    rallies for risk-off reasons.
    """
    return usd_series * usdcad.reindex(usd_series.index).ffill()


def drawdown(s: pd.Series) -> pd.Series:
    return s / s.cummax() - 1.0


def metals_panel(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """prices needs columns: gold, silver (close prices, any USD proxy)."""
    df = align(prices, macro)
    out = pd.DataFrame(index=df.index)
    out["gold"] = df["gold"]
    out["silver"] = df["silver"]

    if "real_yield_10y" in df:
        out["real_yield_10y"] = df["real_yield_10y"]
        out["ry_beta_120d"] = real_yield_beta(df["gold"], df["real_yield_10y"])
        out["ry_slope_60d"] = df["real_yield_10y"].diff(60)
    if "dxy_broad" in df:
        out["dxy"] = df["dxy_broad"]
        out["dxy_mom_60d"] = df["dxy_broad"].pct_change(60)
    if "breakeven_10y" in df:
        out["breakeven_10y"] = df["breakeven_10y"]
    if "usdcad" in df:
        out["gold_cad"] = in_cad(df["gold"], df["usdcad"])
        out["silver_cad"] = in_cad(df["silver"], df["usdcad"])

    out = out.join(gold_silver_ratio(df["gold"], df["silver"]))
    out["gold_dd"] = drawdown(df["gold"])
    out["silver_dd"] = drawdown(df["silver"])
    out["gold_z250"] = zscore(df["gold"], 250)
    return out


def framework_status(panel: pd.DataFrame) -> dict:
    """Is the real-yield framework currently working? Say so explicitly."""
    last = panel.dropna(subset=["ry_beta_120d"]).tail(1)
    if last.empty:
        return {"status": "unknown", "detail": "insufficient overlap of gold and real yield"}
    beta = float(last["ry_beta_120d"].iloc[0])
    if beta < -0.5:
        status, detail = "intact", "gold is trading inversely to real yields as expected"
    elif beta < 0:
        status, detail = "weak", "inverse relationship present but muted"
    else:
        status, detail = "broken", "gold is currently NOT trading off real yields; the framework is not the driver right now"
    return {"status": status, "beta_pct_per_100bp": round(beta * 100, 2), "detail": detail}
