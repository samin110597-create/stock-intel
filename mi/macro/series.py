"""FRED series catalogue.

Each entry declares its release frequency and a staleness tolerance. CPI is
monthly with a ~2 week lag, so a 45-day tolerance is normal and a 90-day age
means the release pipeline broke. Encoding that here is what lets the Data
Health tab distinguish "monthly series, behaving normally" from "stale".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..errors import DataUnavailable
from ..provenance import Provenanced


@dataclass(frozen=True)
class SeriesSpec:
    fred_id: str
    label: str
    freq: str
    max_age_days: float
    note: str = ""


CATALOG: dict[str, SeriesSpec] = {
    "real_yield_10y": SeriesSpec("DFII10", "10Y TIPS real yield", "D", 6,
                                 "The master variable for gold. Watch the slope, not the level."),
    "nominal_10y": SeriesSpec("DGS10", "10Y Treasury", "D", 6),
    "breakeven_10y": SeriesSpec("T10YIE", "10Y breakeven inflation", "D", 6),
    "real_yield_5y": SeriesSpec("DFII5", "5Y TIPS real yield", "D", 6),
    "dxy_broad": SeriesSpec("DTWEXBGS", "Broad dollar index", "D", 12),
    "usdcad": SeriesSpec("DEXCAUS", "USD/CAD", "D", 12),
    "fed_funds": SeriesSpec("DFF", "Effective fed funds", "D", 6),
    "cpi": SeriesSpec("CPIAUCSL", "CPI all items SA", "M", 50),
    "core_cpi": SeriesSpec("CPILFESL", "Core CPI SA", "M", 50),
    "ppi": SeriesSpec("PPIACO", "PPI all commodities", "M", 55),
    "m2": SeriesSpec("M2SL", "M2 money stock", "M", 65),
    "unemployment": SeriesSpec("UNRATE", "Unemployment rate", "M", 45),
    "yield_curve": SeriesSpec("T10Y2Y", "10Y-2Y spread", "D", 6),
    "hy_spread": SeriesSpec("BAMLH0A0HYM2", "HY OAS", "D", 8),
    "fed_balance": SeriesSpec("WALCL", "Fed balance sheet", "W", 14),
}


def load_macro(router, keys: list[str] | None = None, start: str = "2005-01-01") -> Provenanced:
    """Fetch the catalogue. Partial success is allowed but recorded.

    Rationale: unlike price data, a missing single macro series should not
    kill the whole tab — but you must be able to see which ones are missing,
    because the regime classifier's answer changes depending on what it had.
    """
    keys = keys or list(CATALOG)
    frames: list[Provenanced] = []
    failures: dict[str, str] = {}

    for k in keys:
        spec = CATALOG[k]
        try:
            p = router.macro(spec.fred_id, start)
            p.data.columns = [k]
            p.provenance = {k: list(p.provenance.values())[0]}
            p.provenance[k] = type(p.provenance[k])(
                name=k,
                provider=p.provenance[k].provider,
                fetched_at=p.provenance[k].fetched_at,
                as_of=p.provenance[k].as_of,
                rows=p.provenance[k].rows,
                note=f"{spec.freq} | tol {spec.max_age_days}d",
            )
            frames.append(p)
        except DataUnavailable as e:
            failures[k] = str(e).splitlines()[0]

    if not frames:
        raise DataUnavailable("macro:catalog", [])

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    merged.request = "macro:catalog"
    if failures:
        merged.data.attrs["failures"] = failures
    return merged


def staleness_report(p: Provenanced) -> pd.DataFrame:
    """Per-series age against its own tolerance, not a global one."""
    rows = []
    for key, prov in p.provenance.items():
        spec = CATALOG.get(key)
        last_obs = p.data[key].dropna().index.max() if key in p.data else None
        age = (pd.Timestamp.utcnow().tz_localize(None) - last_obs).days if last_obs is not None else None
        tol = spec.max_age_days if spec else 30
        rows.append(
            {
                "series": key,
                "fred_id": spec.fred_id if spec else "?",
                "freq": spec.freq if spec else "?",
                "last_obs": last_obs,
                "age_days": age,
                "tolerance_days": tol,
                "status": "STALE" if (age is not None and age > tol) else "ok",
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "series"], ascending=[False, True])
