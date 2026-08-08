"""Per-field provenance.

Every number that reaches the dashboard can answer three questions:
  1. Which provider produced it?
  2. When was it fetched?
  3. What is the timestamp of the underlying observation (as_of)?

Fetched-at and as-of are different and the difference matters. CPI fetched
this morning is 40 days stale as an observation. Treating those as the same
thing is how a macro model ends up trading on last quarter's world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .errors import StaleData


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class FieldProvenance:
    name: str
    provider: str
    fetched_at: datetime
    as_of: datetime | None = None
    rows: int = 0
    note: str = ""

    def age_days(self, now: datetime | None = None) -> float:
        """Age of the *observation*, not of the fetch."""
        ref = self.as_of or self.fetched_at
        now = now or _utcnow()
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        return (now - ref).total_seconds() / 86400.0


@dataclass
class Provenanced:
    """A DataFrame plus a provenance record for each column."""

    data: pd.DataFrame
    provenance: dict[str, FieldProvenance] = field(default_factory=dict)
    request: str = ""

    def __post_init__(self) -> None:
        missing = set(self.data.columns) - set(self.provenance)
        if missing:
            raise ValueError(
                f"columns without provenance: {sorted(missing)} — "
                "every column must declare its source"
            )

    # -- access ---------------------------------------------------------
    def require(self, *fields: str) -> "Provenanced":
        absent = [f for f in fields if f not in self.data.columns]
        if absent:
            raise KeyError(f"required fields missing: {absent}")
        return self

    def require_fresh(self, max_age_days: float, *fields: str) -> "Provenanced":
        targets = fields or tuple(self.data.columns)
        for f in targets:
            p = self.provenance[f]
            age = p.age_days()
            if age > max_age_days:
                raise StaleData(f, age, max_age_days)
        return self

    def stale(self, max_age_days: float) -> dict[str, float]:
        return {
            name: p.age_days()
            for name, p in self.provenance.items()
            if p.age_days() > max_age_days
        }

    def sources(self) -> dict[str, str]:
        return {k: v.provider for k, v in self.provenance.items()}

    def audit(self) -> pd.DataFrame:
        """Table for the Data Health tab."""
        rows = []
        for name, p in self.provenance.items():
            rows.append(
                {
                    "field": name,
                    "provider": p.provider,
                    "as_of": p.as_of,
                    "fetched_at": p.fetched_at,
                    "age_days": round(p.age_days(), 2),
                    "rows": p.rows,
                    "note": p.note,
                }
            )
        return pd.DataFrame(rows).sort_values("field").reset_index(drop=True)

    # -- combination ----------------------------------------------------
    def join(self, other: "Provenanced", how: str = "inner") -> "Provenanced":
        overlap = set(self.data.columns) & set(other.data.columns)
        if overlap:
            raise ValueError(f"cannot join, overlapping columns: {sorted(overlap)}")
        merged = self.data.join(other.data, how=how)
        prov = {**self.provenance, **other.provenance}
        return Provenanced(merged, prov, f"{self.request} + {other.request}")


def uniform_provenance(
    df: pd.DataFrame,
    provider: str,
    request: str,
    note: str = "",
) -> Provenanced:
    """Helper for the common case: one provider supplied every column."""
    now = _utcnow()
    as_of = None
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
        last = df.index[-1]
        as_of = pd.Timestamp(last).to_pydatetime()
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
    prov = {
        c: FieldProvenance(
            name=c,
            provider=provider,
            fetched_at=now,
            as_of=as_of,
            rows=int(df[c].notna().sum()),
            note=note,
        )
        for c in df.columns
    }
    return Provenanced(df, prov, request)
