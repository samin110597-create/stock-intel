"""Failure types. Everything here is designed to be LOUD.

The single most important rule in this repo: a data problem must never
degrade into a plausible-looking number. If we cannot get a field from a
provider, we raise. We do not forward-fill, we do not zero-fill, we do not
substitute a "close enough" series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MarketIntelError(Exception):
    """Base class for every failure this package raises."""


@dataclass
class Attempt:
    """One provider's shot at answering a request."""

    provider: str
    ok: bool
    detail: str
    elapsed_ms: int = 0

    def __str__(self) -> str:  # pragma: no cover - formatting only
        status = "ok" if self.ok else "FAIL"
        return f"  [{status}] {self.provider} ({self.elapsed_ms}ms): {self.detail}"


class DataUnavailable(MarketIntelError):
    """Every provider in the chain failed.

    Carries the full attempt log so the dashboard can show *why* rather than
    rendering an empty chart.
    """

    def __init__(self, request: str, attempts: list[Attempt]):
        self.request = request
        self.attempts = attempts
        body = "\n".join(str(a) for a in attempts) or "  (no providers configured)"
        super().__init__(f"No provider could satisfy: {request}\n{body}")


class ContractViolation(MarketIntelError):
    """A provider returned data that does not meet the schema contract.

    This is treated as a provider failure, not as usable data. A frame with
    NaN closes or a non-monotonic index is worse than no frame at all,
    because indicators computed on it look fine.
    """


class StaleData(MarketIntelError):
    """Data is real but older than the caller's freshness tolerance."""

    def __init__(self, field_name: str, age_days: float, max_age_days: float):
        self.field_name = field_name
        self.age_days = age_days
        self.max_age_days = max_age_days
        super().__init__(
            f"{field_name} is {age_days:.1f}d old, tolerance is {max_age_days:.1f}d"
        )


class ModelNotQualified(MarketIntelError):
    """The directional model failed its out-of-sample skill gate.

    Raised instead of returning probabilities. An unqualified model returning
    0.69 is more dangerous than a model that refuses to answer, because you
    will size a position on it.
    """


@dataclass
class Health:
    """Aggregated status for the Data Health tab."""

    ok: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.failed and not self.degraded
