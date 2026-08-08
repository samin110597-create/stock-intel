"""The router: the only place in the repo that decides where data comes from.

Behaviour, in order:
  1. Try the cache if the caller allows a stale-enough copy.
  2. Walk the provider chain. Record every attempt with timing and reason.
  3. Validate each response against the contract BEFORE accepting it.
  4. If nothing passes, raise DataUnavailable with the full log.

There is deliberately no step 5 that returns partial or synthesised data.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import pandas as pd

from .. import cache as cache_mod
from ..contracts import validate_holdings, validate_ohlcv, validate_series
from ..errors import Attempt, DataUnavailable, Health
from ..provenance import Provenanced, uniform_provenance
from .base import Provider, RateLimited


class DataRouter:
    def __init__(self, providers: Iterable[Provider], use_cache: bool = True):
        self.providers = [p for p in providers if p.configured]
        self.skipped = [p.name for p in providers if not p.configured]
        self.use_cache = use_cache
        self.log: list[tuple[str, list[Attempt]]] = []

    # -- generic engine --------------------------------------------------
    def _resolve(
        self,
        request: str,
        capability: str,
        call: Callable[[Provider], object],
        validate: Callable[[object, str], object],
        cache_hours: float,
    ) -> Provenanced:
        if self.use_cache and cache_hours > 0:
            hit = cache_mod.read(request, cache_hours)
            if hit is not None:
                self.log.append((request, [Attempt("cache", True, "hit", 0)]))
                return hit

        attempts: list[Attempt] = []
        for name in self.skipped:
            attempts.append(Attempt(name, False, "no API key configured"))

        for p in self.providers:
            if capability not in p.capabilities:
                continue
            t0 = time.time()
            try:
                raw = call(p)
                clean = validate(raw, request)
                ms = int((time.time() - t0) * 1000)
                attempts.append(Attempt(p.name, True, f"{len(clean)} rows", ms))
                self.log.append((request, attempts))
                frame = clean if isinstance(clean, pd.DataFrame) else clean.to_frame()
                return uniform_provenance(frame, p.name, request)
            except RateLimited as e:
                attempts.append(Attempt(p.name, False, f"rate limited: {e}", int((time.time() - t0) * 1000)))
            except NotImplementedError:
                continue
            except Exception as e:  # provider error OR contract violation
                attempts.append(
                    Attempt(p.name, False, f"{type(e).__name__}: {e}", int((time.time() - t0) * 1000))
                )

        self.log.append((request, attempts))
        raise DataUnavailable(request, attempts)

    # -- public surface --------------------------------------------------
    def ohlcv(self, symbol: str, start: str, end: str, cache_hours: float = 12) -> Provenanced:
        req = f"ohlcv:{symbol}:{start}:{end}"
        p = self._resolve(
            req,
            "ohlcv",
            lambda pr: pr.daily_ohlcv(symbol, start, end),
            lambda raw, _: validate_ohlcv(raw, symbol),
            cache_hours,
        )
        cache_mod.write(p)
        return p

    def macro(self, series_id: str, start: str = "2000-01-01", cache_hours: float = 24) -> Provenanced:
        req = f"macro:{series_id}:{start}"
        p = self._resolve(
            req,
            "macro",
            lambda pr: pr.macro_series(series_id, start),
            lambda raw, _: validate_series(raw, series_id),
            cache_hours,
        )
        cache_mod.write(p)
        return p

    def holdings(self, etf: str, cache_hours: float = 168) -> Provenanced:
        req = f"holdings:{etf}"
        p = self._resolve(
            req,
            "etf_holdings",
            lambda pr: pr.etf_holdings(etf),
            lambda raw, _: validate_holdings(raw, etf),
            cache_hours,
        )
        cache_mod.write(p)
        return p

    # -- diagnostics -----------------------------------------------------
    def health(self) -> Health:
        h = Health()
        for name in self.skipped:
            h.failed.append(f"{name}: no API key")
        for p in self.providers:
            h.ok.append(f"{p.name}: {sorted(p.capabilities)}")
        return h

    def attempt_table(self) -> pd.DataFrame:
        rows = []
        for req, attempts in self.log:
            for a in attempts:
                rows.append(
                    {
                        "request": req,
                        "provider": a.provider,
                        "ok": a.ok,
                        "ms": a.elapsed_ms,
                        "detail": a.detail,
                    }
                )
        return pd.DataFrame(rows)


def default_router(use_cache: bool = True) -> DataRouter:
    from .vendors import ALL_PROVIDERS

    return DataRouter([cls() for cls in ALL_PROVIDERS], use_cache=use_cache)
