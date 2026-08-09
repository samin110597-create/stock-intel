"""ETF holdings retrieval.

Honest limitation up front: holdings for niche funds come from provider
coverage that is patchy and changes without notice. This module does not
paper over that. If a fund is not covered, you get DataUnavailable naming
the fund, not an empty overlap matrix that quietly reports zero.

Broad-market and sector funds (SMH, SOXX, QQQ, XLK, SPY) are reliably
covered. Thematic micro-cap funds frequently are not.
"""

from __future__ import annotations

import pandas as pd

from ..errors import DataUnavailable
from ..provenance import Provenanced

# Funds where provider coverage is known to be reliable enough to build on.
WELL_COVERED = {"SPY", "QQQ", "SMH", "SOXX", "XLK", "XLF", "XLE", "IWM", "VOO", "VTI", "ARKK", "GDX", "GDXJ", "SIL"}


def fetch_many(router, etfs: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Returns (holdings, failures). Failures are returned, never swallowed."""
    ok: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    for e in etfs:
        sym = e.upper().strip()
        try:
            p: Provenanced = router.holdings(sym)
            ok[sym] = p.data
        except DataUnavailable as err:
            failed[sym] = str(err).splitlines()[-1].strip()
        except Exception as err:  # contract violation etc.
            failed[sym] = f"{type(err).__name__}: {err}"
    return ok, failed


def coverage_note(etfs: list[str]) -> list[str]:
    risky = [e.upper() for e in etfs if e.upper() not in WELL_COVERED]
    if not risky:
        return []
    return [
        f"{', '.join(risky)}: thematic/low-AUM funds. Provider holdings coverage is "
        "unreliable here. If these fail, the fix is the issuer's own holdings CSV, "
        "not another API key."
    ]
