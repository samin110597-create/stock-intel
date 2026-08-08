"""Secret scrubbing for anything that gets written to a public repo.

This module exists because of a specific, common failure: a provider returns
an error, the error string contains the full request URL, the URL contains
`?apikey=...`, and that string gets committed to a public JSON file. The key
is then in git history forever, and rotating it is the only fix.

Every string that leaves the build and lands in docs/data/ passes through
`clean()`. No exceptions.
"""

from __future__ import annotations

import os
import re

KEY_ENV_VARS = [
    "TWELVEDATA_KEY",
    "POLYGON_KEY",
    "FMP_KEY",
    "FINNHUB_KEY",
    "ALPHAVANTAGE_KEY",
    "FRED_API_KEY",
]

# Catches keys in URLs even if the value is not one we know about, e.g. a key
# passed through a proxy or a vendor we add later.
PARAM_PATTERN = re.compile(
    r"((?:api_?key|apikey|token|key|access_token)=)([^&\s\"'<>]{6,})", re.IGNORECASE
)

REDACTED = "[REDACTED]"


def _known_secrets() -> list[str]:
    out = []
    for var in KEY_ENV_VARS:
        v = os.getenv(var)
        if v and len(v) >= 8:
            out.append(v)
    return sorted(set(out), key=len, reverse=True)


def clean(text: object) -> object:
    """Redact secrets from a string. Recurses through dicts and lists."""
    if isinstance(text, dict):
        return {k: clean(v) for k, v in text.items()}
    if isinstance(text, (list, tuple)):
        return [clean(v) for v in text]
    if not isinstance(text, str):
        return text

    out = text
    for secret in _known_secrets():
        out = out.replace(secret, REDACTED)
    out = PARAM_PATTERN.sub(lambda m: m.group(1) + REDACTED, out)
    return out


def assert_clean(payload: object) -> None:
    """Raise if any known secret survived. Called before every file write.

    Belt and braces: `clean()` should have handled it, but a build that fails
    loudly is infinitely preferable to a key in git history.
    """
    blob = repr(payload)
    for secret in _known_secrets():
        if secret in blob:
            raise RuntimeError(
                "ABORTING: an API key survived redaction and was about to be "
                "written to a file. Do not commit this build."
            )
