#!/usr/bin/env python3
"""Last line of defence before anything is committed.

Scans every file under docs/ for the literal value of each configured API
key. `mi.redact` should already have caught these, but this repo is public
and a key in git history cannot be un-published — it can only be rotated.
Two independent checks is the right number here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mi.redact import KEY_ENV_VARS, PARAM_PATTERN  # noqa: E402

import os  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    secrets = [v for v in (os.getenv(k) for k in KEY_ENV_VARS) if v and len(v) >= 8]
    problems = []

    for path in sorted((ROOT / "docs").rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".js", ".html", ".css"}:
            continue
        text = path.read_text(errors="ignore")
        for s in secrets:
            if s in text:
                problems.append(f"{path.relative_to(ROOT)}: contains a configured API key")
        for m in PARAM_PATTERN.finditer(text):
            if "[REDACTED]" not in m.group(0):
                problems.append(f"{path.relative_to(ROOT)}: unredacted credential parameter")

    if problems:
        print("REFUSING TO COMMIT — credential material found:")
        for p in sorted(set(problems)):
            print("  " + p)
        print("\nRotate the affected key before doing anything else, then re-run.")
        return 1

    print(f"No credential material in docs/ ({len(secrets)} key(s) checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
