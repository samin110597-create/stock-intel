"""Parquet cache with a provenance sidecar.

The cache stores *what a provider actually returned*, keyed by request. It
never merges two providers' answers into one file, because then the sidecar
would be a lie.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .provenance import FieldProvenance, Provenanced

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _key(request: str) -> str:
    return hashlib.sha1(request.encode()).hexdigest()[:16]


def _engine() -> str:
    """Parquet if pyarrow is present, pickle otherwise.

    The cache is a local convenience, not a deliverable. Missing pyarrow
    should degrade the cache format, not take down the app.
    """
    try:
        import pyarrow  # noqa: F401

        return "parquet"
    except ImportError:
        return "pickle"


def _paths(request: str, root: Path) -> tuple[Path, Path]:
    k = _key(request)
    ext = "parquet" if _engine() == "parquet" else "pkl"
    return root / f"{k}.{ext}", root / f"{k}.json"


def write(p: Provenanced, root: Path = CACHE_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = _paths(p.request, root)
    if _engine() == "parquet":
        p.data.to_parquet(data_path)
    else:
        p.data.to_pickle(data_path)
    meta = {
        "request": p.request,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            k: {
                **asdict(v),
                "fetched_at": v.fetched_at.isoformat(),
                "as_of": v.as_of.isoformat() if v.as_of else None,
            }
            for k, v in p.provenance.items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def read(request: str, max_age_hours: float, root: Path = CACHE_DIR) -> Provenanced | None:
    data_path, meta_path = _paths(request, root)
    if not data_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    written = datetime.fromisoformat(meta["written_at"])
    age_h = (datetime.now(timezone.utc) - written).total_seconds() / 3600
    if age_h > max_age_hours:
        return None
    df = pd.read_parquet(data_path) if _engine() == "parquet" else pd.read_pickle(data_path)
    prov = {}
    for k, v in meta["provenance"].items():
        prov[k] = FieldProvenance(
            name=v["name"],
            provider=v["provider"],
            fetched_at=datetime.fromisoformat(v["fetched_at"]),
            as_of=datetime.fromisoformat(v["as_of"]) if v.get("as_of") else None,
            rows=v.get("rows", 0),
            note=(v.get("note", "") + f" [cache {age_h:.1f}h]").strip(),
        )
    return Provenanced(df, prov, meta["request"])


def clear(root: Path = CACHE_DIR) -> int:
    if not root.exists():
        return 0
    n = 0
    for f in list(root.glob("*.parquet")) + list(root.glob("*.pkl")) + list(root.glob("*.json")):
        f.unlink()
        n += 1
    return n
