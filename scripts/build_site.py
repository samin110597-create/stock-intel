#!/usr/bin/env python3
"""Build the static site payload.

Runs in GitHub Actions. Fetches everything the watchlist asks for, then
writes plain JSON into docs/data/ for the Pages front end to read. There is
no server: the browser only ever loads files this script committed.

Two rules govern the whole build:

  1. One failure must not kill the run. A dead provider for MU should not
     cost you the macro tab. Every section is isolated and records its own
     failure into the payload.
  2. Nothing reaches disk without passing through mi.redact. The site is
     public; the build logs are not the only place a key can leak.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mi.errors import DataUnavailable  # noqa: E402
from mi.etf import (concentration_table, coverage_note, fetch_many,  # noqa: E402
                    overlap_matrix, shared_exposure)
from mi.indicators import indicator_pack  # noqa: E402
from mi.macro import (GOLD_PLAYBOOK, classify, framework_status,  # noqa: E402
                      load_macro, metals_panel, staleness_report)
from mi.ml import DirectionalModel, assemble  # noqa: E402
from mi.providers import default_router  # noqa: E402
from mi.redact import assert_clean, clean  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data"
TODAY = dt.date.today().isoformat()
NOW = dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------- helpers
BUNDLE_TARGETS = [ROOT / "docs" / "data", ROOT / "local" / "data"]
_bundle: dict = {}


def write(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    safe = clean(payload)
    assert_clean(safe)
    (OUT / name).write_text(json.dumps(safe, indent=1, default=str))
    _bundle[name.replace(".json", "")] = safe
    print(f"  wrote docs/data/{name}")


def write_bundle() -> None:
    """Emit the same payload as a .js file.

    A page opened straight from disk (file://) cannot fetch a .json sitting
    next to it — the browser blocks it as a cross-origin read. A <script>
    tag is not subject to that restriction, so the local page gets its data
    as an assignment instead of a fetch. Same content, different wrapper.
    """
    assert_clean(_bundle)
    body = "window.SI_DATA = " + json.dumps(_bundle, default=str) + ";\n"
    for target in BUNDLE_TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        (target / "bundle.js").write_text(body)
        print(f"  wrote {target.relative_to(ROOT)}/bundle.js")


def num(v, digits: int = 4):
    """JSON-safe number. NaN becomes null rather than the string 'NaN',
    which would render as a plausible-looking value in the browser."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, digits)


def series_points(s: pd.Series, n: int, digits: int = 4) -> list:
    s = s.dropna().tail(n)
    return [[d.strftime("%Y-%m-%d"), num(v, digits)] for d, v in s.items()]


def aligned_chart(df: pd.DataFrame, cols: dict[str, pd.Series], n: int, digits: int = 2) -> dict:
    """Chart payload with ONE shared date axis.

    Repeating the date string on every point roughly triples the file size,
    and this JSON is committed to git on every run. Over a year of daily
    builds that difference is the whole repo.
    """
    idx = df.index[-n:]
    out = {"dates": [d.strftime("%Y-%m-%d") for d in idx]}
    for name, s in cols.items():
        out[name] = [num(v, digits) for v in s.reindex(idx)]
    return out


def why(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- sections
def build_stocks(router, cfg) -> dict:
    out, failures = {}, {}
    for sym in cfg["stocks"]:
        try:
            p = router.ohlcv(sym, cfg["history_start"], TODAY)
            px = p.data
            ind = indicator_pack(px)
            last, prev = ind.iloc[-1], px["close"].iloc[-2]
            close = float(px["close"].iloc[-1])
            recent = ind.tail(60)

            events = []
            for flag, label in (("choch", "Change of character"), ("bos", "Break of structure"),
                                ("sweep_low", "Sell-side sweep"), ("sweep_high", "Buy-side sweep")):
                hits = recent.index[recent[flag].fillna(False)]
                if len(hits):
                    events.append({"event": label, "date": hits[-1].strftime("%Y-%m-%d")})

            out[sym] = {
                "close": num(close, 2),
                "change_pct": num((close / prev - 1) * 100, 2),
                "as_of": px.index[-1].strftime("%Y-%m-%d"),
                "provider": p.sources()["close"],
                "bars": int(len(px)),
                "metrics": {
                    "RSI (14)": num(last["rsi14"], 1),
                    "vs EMA21 %": num(last["dist_ema21"] * 100, 2),
                    "vs SMA200 %": num(last["dist_sma200"] * 100, 2),
                    "Relative volume": num(last["rvol20"], 2),
                    "ATR (14)": num(last["atr14"], 2),
                    "Realised vol %": num(last["vol20"] * 100, 1),
                    "MACD histogram": num(last["macd_hist"], 3),
                },
                "structure": {1: "Bullish", -1: "Bearish", 0: "None"}[int(last["trend"])],
                "events": events,
                "chart": aligned_chart(
                    px,
                    {"close": px["close"], "ema21": ind["ema21"], "sma200": ind["sma200"]},
                    cfg["chart_points"],
                ),
            }
            print(f"  {sym}: ok via {out[sym]['provider']}")
        except Exception as e:
            failures[sym] = why(e)
            print(f"  {sym}: FAILED {why(e)}")
    return {"stocks": out, "failures": failures}


def build_macro(router, cfg) -> dict:
    macro_p = load_macro(router, start=cfg["history_start"])
    macro = macro_p.data
    call = classify(macro)

    payload = {
        "regime": call.regime,
        "confidence": call.confidence,
        "playbook": GOLD_PLAYBOOK.get(call.regime, ""),
        "axes": call.axes,
        "detail": {k: num(v, 2) for k, v in call.detail.items()},
        "missing": call.missing,
        "freshness": json.loads(staleness_report(macro_p).to_json(orient="records", date_format="iso")),
        "series": {},
        "metals": None,
        "metals_error": None,
    }
    for key in ("real_yield_10y", "dxy_broad", "breakeven_10y", "hy_spread"):
        if key in macro:
            payload["series"][key] = series_points(macro[key], cfg["chart_points"], 3)

    try:
        g = router.ohlcv(cfg["metals"]["gold"], cfg["history_start"], TODAY).data
        s = router.ohlcv(cfg["metals"]["silver"], cfg["history_start"], TODAY).data
        prices = pd.DataFrame({"gold": g["close"], "silver": s["close"]}).dropna()
        panel = metals_panel(prices, macro)
        latest = panel.iloc[-1]
        payload["metals"] = {
            "gold_proxy": cfg["metals"]["gold"],
            "silver_proxy": cfg["metals"]["silver"],
            "framework": clean(framework_status(panel)),
            "latest": {
                "Gold": num(latest.get("gold"), 2),
                "Silver": num(latest.get("silver"), 2),
                "Gold / silver ratio": num(latest.get("gs_ratio"), 1),
                "Ratio z-score (250d)": num(latest.get("gs_z"), 2),
                "Gold drawdown %": num(latest.get("gold_dd", np.nan) * 100, 1),
                "Gold in CAD": num(latest.get("gold_cad"), 2),
                "Silver in CAD": num(latest.get("silver_cad"), 2),
            },
            "chart": aligned_chart(
                panel,
                {"gs_ratio": panel["gs_ratio"], "gold": panel["gold"]},
                cfg["chart_points"],
            ),
        }
    except Exception as e:
        payload["metals_error"] = why(e)
        print(f"  metals: FAILED {why(e)}")
    return payload


def build_models(router, cfg) -> dict:
    mcfg = cfg["model"]
    horizon = mcfg["horizon_days"]
    try:
        macro = load_macro(router, start=cfg["history_start"]).data
    except Exception:
        macro = None

    out = {}
    for sym in mcfg["symbols"]:
        try:
            px = router.ohlcv(sym, cfg["history_start"], TODAY).data
            X, y = assemble(px, macro, horizon=horizon)
            m = DirectionalModel(horizon=horizon).fit(X, y)
            v = m.verdict()
            entry = {
                "status": v["status"],
                "horizon_days": horizon,
                "reasons": v["reasons"],
                "scorecard": {
                    "Out-of-sample rows": v["n"],
                    "Base rate": v["base_rate"],
                    "Brier score": v["brier"],
                    "Brier skill": v["brier_skill"],
                    "Log loss": v["log_loss"],
                    "Permutation p-value": v["permutation_p"],
                    "Folds with positive skill": f"{v.get('folds_positive')} of {len(v.get('fold_skill', []))}",
                },
                "fold_skill": v.get("fold_skill"),
                "reliability": json.loads(m.reliability_.to_json(orient="records")),
                "deciles": json.loads(m.edge_summary().reset_index().to_json(orient="records")),
                "probability": None,
            }
            if m.qualified_:
                entry["probability"] = num(float(m.predict_proba(X.tail(1)).iloc[0]), 4)
            out[sym] = entry
            print(f"  {sym}: {v['status']} (skill {v['brier_skill']:+.4f}, p={v['permutation_p']})")
        except Exception as e:
            out[sym] = {"status": "ERROR", "reasons": [why(e)], "scorecard": {}, "probability": None}
            print(f"  {sym}: ERROR {why(e)}")
    return {"models": out, "trained_with_macro": macro is not None}


def build_etf(router, cfg) -> dict:
    etfs = cfg["etfs"]
    holdings, failures = fetch_many(router, etfs)
    payload = {
        "requested": etfs,
        "failures": failures,
        "notes": coverage_note(etfs),
        "concentration": [],
        "overlap_weight": None,
        "overlap_names": None,
        "look_through": [],
    }
    if holdings:
        payload["concentration"] = json.loads(
            concentration_table(holdings).to_json(orient="records")
        )
    if len(holdings) >= 2:
        wm = overlap_matrix(holdings, "weight")
        nm = overlap_matrix(holdings, "name")
        payload["overlap_weight"] = {"labels": list(wm.columns), "matrix": wm.to_numpy().tolist()}
        payload["overlap_names"] = {"labels": list(nm.columns), "matrix": nm.to_numpy().tolist()}
        payload["look_through"] = json.loads(
            shared_exposure(holdings, top=20).to_json(orient="records")
        )
    return payload


# ---------------------------------------------------------------- main
def main() -> int:
    cfg = json.loads((ROOT / "config" / "watchlist.json").read_text())
    router = default_router(use_cache=False)

    print("Providers configured:", [p.name for p in router.providers] or "NONE")
    if router.skipped:
        print("Providers missing a key:", router.skipped)

    status = {
        "built_at": NOW.strftime("%Y-%m-%d %H:%M UTC"),
        "run_id": os.getenv("GITHUB_RUN_NUMBER", "local"),
        "commit": os.getenv("GITHUB_SHA", "")[:7],
        "providers_live": [p.name for p in router.providers],
        "providers_missing_key": router.skipped,
        "sections": {},
    }
    hard_failure = False

    for name, fn in (
        ("stocks", build_stocks),
        ("macro", build_macro),
        ("models", build_models),
        ("etf", build_etf),
    ):
        print(f"\n[{name}]")
        try:
            write(f"{name}.json", fn(router, cfg))
            status["sections"][name] = {"ok": True, "error": None}
        except Exception as e:
            traceback.print_exc()
            status["sections"][name] = {"ok": False, "error": why(e)}
            write(f"{name}.json", {"error": why(e)})
            if name in ("stocks", "macro"):
                hard_failure = True

    status["attempts"] = json.loads(router.attempt_table().to_json(orient="records")) if router.log else []
    write("status.json", status)
    _bundle["built_at"] = status["built_at"]
    write_bundle()

    print("\nBuild finished.")
    for name, s in status["sections"].items():
        print(f"  {name}: {'ok' if s['ok'] else 'FAILED — ' + str(s['error'])}")

    # A build where core sections died still publishes, so the site can show
    # you the failure. It exits non-zero so the Actions run is visibly red.
    return 1 if hard_failure else 0


if __name__ == "__main__":
    sys.exit(main())
