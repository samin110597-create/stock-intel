"""Scoring that tells you whether a probability means anything.

Accuracy is the wrong metric here and it is the one everybody reports. A
model that says 55% every day on a series with a 55% base rate has good
accuracy and zero information. Brier Skill Score against the base rate is the
number that matters: it is 0 for a model that has learned nothing, regardless
of how confident it sounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_skill(y: np.ndarray, p: np.ndarray, base: float | None = None) -> float:
    base = float(np.mean(y)) if base is None else base
    ref = np.mean((base - y) ** 2)
    if ref == 0:
        return 0.0
    return float(1 - np.mean((p - y) ** 2) / ref)


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """The plot you should look at before trusting any probability.

    If the 'predicted' and 'observed' columns diverge, the model's confidence
    is fiction even when its ranking is fine.
    """
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                "n": int(m.sum()),
                "predicted": round(float(p[m].mean()), 4),
                "observed": round(float(y[m].mean()), 4),
                "gap": round(float(p[m].mean() - y[m].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def block_permutation_pvalue(
    y: np.ndarray, p: np.ndarray, block: int = 21, n_iter: int = 2000, seed: int = 0
) -> float:
    """How often does shuffled data produce this much skill by luck?

    Blocks preserve autocorrelation, so this is a fair null. A p-value above
    ~0.05 means the measured skill is indistinguishable from noise and you
    should not trade the model no matter how good the equity curve looks.
    """
    rng = np.random.default_rng(seed)
    observed = brier_skill(y, p)
    n = len(y)
    nb = max(1, n // block)
    blocks = np.array_split(np.arange(n), nb)
    count = 0
    for _ in range(n_iter):
        order = rng.permutation(len(blocks))
        perm = np.concatenate([blocks[i] for i in order])[:n]
        if brier_skill(y[perm], p) >= observed:
            count += 1
    return (count + 1) / (n_iter + 1)


def summarize(
    y: np.ndarray,
    p: np.ndarray,
    fold_ids: np.ndarray | None = None,
    fold_base: dict[int, float] | None = None,
) -> dict:
    """`fold_base` is the base rate KNOWN AT TRAIN TIME for each fold.

    This matters more than it looks. Scoring a fold against its own test-period
    base rate grades the model against a benchmark that had to see the future.
    On a series whose base rate drifts — which is every real price series — that
    makes a genuinely skillful model look useless. The honest reference is the
    base rate you could have computed before the fold started.
    """
    ref_base = None
    if fold_base and fold_ids is not None:
        ref = np.array([fold_base[int(f)] for f in fold_ids])
        ref_base = float(np.mean(ref))
    out = {
        "n": int(len(y)),
        "base_rate": round(float(np.mean(y)), 4),
        "train_base_rate": round(ref_base, 4) if ref_base is not None else None,
        "brier": round(brier(y, p), 5),
        "brier_skill": round(brier_skill(y, p, ref_base), 5),
        "log_loss": round(log_loss(y, p), 5),
        "mean_pred": round(float(np.mean(p)), 4),
        "pred_std": round(float(np.std(p)), 4),
    }
    if fold_ids is not None:
        per = [
            brier_skill(
                y[fold_ids == f],
                p[fold_ids == f],
                (fold_base or {}).get(int(f)),
            )
            for f in np.unique(fold_ids)
        ]
        out["fold_skill"] = [round(v, 4) for v in per]
        out["fold_skill_mean"] = round(float(np.mean(per)), 5)
        out["fold_skill_se"] = round(float(np.std(per, ddof=1) / np.sqrt(len(per))), 5) if len(per) > 1 else None
        out["folds_positive"] = int(sum(v > 0 for v in per))
    return out
