"""The directional model, and the gate that stops it from talking.

Design stance: this model is guilty until proven innocent. It produces
probabilities only after clearing an out-of-sample skill test AND a block
permutation test. If it fails, `predict_proba` raises rather than returning
a number that would look identical to a good one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..errors import ModelNotQualified
from . import calibration as cal
from .walkforward import walk_forward_splits


def _estimator(seed: int = 0):
    """Small, heavily regularised, calibrated.

    Depth 3 and 200 leaves-worth of capacity is not an accident. On ~1500
    daily rows with 25 features, anything larger memorises. The isotonic
    calibration wrapper is fit on inner folds so the probabilities are not
    just the classifier's raw scores.
    """
    base = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=250,
        learning_rate=0.03,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", CalibratedClassifierCV(base, method="isotonic", cv=3)),
        ]
    )


@dataclass
class Gate:
    """Thresholds a model must clear before it is allowed to emit probabilities."""

    min_brier_skill: float = 0.005
    min_folds_positive_frac: float = 0.6
    max_permutation_p: float = 0.05
    min_oos_rows: int = 300

    def evaluate(self, report: dict, p_value: float) -> tuple[bool, list[str]]:
        reasons = []
        if report["n"] < self.min_oos_rows:
            reasons.append(f"only {report['n']} OOS rows (need {self.min_oos_rows})")
        if report["brier_skill"] < self.min_brier_skill:
            reasons.append(
                f"Brier skill {report['brier_skill']:.4f} < {self.min_brier_skill} "
                "(no better than predicting the base rate)"
            )
        folds = report.get("fold_skill") or []
        if folds:
            frac = report["folds_positive"] / len(folds)
            if frac < self.min_folds_positive_frac:
                reasons.append(
                    f"skill positive in only {report['folds_positive']}/{len(folds)} folds "
                    "(unstable across time, likely one lucky regime)"
                )
        if p_value > self.max_permutation_p:
            reasons.append(
                f"permutation p={p_value:.3f} > {self.max_permutation_p} "
                "(skill indistinguishable from noise)"
            )
        return (not reasons), reasons


@dataclass
class DirectionalModel:
    horizon: int = 5
    n_folds: int = 6
    seed: int = 0
    gate: Gate = field(default_factory=Gate)

    fitted_: bool = False
    qualified_: bool = False
    reasons_: list[str] = field(default_factory=list)
    report_: dict = field(default_factory=dict)
    oof_: pd.DataFrame | None = None
    reliability_: pd.DataFrame | None = None
    permutation_p_: float = 1.0
    final_: object = None
    features_: list[str] = field(default_factory=list)
    fold_base_: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectionalModel":
        self.features_ = list(X.columns)
        Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)

        preds = np.full(len(y), np.nan)
        folds = np.full(len(y), -1)
        fold_base: dict[int, float] = {}
        for f in walk_forward_splits(X.index, self.n_folds, self.horizon):
            est = _estimator(self.seed)
            est.fit(Xv[f.train], yv[f.train])
            preds[f.test] = est.predict_proba(Xv[f.test])[:, 1]
            folds[f.test] = f.index
            # benchmark the fold against what was knowable before it started
            fold_base[f.index] = float(np.mean(yv[f.train]))

        mask = ~np.isnan(preds)
        if mask.sum() == 0:
            raise ValueError("no out-of-sample predictions were produced")

        yo, po, fo = yv[mask], preds[mask], folds[mask]
        self.oof_ = pd.DataFrame({"y": yo, "p": po, "fold": fo}, index=X.index[mask])
        self.report_ = cal.summarize(yo, po, fo, fold_base)
        self.fold_base_ = fold_base
        self.reliability_ = cal.reliability_table(yo, po)
        self.permutation_p_ = cal.block_permutation_pvalue(yo, po, block=self.horizon * 4)
        self.report_["permutation_p"] = round(self.permutation_p_, 4)

        self.qualified_, self.reasons_ = self.gate.evaluate(self.report_, self.permutation_p_)
        self.fitted_ = True

        if self.qualified_:
            self.final_ = _estimator(self.seed).fit(Xv, yv)
        return self

    # -- output ---------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise ModelNotQualified("model has not been fitted")
        if not self.qualified_:
            raise ModelNotQualified(
                "model failed its out-of-sample gate and will not emit probabilities:\n  - "
                + "\n  - ".join(self.reasons_)
            )
        p = self.final_.predict_proba(X[self.features_].to_numpy(dtype=float))[:, 1]
        return pd.Series(p, index=X.index, name="p_up")

    def verdict(self) -> dict:
        """Always safe to call. This is what the dashboard shows."""
        if not self.fitted_:
            return {"status": "not fitted"}
        return {
            "status": "QUALIFIED" if self.qualified_ else "NOT QUALIFIED",
            "horizon_days": self.horizon,
            "reasons": self.reasons_,
            **self.report_,
        }

    def edge_summary(self) -> pd.DataFrame:
        """Observed hit rate by predicted-probability decile, out of sample.

        If the top decile does not outperform the bottom decile, the model
        has no ranking ability either, which is worth knowing even when the
        calibration gate passes.
        """
        if self.oof_ is None:
            raise ModelNotQualified("fit first")
        d = self.oof_.copy()
        d["decile"] = pd.qcut(d["p"], 10, labels=False, duplicates="drop")
        g = d.groupby("decile").agg(n=("y", "size"), pred=("p", "mean"), observed=("y", "mean"))
        return g.round(4)
