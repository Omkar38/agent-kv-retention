"""Does the predictor fail because of the data, or because of what we fed it?

Week 4 described the current tool call and nothing about the session it sits
in. This script adds the session-history features from `src/features.py` and
measures what they are worth, as an ablation ladder:

    week4_base               provider, model, tool, sizes, turn index
    plus_history             + what already happened in this session
    plus_history_clock       + time of day
    plus_history_clock_user  + user identity (flagged: optimistic, see below)

Scoring is on two levels, because they can disagree:

  **Ranking quality** -- average precision and ROC-AUC on the long-pause bit.
  These are threshold-free, so they measure the signal the model extracts
  rather than where we happen to put the cut. With a ~6% base rate, AP is the
  honest headline and accuracy would be meaningless.

  **Realised cost** -- the same features driven through the lambda-policy and
  scored against the clairvoyant optimum. A model can rank better and still
  not change a single retention decision, so this is what actually counts.

`user` is reported separately because sessions from the same user appear in
both train and test. A serving system does know its tenants, so conditioning
on them is fair, but the measured gain is an upper bound on what a previously
unseen user would get.

Run from the repository root:

    python3 -m experiments.run_features
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.classifier import bit_quality, select_threshold
from src.costs import CostModel
from src.evaluate import score, taus_for
from src.features import build_history_features, categoricals_in, feature_sets, numerics_in
from src.policies import make_break_even_ttl, make_lambda_policy
from experiments.run_classifier import feature_frame

OUT = Path("results/policy")
SEED = 560
LAMBDAS = [0.25, 0.5, 0.75]
C_MODES = ["fixed", "proportional"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_model(columns: list[str], seed: int = SEED) -> Pipeline:
    cats, nums = categoricals_in(columns), numerics_in(columns)
    steps = []
    if cats:
        steps.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
                cats,
            )
        )
    if nums:
        steps.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                nums,
            )
        )
    prep = ColumnTransformer(steps, sparse_threshold=0)
    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=31, learning_rate=0.06,
        min_samples_leaf=40, l2_regularization=2.0, random_state=seed,
    )
    return Pipeline([("preprocess", prep), ("estimator", model)])


def fit_calibrated(train: pd.DataFrame, y: np.ndarray, columns: list[str], seed: int = SEED):
    """Fit, then isotonically calibrate on sessions held out from the fit."""
    rng = np.random.default_rng(seed)
    sessions = pd.unique(train["session_id"].to_numpy())
    held = set(rng.choice(sessions, size=max(1, int(len(sessions) * 0.25)), replace=False))
    is_cal = train["session_id"].isin(held).to_numpy()
    if y[~is_cal].sum() == 0 or y[is_cal].sum() == 0:
        is_cal = np.zeros(len(train), dtype=bool)

    base = build_model(columns, seed)
    base.fit(train[columns][~is_cal] if is_cal.any() else train[columns],
             y[~is_cal] if is_cal.any() else y)
    if not is_cal.any():
        return base
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    except ImportError:  # pragma: no cover - scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrated.fit(train[columns][is_cal], y[is_cal])
    return calibrated


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    raw = {split: feature_frame(split) for split in ("train", "val", "test")}
    print(
        "pauses — "
        + " · ".join(f"{k} {len(v):,}/{v.session_id.nunique():,} sessions" for k, v in raw.items())
    )

    ranking_rows, quality_rows, cost_rows = [], [], []

    for c_mode in C_MODES:
        cost = CostModel(c_mode=c_mode)

        # History features depend on b, so they are rebuilt per cost mode.
        frames = {}
        for split, frame in raw.items():
            b = np.asarray(cost.break_even(frame["context_tokens"].to_numpy()), dtype=float)
            frames[split] = build_history_features(frame, break_even_s=b)

        labels, break_evens = {}, {}
        for split, frame in frames.items():
            b = np.asarray(cost.break_even(frame["context_tokens"].to_numpy()), dtype=float)
            break_evens[split] = b
            labels[split] = (frame["pause_s"].to_numpy(dtype=float) >= b).astype(int)

        print(f"\n=== c_mode={c_mode}   test base rate {labels['test'].mean():.3%} ===")

        for name, columns in feature_sets().items():
            model = fit_calibrated(frames["train"], labels["train"], columns)
            p_test = model.predict_proba(frames["test"][columns])[:, 1]
            p_val = model.predict_proba(frames["val"][columns])[:, 1]

            ap = average_precision_score(labels["test"], p_test)
            auc = roc_auc_score(labels["test"], p_test)
            ranking_rows.append(
                {
                    "c_mode": c_mode, "feature_set": name,
                    "average_precision": float(ap), "roc_auc": float(auc),
                    "base_rate": float(labels["test"].mean()),
                    "ap_lift_over_base": float(ap / labels["test"].mean()),
                    "n_features": len(columns),
                }
            )

            chosen = select_threshold(
                p_val, frames["val"]["pause_s"].to_numpy(dtype=float),
                frames["val"]["context_tokens"].to_numpy(dtype=float), cost, lam=0.5,
            )
            chosen.pop("curve")
            q = bit_quality(labels["test"], p_test >= chosen["threshold"])
            q.update(c_mode=c_mode, feature_set=name, threshold=chosen["threshold"])
            quality_rows.append(q)
            print(
                f"  {name:26s} AP={ap:.4f} ({ap / labels['test'].mean():5.2f}x base)  "
                f"AUC={auc:.4f}  recall={q['recall']:.3f} precision={q['precision']:.3f}"
            )

            # End-to-end: does better ranking actually lower realised cost?
            scored = frames["test"].copy()
            scored["tool_key"] = (
                scored["provider"].astype(str) + "|" + scored["tool_signature"].astype(str)
            )
            scored["predicted_duration_s"] = np.where(
                p_test >= chosen["threshold"], break_evens["test"] * 2.0, 0.0
            )
            table = score(
                {f"lambda_{lam:g}": make_lambda_policy(cost, lam) for lam in LAMBDAS},
                scored, cost,
            )
            table["c_mode"] = c_mode
            table["feature_set"] = name
            cost_rows.append(table)

        # The prediction-free reference, scored on the same rows.
        reference = frames["test"].copy()
        reference["tool_key"] = "x"
        reference["predicted_duration_s"] = 0.0
        ref = score({"break_even_ttl": make_break_even_ttl(cost)}, reference, cost)
        ref["c_mode"] = c_mode
        ref["feature_set"] = "none"
        cost_rows.append(ref)

    pd.DataFrame(ranking_rows).to_csv(OUT / "feature_ranking.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(OUT / "feature_bit_quality.csv", index=False)
    pd.concat(cost_rows, ignore_index=True).to_csv(OUT / "feature_costs.csv", index=False)

    manifest = {
        "git_commit": git_commit(), "seed": SEED,
        "feature_sets": {k: v for k, v in feature_sets().items()},
        "lambdas": LAMBDAS,
        "n_train": int(len(raw["train"])), "n_test": int(len(raw["test"])),
        "elapsed_s": round(time.time() - started, 1),
        "python": platform.python_version(), "platform": platform.platform(),
        "caveat_user_feature": (
            "Sessions from the same user appear in train and test; the user-feature "
            "row is an upper bound on what an unseen user would gain."
        ),
    }
    (OUT / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT}/feature_ranking.csv in {manifest['elapsed_s']}s")


if __name__ == "__main__":
    main()
