"""Does session history lower realised cost? Settled by cross-validation.

`run_features.py` shows that adding session history raises average precision
on the long-pause bit from 0.336 to 0.519 (c fixed), and that this gain is
statistically solid. It also shows the realised-cost gain over the
prediction-free rule -- +0.89% -- sitting inside a confidence interval that
crosses zero.

That is a sample-size problem, not a null result. The held-out split has 384
sessions, and the cost metric is driven almost entirely by the ~4.5% of pauses
that are long; the other 95% score identically under every policy. So the
effective sample behind the cost number is a few thousand pauses, not 47,773.

This script removes that limitation by rotating every session through the test
position: 5-fold cross-validation grouped by session, so all 3,667 sessions are
scored out-of-fold exactly once. The confidence interval is then bootstrapped
over sessions on the pooled out-of-fold predictions.

This supplements rather than replaces the inherited split. The fixed
train/val/test assignment stays canonical for every other result in the
project, so numbers here are labelled as cross-validated and not mixed with it.

Run from the repository root:

    python3 -m experiments.run_feature_cv
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.classifier import select_threshold
from src.costs import CostModel
from src.evaluate import taus_for
from src.features import build_history_features, feature_sets
from src.policies import make_break_even_ttl, make_lambda_policy
from experiments.run_classifier import feature_frame
from experiments.run_features import fit_calibrated

OUT = Path("results/policy")
SEED = 560
FOLDS = 5
LAMBDAS = [0.25, 0.5, 0.75]
SETS = ["week4_base", "plus_history"]
C_MODES = ["fixed", "proportional"]
BOOTSTRAP = 2000


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def bootstrap_difference(
    session_ids: np.ndarray,
    numerator_a: np.ndarray,
    numerator_b: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Session-level bootstrap of (ratio_a - ratio_b), both against one optimum."""
    unique = np.unique(session_ids)
    index = {s: np.where(session_ids == s)[0] for s in unique}
    point = numerator_a.sum() / denominator.sum() - numerator_b.sum() / denominator.sum()
    draws = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        pick = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[s] for s in pick])
        d = denominator[rows].sum()
        draws[i] = (numerator_a[rows].sum() - numerator_b[rows].sum()) / d
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_gain_positive": float((draws > 0).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(SEED)

    # One pool of every scorable pause; folds are cut over sessions.
    pool = pd.concat([feature_frame(s) for s in ("train", "val", "test")], ignore_index=True)
    sessions = pool["session_id"].to_numpy()
    print(f"pool: {len(pool):,} pauses over {pd.unique(sessions).size:,} sessions, {FOLDS}-fold by session")

    rows, summary = [], []

    for c_mode in C_MODES:
        cost = CostModel(c_mode=c_mode)
        frame = build_history_features(
            pool, break_even_s=np.asarray(cost.break_even(pool["context_tokens"].to_numpy()), dtype=float)
        ).reset_index(drop=True)
        b = np.asarray(cost.break_even(frame["context_tokens"].to_numpy()), dtype=float)
        y = (frame["pause_s"].to_numpy(dtype=float) >= b).astype(int)
        T = frame["pause_s"].to_numpy(dtype=float)
        ctx = frame["context_tokens"].to_numpy(dtype=float)
        opt = cost.clairvoyant(T, ctx)
        groups = frame["session_id"].to_numpy()

        oof = {name: np.full(len(frame), np.nan) for name in SETS}
        splitter = GroupKFold(n_splits=FOLDS)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, y, groups), start=1):
            train = frame.iloc[train_idx]
            # A slice of the training sessions picks the acting threshold, so
            # the out-of-fold rows are never used to tune anything.
            tr_sessions = pd.unique(train["session_id"].to_numpy())
            held = set(rng.choice(tr_sessions, size=max(1, int(len(tr_sessions) * 0.15)), replace=False))
            is_val = train["session_id"].isin(held).to_numpy()
            for name in SETS:
                columns = feature_sets()[name]
                model = fit_calibrated(train[~is_val], y[train_idx][~is_val], columns)
                oof[name][test_idx] = model.predict_proba(frame.iloc[test_idx][columns])[:, 1]
            print(f"  {c_mode:12s} fold {fold}/{FOLDS} done")

        # Threshold chosen once on pooled out-of-fold predictions would peek at
        # the evaluation rows, so it is taken from the inherited validation
        # split instead -- the same operating point run_features.py reports.
        val_mask = frame["split"].to_numpy() == "val"
        reference = frame.assign(predicted_duration_s=0.0, tool_key="x")
        break_even_cost = cost.cost(
            taus_for(make_break_even_ttl(cost), reference), T, ctx
        )

        best_cost = {}
        for name in SETS:
            p = oof[name]
            chosen = select_threshold(
                p[val_mask], T[val_mask], ctx[val_mask], cost, lam=0.5
            )
            chosen.pop("curve")
            ap = average_precision_score(y, p)
            auc = roc_auc_score(y, p)

            best, best_lam = None, None
            for lam in LAMBDAS:
                scored = frame.assign(
                    predicted_duration_s=np.where(p >= chosen["threshold"], b * 2.0, 0.0),
                    tool_key="x",
                )
                c = cost.cost(taus_for(make_lambda_policy(cost, lam), scored), T, ctx)
                if best is None or c.sum() < best.sum():
                    best, best_lam = c, lam
            best_cost[name] = best

            rows.append(
                {
                    "c_mode": c_mode, "feature_set": name,
                    "average_precision": float(ap), "roc_auc": float(auc),
                    "base_rate": float(y.mean()), "threshold": chosen["threshold"],
                    "best_lambda": best_lam,
                    "total_ratio": float(best.sum() / opt.sum()),
                    "break_even_ratio": float(break_even_cost.sum() / opt.sum()),
                    "n_scored": int(len(frame)), "n_sessions": int(pd.unique(groups).size),
                }
            )

        for label, a, bb in [
            ("history_vs_break_even", break_even_cost, best_cost["plus_history"]),
            ("history_vs_week4", best_cost["week4_base"], best_cost["plus_history"]),
            ("week4_vs_break_even", break_even_cost, best_cost["week4_base"]),
        ]:
            stat = bootstrap_difference(groups, a, bb, opt, rng)
            stat.update(c_mode=c_mode, comparison=label)
            summary.append(stat)
            sig = "significant" if stat["ci_low"] > 0 else "not significant"
            print(
                f"  {c_mode:12s} {label:24s} gain={stat['point']:+.4f} "
                f"CI [{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}]  {sig}"
            )

    pd.DataFrame(rows).to_csv(OUT / "feature_cv.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "feature_cv_significance.csv", index=False)
    manifest = {
        "git_commit": git_commit(), "seed": SEED, "folds": FOLDS,
        "bootstrap_resamples": BOOTSTRAP, "feature_sets": SETS, "lambdas": LAMBDAS,
        "n_pauses": int(len(pool)), "n_sessions": int(pd.unique(sessions).size),
        "elapsed_s": round(time.time() - started, 1),
        "python": platform.python_version(), "platform": platform.platform(),
        "note": (
            "Cross-validated supplement to the inherited split, used only to "
            "resolve statistical power on the cost metric. The fixed "
            "train/val/test assignment remains canonical elsewhere."
        ),
    }
    (OUT / "feature_cv_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT}/feature_cv.csv in {manifest['elapsed_s']}s")


if __name__ == "__main__":
    main()
