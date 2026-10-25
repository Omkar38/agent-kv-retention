"""Retrain the predictor on the decision, and score it against the duration models.

Week 4 asked "how long will this pause be?" and answered it with a conditional
median. The policy asks "will this pause outlast b?" and reads nothing else.
This script trains on the second question directly, with the same feature
whitelist and the same session-level splits, so the only thing that changes is
the target.

It then puts four signals through the identical lambda-policy and scores them
against the clairvoyant optimum:

    q50          the Week 4 selection, thresholded at b
    q90          an upper quantile of the same regressor
    bit @ 0.5    the new classifier, thresholded symmetrically
    bit @ p*     the new classifier, thresholded where the cost model says

The last two differ only in the threshold, which isolates how much of the
result is the better target and how much is acting on the cost asymmetry.

Run from the repository root:

    python3 -m experiments.run_classifier
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.classifier import (
    FEATURES,
    bit_quality,
    cost_optimal_threshold,
    fit_bit_predictor,
    reliability,
    select_threshold,
)
from src.costs import CostModel
from src.data import DATASET, load_evaluation_frame
from src.evaluate import score
from src.policies import (
    ContinuumStyle,
    always_retain,
    immediate_evict,
    make_break_even_ttl,
    make_clairvoyant,
    make_lambda_policy,
)
from week4_handoff import tool_signature

OUT = Path("results/policy")
SEED = 560
LAMBDAS = [0.25, 0.5, 0.75, 1.0]
C_MODES = ["fixed", "proportional"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def feature_frame(split: str) -> pd.DataFrame:
    """The supervised cohort for one split, with Week 4's features attached."""
    frame = pd.read_parquet(DATASET)
    frame = frame[
        frame["analysis_track"].eq("tool_pause_prediction")
        & frame["pause_s"].notna()
        & np.isfinite(frame["pause_s"])
        & (frame["pause_s"] > 0)
        & frame["split"].eq(split)
    ].copy()
    frame["tool_signature"] = frame["feature_tool_names"].map(tool_signature)
    for column in ("feature_has_tools", "feature_multiple_tools"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["context_tokens"] = frame["context_tokens"].astype(float)
    return frame.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train = feature_frame("train")
    val = feature_frame("val")
    test = feature_frame("test")
    print(
        f"train {len(train):,} / {train.session_id.nunique():,} sessions · "
        f"val {len(val):,} / {val.session_id.nunique():,} · "
        f"test {len(test):,} / {test.session_id.nunique():,}"
    )

    # The regressor signals, joined on the canonical row index so the two
    # predictors are scored on exactly the same held-out pauses.
    q50 = load_evaluation_frame(model="hgb_q50")
    q90 = load_evaluation_frame(model="hgb_q90")
    key = ["session_id", "round_index"]
    q90_lookup = q90.set_index(key)["predicted_duration_s"]

    base = q50.copy()
    base["pred_q90_s"] = base.set_index(key).index.map(q90_lookup).to_numpy()
    # Drop columns the feature frame also supplies, so the merge does not
    # suffix them into provider_x / provider_y and strand the model's inputs.
    base = base.drop(columns=[c for c in FEATURES if c in base.columns])
    merged = base.merge(
        test[key + FEATURES].drop_duplicates(key), on=key, how="inner", validate="one_to_one"
    )
    print(f"scored on {len(merged):,} held-out pauses with features attached")

    quality_rows, threshold_rows, cost_rows, reliability_rows = [], [], [], []

    for c_mode in C_MODES:
        cost = CostModel(c_mode=c_mode)
        b_train = np.asarray(cost.break_even(train["context_tokens"].to_numpy()), dtype=float)
        b_test = np.asarray(cost.break_even(merged["context_tokens"].to_numpy()), dtype=float)
        u_train = train["pause_s"].to_numpy(dtype=float) / b_train
        y_test = (merged["pause_s"].to_numpy(dtype=float) >= b_test).astype(int)

        continuum = ContinuumStyle(cost).fit(
            train["provider"].astype(str) + "|" + train["feature_tool_names"].fillna("none").astype(str),
            train["pause_s"],
        )

        _ = cost_optimal_threshold(u_train, 0.5)  # recorded per cost mode below

        # One classifier per cost mode: the target itself depends on b.
        fitted = fit_bit_predictor(train, b_train, lam=0.5, seed=SEED)
        probability = fitted.probability(merged)

        # The acting threshold is chosen on validation sessions, never on test.
        b_val = np.asarray(cost.break_even(val["context_tokens"].to_numpy()), dtype=float)
        chosen = select_threshold(
            fitted.probability(val),
            val["pause_s"].to_numpy(dtype=float),
            val["context_tokens"].to_numpy(dtype=float),
            cost,
            lam=0.5,
        )
        curve = chosen.pop("curve")
        curve["c_mode"] = c_mode
        curve.to_csv(OUT / f"classifier_threshold_curve_{c_mode}.csv", index=False)
        threshold_rows.append(
            {
                "c_mode": c_mode,
                "derived_p_star": fitted.threshold,
                "validation_selected": chosen["threshold"],
                "validation_total_ratio": chosen["validation_total_ratio"],
                **{f"derived_{k}": v for k, v in fitted.threshold_detail.items() if k != "threshold"},
            }
        )
        print(
            f"  {c_mode:12s} base rate={y_test.mean():.3%}  "
            f"derived p*={fitted.threshold:.4f}  validation-selected={chosen['threshold']:.3f}"
        )

        rel = reliability(y_test, probability)
        rel["c_mode"] = c_mode
        reliability_rows.append(rel)

        signals = {
            "q50": merged["predicted_duration_s"].to_numpy() >= b_test,
            "q90": merged["pred_q90_s"].to_numpy() >= b_test,
            "bit@0.5": probability >= 0.5,
            "bit@p*_derived": probability >= fitted.threshold,
            "bit@val-selected": probability >= chosen["threshold"],
        }
        for name, predicted_long in signals.items():
            q = bit_quality(y_test, predicted_long)
            q.update(c_mode=c_mode, signal=name)
            quality_rows.append(q)
            print(
                f"    {name:9s} recall={q['recall']:.3f} precision={q['precision']:.3f} "
                f"flip={q['flip_rate']:.3f}"
            )

        # Score each signal through the identical policy. A predicted duration
        # of 2b or 0 reproduces the intended bit exactly, since the policy reads
        # only whether the prediction clears b.
        for name, predicted_long in signals.items():
            frame = merged.copy()
            frame["predicted_duration_s"] = np.where(predicted_long, b_test * 2.0, 0.0)
            frame["tool_key"] = (
                frame["provider"].astype(str) + "|" + frame["tool_signature"].astype(str)
            )
            policies = {f"lambda_{lam:g}": make_lambda_policy(cost, lam) for lam in LAMBDAS}
            table = score(policies, frame, cost)
            table["c_mode"] = c_mode
            table["signal"] = name
            cost_rows.append(table)

        # Prediction-free and offline references, scored once per cost mode.
        reference = merged.copy()
        reference["tool_key"] = (
            reference["provider"].astype(str) + "|" + reference["tool_signature"].astype(str)
        )
        ref_table = score(
            {
                "immediate_evict": immediate_evict,
                "always_retain": always_retain,
                "break_even_ttl": make_break_even_ttl(cost),
                "continuum_style": continuum,
                "clairvoyant": make_clairvoyant(cost),
            },
            reference,
            cost,
        )
        ref_table["c_mode"] = c_mode
        ref_table["signal"] = "none"
        cost_rows.append(ref_table)

    pd.DataFrame(quality_rows).to_csv(OUT / "classifier_bit_quality.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT / "classifier_thresholds.csv", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(
        OUT / "classifier_reliability.csv", index=False
    )
    costs = pd.concat(cost_rows, ignore_index=True)
    costs.to_csv(OUT / "classifier_costs.csv", index=False)

    manifest = {
        "git_commit": git_commit(),
        "seed": SEED,
        "n_train": int(len(train)),
        "n_test_scored": int(len(merged)),
        "train_sessions": int(train.session_id.nunique()),
        "test_sessions": int(merged.session_id.nunique()),
        "features": FEATURES,
        "lambdas": LAMBDAS,
        "elapsed_s": round(time.time() - started, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (OUT / "classifier_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT}/classifier_costs.csv in {manifest['elapsed_s']}s")


if __name__ == "__main__":
    main()
