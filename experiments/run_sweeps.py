"""Simulator sweeps: memory budget, concurrency, lambda, and prediction signal.

The single-pause study says whether acting on a prediction lowers the modelled
cost. This says whether it lowers job completion time once sessions actually
compete for a fixed KV pool -- which is the claim a serving system cares about,
and the one no per-pause cost model can make.

Two axes carry most of the argument:

  c_mode     "fixed" vs "proportional" (open decision #2). Under "proportional"
             the break-even time is identical for every request, which makes a
             fixed TTL exactly equivalent to the break-even rule. Running both
             settles the decision with evidence instead of assertion.

  signal     Which quantile of the same gradient-boosted model drives the
             policy. Week 4 selected q50 by log error; q50 finds only ~22% of
             the long pauses the policy actually needs to catch, so q90 is
             carried alongside it.

Run from the repository root:

    python3 -m experiments.run_sweeps

Writes results/policy/sweeps.csv.
"""

from __future__ import annotations

import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.costs import CostModel, describe
from src.data import PREDICTION_COLUMNS, load_evaluation_frame, training_durations
from src.policies import (
    ContinuumStyle,
    always_retain,
    immediate_evict,
    make_break_even_ttl,
    make_clairvoyant,
    make_fixed_ttl,
    make_lambda_policy,
)
from src.sim import run_one

OUT = Path("results/policy")
SEED = 560

BUDGETS = [150_000, 300_000, 600_000, 1_200_000]
ARRIVAL_SCALES = [100.0, 500.0, 2000.0]
LAMBDAS = [0.1, 0.25, 0.5, 0.75, 1.0]
SIGNALS = ["hgb_q50", "hgb_q90"]
C_MODES = ["fixed", "proportional"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    frames = {sig: load_evaluation_frame(model=sig) for sig in SIGNALS}
    base = frames[SIGNALS[0]]
    train = training_durations()
    print(f"{len(base):,} pauses over {base.session_id.nunique():,} held-out sessions")

    rows = []
    total = len(C_MODES) * len(BUDGETS) * len(ARRIVAL_SCALES)
    done = 0

    for c_mode in C_MODES:
        cost = CostModel(c_mode=c_mode)
        b = np.asarray(cost.break_even(base["context_tokens"].to_numpy()), dtype=float)
        fixed_ttl_s = float(np.median(b))
        continuum = ContinuumStyle(cost).fit(train["tool_key"], train["pause_s"])

        # Prediction-free policies: the signal axis does not apply to them, so
        # they are run once per configuration and labelled "none".
        prediction_free = {
            "immediate_evict": immediate_evict,
            "always_retain_lru": always_retain,
            f"fixed_ttl_{fixed_ttl_s:.0f}s": make_fixed_ttl(fixed_ttl_s),
            "break_even_ttl": make_break_even_ttl(cost),
            "continuum_style": continuum,
            "clairvoyant": make_clairvoyant(cost),
        }

        for budget, scale in itertools.product(BUDGETS, ARRIVAL_SCALES):
            done += 1
            print(f"[{done}/{total}] c_mode={c_mode} budget={budget:,} arrival_scale={scale:g}")

            for name, policy in prediction_free.items():
                r = run_one(base, policy, cost, kv_budget_tokens=budget, arrival_scale=scale)
                r.update(
                    policy=name, signal="none", lam=np.nan,
                    c_mode=c_mode, budget_tokens=budget, arrival_scale=scale,
                )
                rows.append(r)

            for signal, lam in itertools.product(SIGNALS, LAMBDAS):
                r = run_one(
                    frames[signal], make_lambda_policy(cost, lam), cost,
                    kv_budget_tokens=budget, arrival_scale=scale,
                )
                r.update(
                    policy=f"lambda_{lam:g}", signal=signal, lam=lam,
                    c_mode=c_mode, budget_tokens=budget, arrival_scale=scale,
                )
                rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "sweeps.csv", index=False)

    manifest = {
        "git_commit": git_commit(),
        "seed": SEED,
        "n_pauses": int(len(base)),
        "n_sessions": int(base.session_id.nunique()),
        "budgets": BUDGETS,
        "arrival_scales": ARRIVAL_SCALES,
        "lambdas": LAMBDAS,
        "signals": SIGNALS,
        "prediction_columns": {s: PREDICTION_COLUMNS[s] for s in SIGNALS},
        "runs": int(len(out)),
        "elapsed_s": round(time.time() - started, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cost_models": {m: describe(CostModel(c_mode=m), base["context_tokens"].to_numpy()) for m in C_MODES},
    }
    (OUT / "sweeps_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(out)} runs in {manifest['elapsed_s']}s -> {OUT}/sweeps.csv")


if __name__ == "__main__":
    main()
