"""The scarce regime: policies compared where memory pressure actually bites.

The first sweep found two things that make a naive reading of it wrong:

  1. At a 150k-token budget, 47.6% of admissions have a context larger than the
     entire pool. Those runs measure an undersized pool, not a retention policy,
     so budgets below 600k are excluded here.

  2. At budgets where that is not true, the held-out trace's 384 sessions cannot
     saturate the pool -- utilisation plateaus near 0.49 however hard arrival
     gaps are compressed. Concurrency is bounded by session overlap, not arrival
     rate, so load is raised by replicating the session population instead.

Under replication the system becomes bound by serialised re-prefill rather than
by KV capacity, which is the channel through which retention policy actually
reaches job completion time: a policy that evicts more causes more re-prefill,
and re-prefill is the scarce resource.

Run from the repository root:

    python3 -m experiments.run_scarce
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

from src.costs import CostModel
from src.data import load_evaluation_frame, training_durations
from src.policies import (
    ContinuumStyle,
    always_retain,
    immediate_evict,
    make_break_even_ttl,
    make_clairvoyant,
    make_fixed_ttl,
    make_lambda_policy,
)
from src.sim import replicate_sessions, run_one

OUT = Path("results/policy")
SEED = 560

BUDGETS = [600_000, 1_200_000]  # below 600k the pool is smaller than the contexts
REPLICAS = [1, 4, 8]
LAMBDAS = [0.25, 0.5, 0.75, 1.0]
SIGNALS = ["hgb_q50", "hgb_q90"]
ARRIVAL_SCALE = 2000.0


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

    frames = {s: load_evaluation_frame(model=s) for s in SIGNALS}
    base = frames[SIGNALS[0]]
    train = training_durations()
    cost = CostModel(c_mode="fixed")
    fixed_ttl_s = float(np.median(cost.break_even(base["context_tokens"].to_numpy())))
    continuum = ContinuumStyle(cost).fit(train["tool_key"], train["pause_s"])

    prediction_free = {
        "immediate_evict": immediate_evict,
        "always_retain_lru": always_retain,
        f"fixed_ttl_{fixed_ttl_s:.0f}s": make_fixed_ttl(fixed_ttl_s),
        "break_even_ttl": make_break_even_ttl(cost),
        "continuum_style": continuum,
        "clairvoyant_per_pause": make_clairvoyant(cost),
    }

    rows = []
    for budget, replicas in itertools.product(BUDGETS, REPLICAS):
        replicated = {s: replicate_sessions(f, replicas) for s, f in frames.items()}
        n_sessions = replicated[SIGNALS[0]].session_id.nunique()
        print(f"budget={budget:,} replicas={replicas} ({n_sessions:,} sessions)")

        for name, policy in prediction_free.items():
            r = run_one(
                replicated[SIGNALS[0]], policy, cost,
                kv_budget_tokens=budget, arrival_scale=ARRIVAL_SCALE,
            )
            r.update(policy=name, signal="none", lam=np.nan,
                     budget_tokens=budget, replicas=replicas, n_sessions=n_sessions)
            rows.append(r)

        for signal, lam in itertools.product(SIGNALS, LAMBDAS):
            r = run_one(
                replicated[signal], make_lambda_policy(cost, lam), cost,
                kv_budget_tokens=budget, arrival_scale=ARRIVAL_SCALE,
            )
            r.update(policy=f"lambda_{lam:g}", signal=signal, lam=lam,
                     budget_tokens=budget, replicas=replicas, n_sessions=n_sessions)
            rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "scarce.csv", index=False)

    manifest = {
        "git_commit": git_commit(),
        "seed": SEED,
        "budgets": BUDGETS,
        "replicas": REPLICAS,
        "arrival_scale": ARRIVAL_SCALE,
        "lambdas": LAMBDAS,
        "signals": SIGNALS,
        "runs": int(len(out)),
        "elapsed_s": round(time.time() - started, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "note": (
            "Budgets below 600k excluded: single contexts exceed the pool. "
            "Replicated sessions raise offered load and describe a hypothetical "
            "server at N times this trace's concurrency, not additional evidence."
        ),
    }
    (OUT / "scarce_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(out)} runs in {manifest['elapsed_s']}s -> {OUT}/scarce.csv")


if __name__ == "__main__":
    main()
