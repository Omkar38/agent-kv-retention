"""Single-pause evaluation: every policy, both cost modes, full error sweep.

This is the capacity-free half of the evaluation. It answers the question the
proposal puts first -- when does acting on a duration prediction beat ignoring
it -- without the confound of requests competing for memory. The simulator adds
that competition afterwards.

Run from the repository root:

    python3 -m experiments.run_single_pause

Writes to results/policy/.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.costs import CostModel, describe
from src.data import load_evaluation_frame, training_durations
from src.evaluate import decompose, score, worst_case_probe
from src.policies import (
    ContinuumStyle,
    always_retain,
    immediate_evict,
    make_break_even_ttl,
    make_clairvoyant,
    make_fixed_ttl,
    make_lambda_policy,
)
from src.predictor import apply_error, decision_flip_rate

OUT = Path("results/policy")
LAMBDAS = [0.1, 0.25, 0.5, 0.75, 1.0]
SEED = 560


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_policies(cost: CostModel, continuum: ContinuumStyle, fixed_ttl_s: float) -> dict:
    policies = {
        "immediate_evict": immediate_evict,
        "always_retain": always_retain,
        f"fixed_ttl_{fixed_ttl_s:.0f}s": make_fixed_ttl(fixed_ttl_s),
        "break_even_ttl": make_break_even_ttl(cost),
        "continuum_style": continuum,
    }
    for lam in LAMBDAS:
        policies[f"lambda_{lam:g}"] = make_lambda_policy(cost, lam)
    policies["clairvoyant"] = make_clairvoyant(cost)
    return policies


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Two signals from the same gradient-boosted model. Week 4 selected q50 by
    # validation log1p-MAE; q50 recovers only ~22% of the long pauses the
    # policy has to catch, so the recall-oriented q90 is carried alongside it.
    frames = {sig: load_evaluation_frame(model=sig) for sig in ("hgb_q50", "hgb_q90")}
    frame = frames["hgb_q50"]
    train = training_durations()
    print(f"held-out pauses: {len(frame):,} over {frame.session_id.nunique():,} sessions")

    manifest = {
        "git_commit": git_commit(),
        "seed": SEED,
        "n_pauses": int(len(frame)),
        "n_sessions": int(frame.session_id.nunique()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lambdas": LAMBDAS,
    }

    cost_tables, flip_tables, probe_tables, decomp_tables, signal_tables = [], [], [], [], []

    for c_mode in ("fixed", "proportional"):
        cost = CostModel(c_mode=c_mode)
        b = cost.break_even(frame["context_tokens"].to_numpy())
        # Fixed TTL gets the best constant available: the median break-even.
        # Handing it the right answer makes it a strong baseline, not a strawman.
        fixed_ttl_s = float(np.median(b))
        continuum = ContinuumStyle(cost).fit(train["tool_key"], train["pause_s"])
        policies = build_policies(cost, continuum, fixed_ttl_s)

        manifest[f"cost_model_{c_mode}"] = describe(cost, frame["context_tokens"].to_numpy())
        manifest[f"fixed_ttl_{c_mode}_s"] = fixed_ttl_s

        # -- implementation audit: do the bounds hold where theory says they bind?
        for lam in LAMBDAS[:-1]:
            probe = worst_case_probe(cost, lam)
            probe["c_mode"] = c_mode
            probe_tables.append(probe)

        # -- which prediction signal should drive the policy?
        #
        # The policy reads one bit, so what matters is recall on long pauses,
        # not log error. Scoring q50 against q90 under the same cost model
        # isolates that choice from everything else.
        for signal, signal_frame in frames.items():
            table = score(policies, signal_frame, cost)
            table["c_mode"] = c_mode
            table["signal"] = signal
            signal_tables.append(table)

            d = decompose(signal_frame, cost, LAMBDAS[:-1])
            d["c_mode"] = c_mode
            d["signal"] = signal
            d["long_pause_recall"] = d.attrs["long_pause_recall"]
            d["long_pause_precision"] = d.attrs["long_pause_precision"]
            decomp_tables.append(d)
            print(
                f"  {c_mode:12s} {signal:8s} long-pause recall="
                f"{d.attrs['long_pause_recall']:.3f} precision={d.attrs['long_pause_precision']:.3f}"
            )

        # -- the error sweep
        conditions = [("clean", 0.0), ("oracle", 0.0)]
        conditions += [("noise", s) for s in (0.25, 0.5, 1.0, 2.0)]
        conditions += [("under", a) for a in (0.8, 0.5, 0.25, 0.1)]
        conditions += [("over", a) for a in (1.25, 2.0, 4.0, 10.0)]
        conditions += [("adversarial", f) for f in (0.05, 0.1, 0.25, 0.5, 1.0)]

        for mode, severity in conditions:
            degraded = frame.copy()
            degraded["predicted_duration_s"] = apply_error(
                frame["pause_s"].to_numpy(),
                frame["predicted_duration_s"].to_numpy(),
                np.asarray(b, dtype=float),
                mode=mode,
                severity=severity,
                rng=np.random.default_rng(SEED),
            )

            table = score(policies, degraded, cost)
            table["c_mode"] = c_mode
            table["error_mode"] = mode
            table["severity"] = severity
            cost_tables.append(table)

            flips = decision_flip_rate(
                degraded["pause_s"].to_numpy(),
                degraded["predicted_duration_s"].to_numpy(),
                np.asarray(b, dtype=float),
            )
            flips.update({"c_mode": c_mode, "error_mode": mode, "severity": severity})
            flip_tables.append(flips)

            print(f"  {c_mode:12s} {mode:12s} sev={severity:<5g} flip={flips['flip_rate']:.3f}")

    costs = pd.concat(cost_tables, ignore_index=True)
    costs.to_csv(OUT / "single_pause_costs.csv", index=False)
    pd.DataFrame(flip_tables).to_csv(OUT / "single_pause_flips.csv", index=False)
    pd.concat(probe_tables, ignore_index=True).to_csv(OUT / "worst_case_probe.csv", index=False)
    pd.concat(signal_tables, ignore_index=True).to_csv(OUT / "signal_comparison.csv", index=False)
    pd.concat(decomp_tables, ignore_index=True).to_csv(OUT / "decomposition.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {OUT}/single_pause_costs.csv  ({len(costs)} rows)")


if __name__ == "__main__":
    main()
