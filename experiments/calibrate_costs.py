"""Week 9: calibrate prefill rate from MEASURED runs, then inspect B/c sensitivity.

No synthetic measurements are created. A GPU benchmark is available at
experiments/benchmark_prefill.py; alternatively supply measurements collected
elsewhere with the same CSV schema.

Usage:
    python3 -m experiments.calibrate_costs --measurements measurements.csv
    python3 -m experiments.calibrate_costs --measurements measurements.csv \
      --c0-values 0.1,0.2,0.4,0.8 --budgets 300000,600000

The output is a SCENARIO analysis, not an empirical estimate of occupancy
cost c. c0 and queueing_penalty_s remain user-supplied assumptions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.costs import CostModel

REQUIRED = ("context_tokens", "prefill_time_s")


def parse_positive_floats(value: str, kind: str) -> list[float]:
    result = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not result or any(not np.isfinite(x) or x <= 0 for x in result):
        raise ValueError(f"{kind} must contain finite, positive numbers")
    return result


def validate_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Require actual, complete positive prefill measurements, not placeholders."""
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")
    out = frame.copy()
    for name in REQUIRED:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    good = (
        np.isfinite(out["context_tokens"].to_numpy(dtype=float))
        & np.isfinite(out["prefill_time_s"].to_numpy(dtype=float))
        & (out["context_tokens"].to_numpy(dtype=float) > 0)
        & (out["prefill_time_s"].to_numpy(dtype=float) > 0)
    )
    if not bool(good.all()):
        raise ValueError(
            f"{int((~good).sum())} measurement rows have missing, nonfinite or "
            "nonpositive context_tokens/prefill_time_s; correct the input"
        )
    return out


def estimate_rate(frame: pd.DataFrame) -> dict:
    """Robust descriptive estimate of throughput; no claim of universal scaling.

    Median over measured contexts is intentional. Also report rate dispersion
    so hardware/context dependence cannot be hidden by a single number.
    """
    frame = validate_measurements(frame)
    rates = frame["context_tokens"].to_numpy(dtype=float) / frame["prefill_time_s"].to_numpy(dtype=float)
    return {
        "sample_count": int(len(rates)),
        "tokens_per_s_median": float(np.median(rates)),
        "tokens_per_s_p10": float(np.quantile(rates, 0.10)),
        "tokens_per_s_p90": float(np.quantile(rates, 0.90)),
        "context_tokens_min": float(frame["context_tokens"].min()),
        "context_tokens_max": float(frame["context_tokens"].max()),
        "method": "median of per-run observed context_tokens / prefill_time_s",
    }


def build_scenarios(
    throughput: float,
    contexts: list[float],
    budgets: list[int],
    c0_values: list[float],
    queueing_penalty_s: float = 0.0,
) -> pd.DataFrame:
    """Calculate B,c,b across explicit assumptions without fitting c from prefill."""
    if throughput <= 0 or not np.isfinite(throughput):
        raise ValueError("throughput must be positive")
    if queueing_penalty_s < 0 or not np.isfinite(queueing_penalty_s):
        raise ValueError("queueing_penalty_s must be finite and nonnegative")
    rows = []
    for budget in budgets:
        for c0 in c0_values:
            for c_mode in ("fixed", "proportional"):
                cost = CostModel(
                    prefill_tokens_per_s=float(throughput),
                    kv_budget_tokens=int(budget),
                    c_mode=c_mode,
                    c0=float(c0),
                    queueing_penalty_s=float(queueing_penalty_s),
                )
                for tokens in contexts:
                    B = float(cost.B(tokens))
                    c = float(cost.c(tokens))
                    rows.append({
                        "context_tokens": int(tokens),
                        "kv_budget_tokens": int(budget),
                        "c_mode": c_mode,
                        "c0_assumed": float(c0),
                        "queueing_penalty_s_assumed": float(queueing_penalty_s),
                        "prefill_tokens_per_s_measured_median": float(throughput),
                        "B_s_modeled": B,
                        "c_gpu_s_per_wall_s_assumed": c,
                        "break_even_s": B / c,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True,
                        help="Measured CSV containing context_tokens and prefill_time_s")
    parser.add_argument("--out-dir", type=Path, default=Path("results/calibration"),
                        help="Output folder; never overwrites existing policy results")
    parser.add_argument("--contexts", default="4096,16384,65536,122065,236575",
                        help="Comma-separated context lengths for scenarios")
    parser.add_argument("--budgets", default="300000,600000,1200000",
                        help="Comma-separated KV budgets in token-equivalent units")
    parser.add_argument("--c0-values", default="0.1,0.2,0.4,0.8",
                        help="Assumed occupancy-cost coefficients; NOT hardware-measured")
    parser.add_argument("--queueing-penalty-s", type=float, default=0.0,
                        help="Optional assumed penalty for single-pause cost only")
    args = parser.parse_args()
    measured = estimate_rate(pd.read_csv(args.measurements))
    contexts = parse_positive_floats(args.contexts, "contexts")
    budget_values = parse_positive_floats(args.budgets, "budgets")
    budgets = [int(x) for x in budget_values]
    if any(x != y for x, y in zip(budget_values, budgets)):
        parser.error("budgets must contain integers")
    c0_values = parse_positive_floats(args.c0_values, "c0-values")
    scenarios = build_scenarios(
        measured["tokens_per_s_median"], contexts, budgets, c0_values,
        queueing_penalty_s=args.queueing_penalty_s,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(args.out_dir / "break_even_scenarios.csv", index=False)
    manifest = {
        "measurement_source": str(args.measurements.resolve()),
        "measured_prefill": measured,
        "interpretation": (
            "Only prefill throughput is measured. B is estimated from throughput "
            "and context length. c_mode, c0, queueing penalty and KV budget "
            "are scenario assumptions, not directly calibrated values."
        ),
        "contexts": contexts,
        "budgets_tokens": budgets,
        "c0_assumptions": c0_values,
        "queueing_penalty_s_assumed": args.queueing_penalty_s,
    }
    (args.out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(measured, indent=2))
    print("\nScenario break-even times:")
    print(scenarios.groupby(["c_mode", "c0_assumed", "kv_budget_tokens"])
          ["break_even_s"].agg(["min", "median", "max"]).to_string())
    print(f"\nSaved {args.out_dir} (no original dataset or results modified)")


if __name__ == "__main__":
    main()
