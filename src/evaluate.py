"""Single-pause scoring: every policy against the clairvoyant optimum.

This is the capacity-free view. Each pause is an independent ski-rental
instance, so the theoretical worst-case ratios apply exactly and can be
checked rather than assumed. The simulator adds the coupling between requests
that this view deliberately omits.

Two aggregations of "cost relative to the clairvoyant optimum", and they
disagree sharply on this workload:

    mean_ratio   mean of per-pause ratios. Dominated by the many short pauses,
                 which every policy serves at ratio 1. Flatters everything.
    total_ratio  sum of costs over sum of optimal costs. Weights long pauses by
                 what they actually cost. This is the one that matters.

Report both and say which is which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import CostModel
from .policies import Decision, Policy


def taus_for(policy: Policy, frame: pd.DataFrame) -> np.ndarray:
    """Evaluate a policy over every row of a pause table."""
    predicted = frame["predicted_duration_s"].to_numpy(dtype=float)
    context = frame["context_tokens"].to_numpy(dtype=float)
    tools = frame["tool_key"].astype(str).to_numpy()
    truth = frame["pause_s"].to_numpy(dtype=float)
    out = np.empty(len(frame), dtype=float)
    for i in range(len(frame)):
        out[i] = policy(
            Decision(
                predicted_duration_s=predicted[i],
                context_tokens=context[i],
                tool_key=tools[i],
                true_duration_s=truth[i],
            )
        )
    return out


def score(
    policies: dict[str, Policy],
    frame: pd.DataFrame,
    cost: CostModel,
) -> pd.DataFrame:
    """Cost and competitive ratio per policy over one pause table."""
    T = frame["pause_s"].to_numpy(dtype=float)
    ctx = frame["context_tokens"].to_numpy(dtype=float)
    opt = cost.clairvoyant(T, ctx)
    b = cost.break_even(ctx)

    rows = []
    for name, policy in policies.items():
        tau = taus_for(policy, frame)
        c = cost.cost(tau, T, ctx)
        ratio = np.divide(c, opt, out=np.ones_like(c), where=opt > 0)
        evicted = T > tau
        rows.append(
            {
                "policy": name,
                "mean_ratio": float(ratio.mean()),
                "total_ratio": float(c.sum() / opt.sum()),
                "p95_ratio": float(np.percentile(ratio, 95)),
                "max_ratio": float(ratio.max()),
                "total_cost_gpu_s": float(c.sum()),
                "eviction_rate": float(evicted.mean()),
                # Tokens that must be re-prefilled because we let the cache go.
                "recomputed_tokens": float(ctx[evicted].sum()),
                "mean_tau_s": float(np.mean(tau[np.isfinite(tau)])) if np.isfinite(tau).any() else float("inf"),
            }
        )
    out = pd.DataFrame(rows).sort_values("total_ratio").reset_index(drop=True)
    out.attrs["median_break_even_s"] = float(np.median(b))
    return out


def decompose(
    frame: pd.DataFrame,
    cost: CostModel,
    lambdas: list[float],
) -> pd.DataFrame:
    """Split total cost by whether the prediction got the decisive bit right.

    The lambda-policy reads the prediction through one bit -- whether
    That >= b -- so every pause falls in one of four cells. Aggregate ratios
    hide which cell drives the result; this does not.

    The cell that matters is under-called long pauses. A gain is only available
    on correctly-called long pauses, so when under-calls outnumber them the
    policy cannot win no matter how lambda is tuned.
    """
    from .policies import make_break_even_ttl, make_lambda_policy

    T = frame["pause_s"].to_numpy(dtype=float)
    ctx = frame["context_tokens"].to_numpy(dtype=float)
    P = frame["predicted_duration_s"].to_numpy(dtype=float)
    b = np.asarray(cost.break_even(ctx), dtype=float)

    true_long, pred_long = T >= b, P >= b
    cells = {
        "correct_short": (~true_long) & (~pred_long),
        "over_called": (~true_long) & pred_long,
        "correct_long": true_long & pred_long,
        "under_called": true_long & (~pred_long),
    }

    opt = cost.clairvoyant(T, ctx)
    breakeven = cost.cost(taus_for(make_break_even_ttl(cost), frame), T, ctx)
    lam_costs = {
        lam: cost.cost(taus_for(make_lambda_policy(cost, lam), frame), T, ctx) for lam in lambdas
    }

    rows = []
    for name, mask in cells.items():
        row = {
            "cell": name,
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "opt_cost": float(opt[mask].sum()),
            "break_even_cost": float(breakeven[mask].sum()),
        }
        for lam, c in lam_costs.items():
            row[f"lambda_{lam:g}_cost"] = float(c[mask].sum())
        rows.append(row)

    recall = cells["correct_long"].sum() / max(true_long.sum(), 1)
    precision = cells["correct_long"].sum() / max(pred_long.sum(), 1)
    out = pd.DataFrame(rows)
    out.attrs["long_pause_recall"] = float(recall)
    out.attrs["long_pause_precision"] = float(precision)
    return out


def worst_case_probe(cost: CostModel, lam: float, context_tokens: float = 122_000) -> pd.DataFrame:
    """Drive the lambda-policy to its two tight worst cases and check the bound.

    The bound 1 + 1/lambda is attained by a pause ending just after the
    deadline the policy set:

      - predicted long  -> tau = lambda*b, so T just above lambda*b
      - predicted short -> tau = b/lambda, so T just above b/lambda

    If the simulator and the cost model agree with the algebra, these two rows
    hit the bound and nothing exceeds it. This is a check on our implementation,
    not a result about the workload.
    """
    from .policies import make_lambda_policy

    policy = make_lambda_policy(cost, lam)
    b = float(cost.break_even(context_tokens))
    bound = 1.0 + 1.0 / lam

    cases = [
        ("predicted_long", 10 * b, lam * b * 1.0001),
        ("predicted_short", 0.0, (b / lam) * 1.0001),
    ]
    rows = []
    for label, predicted, T in cases:
        tau = policy(
            Decision(
                predicted_duration_s=predicted,
                context_tokens=context_tokens,
                tool_key="probe",
                true_duration_s=T,
            )
        )
        c = float(cost.cost(tau, T, context_tokens))
        opt = float(cost.clairvoyant(T, context_tokens))
        rows.append(
            {
                "case": label,
                "lambda": lam,
                "tau_s": tau,
                "true_duration_s": T,
                "ratio": c / opt,
                "bound_1_plus_1_over_lambda": bound,
                "within_bound": c / opt <= bound + 1e-9,
            }
        )
    return pd.DataFrame(rows)
