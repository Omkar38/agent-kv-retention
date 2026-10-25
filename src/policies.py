"""The retention policies, as interchangeable functions over identical traces.

Every policy answers one question: given what is knowable when a pause begins,
how long should this cache be retained? Each returns tau in seconds, where 0 is
"evict immediately" and inf is "retain until something else forces eviction".

The policies differ only in what they condition on:

    immediate_evict       nothing
    always_retain         nothing
    fixed_ttl             a constant chosen offline
    break_even_ttl        this request's B/c
    continuum_style       this tool's historical duration distribution
    lambda_policy         a per-call duration prediction, fenced by B/c
    clairvoyant           the true duration (offline; the ratio denominator)

LRU is not in this list because it is not a retention rule. It is a victim-
selection rule that applies when memory is already full, and it lives in the
simulator. Pairing always_retain with LRU eviction is what current engines do
once you stop evicting on decode-end, so that combination is the honest
"always-retain" baseline under capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class Decision:
    """What the policy is allowed to see when the pause begins.

    Nothing here may depend on the outcome of the pause. `true_duration` is
    carried for the clairvoyant policy and for scoring only; any policy other
    than `clairvoyant` that reads it is cheating, and the audit in
    `tests/test_policies.py` checks that none of them do.
    """

    predicted_duration_s: float
    context_tokens: float
    tool_key: str
    true_duration_s: float = float("nan")


Policy = Callable[[Decision], float]


# -- prediction-free ------------------------------------------------------


def immediate_evict(_: Decision) -> float:
    """Current vLLM / SGLang behaviour: a request that stops decoding is done."""
    return 0.0


def always_retain(_: Decision) -> float:
    """Never voluntarily release. Under capacity, LRU decides who actually goes."""
    return float("inf")


def make_fixed_ttl(tau_s: float) -> Policy:
    """One timeout for everything, chosen offline. SAGA's shape, roughly."""

    def policy(_: Decision) -> float:
        return tau_s

    return policy


def make_break_even_ttl(cost: CostModel) -> Policy:
    """tau = b = B/c. The best any deterministic prediction-free rule can do.

    Worst-case ratio 2, and no prediction-free rule beats that, so this -- not
    immediate eviction -- is the baseline our contribution has to clear.
    """

    def policy(d: Decision) -> float:
        return float(cost.break_even(d.context_tokens))

    return policy


# -- distribution-aware ---------------------------------------------------


class ContinuumStyle:
    """Our reconstruction of Continuum / CacheTTL from the published description.

    Continuum selects the TTL maximising expected net benefit against memory
    occupancy, using the empirical CDF of that tool's recorded durations. We
    implement the equivalent minimisation of expected cost:

        tau* = argmin_tau  E[ c*min(T,tau) + B*1[T>tau] ]

    with the expectation taken over the tool's *training* durations. The
    minimum always falls on an observed duration or at zero, so we evaluate
    candidates rather than optimising continuously.

    Numbers from this class are not comparable to figures in the original
    paper: different hardware, model, workload, and a reconstruction rather
    than the authors' code.

    The point of including it is structural, not competitive. Its distribution
    is per-tool -- conditioned on nothing about the specific call -- and it is
    fitted on the training split and then evaluated on held-out sessions. That
    is precisely the divergence no published result reports.
    """

    def __init__(self, cost: CostModel, max_candidates: int = 256, ttl_cap_s: float = 300.0):
        self.cost = cost
        self.max_candidates = max_candidates
        self.ttl_cap_s = ttl_cap_s
        self._candidates: dict[str, np.ndarray] = {}
        self._global: np.ndarray = np.array([0.0])

    def fit(self, tool_key: pd.Series, duration_s: pd.Series) -> "ContinuumStyle":
        """Record each tool's empirical duration distribution. Training split only."""
        frame = pd.DataFrame({"tool": tool_key.astype(str), "d": duration_s.astype(float)})
        frame = frame[np.isfinite(frame["d"]) & (frame["d"] >= 0)]
        self._global = self._summarise(frame["d"].to_numpy())
        for tool, group in frame.groupby("tool", sort=False):
            self._candidates[tool] = self._summarise(group["d"].to_numpy())
        return self

    def _summarise(self, durations: np.ndarray) -> np.ndarray:
        """Compress a duration sample to a fixed set of quantiles.

        Keeps the fit O(1) per tool regardless of how often the tool appears,
        and keeps evaluation cheap. The quantile grid is denser in the tail,
        which is where the TTL decision actually lives.
        """
        if durations.size == 0:
            return np.array([0.0])
        qs = np.concatenate([
            np.linspace(0.0, 0.9, 64, endpoint=False),
            np.linspace(0.9, 0.999, 64),
        ])
        return np.unique(np.quantile(durations, qs))

    def __call__(self, d: Decision) -> float:
        sample = self._candidates.get(d.tool_key)
        if sample is None or sample.size == 0:
            sample = self._global
        B = float(self.cost.B(d.context_tokens))
        c = float(self.cost.c(d.context_tokens))

        # Candidate TTLs: zero, each observed quantile, capped.
        taus = np.concatenate([[0.0], sample[sample <= self.ttl_cap_s], [self.ttl_cap_s]])
        taus = np.unique(taus)

        # E[cost] under the empirical distribution, vectorised over candidates.
        held = np.minimum(sample[None, :], taus[:, None]).mean(axis=1)
        miss = (sample[None, :] > taus[:, None]).mean(axis=1)
        expected = c * held + B * miss
        return float(taus[int(np.argmin(expected))])


# -- prediction-aware -----------------------------------------------------


def make_lambda_policy(cost: CostModel, lam: float) -> Policy:
    """tau = pi(That, c, B, lambda): the prediction, fenced by the break-even time.

    Purohit, Svitkina & Kumar (NeurIPS 2018), Algorithm 2, in our notation.
    lambda in (0, 1], and *smaller lambda means more trust in the prediction*:

        That >= b  (long pause predicted)   ->  tau = lambda * b
        That <  b  (short pause predicted)  ->  tau = b / lambda

    At lambda = 1 both branches collapse to the prediction-free break-even rule.
    As lambda -> 0 it becomes "evict at once if long, hold forever if short",
    which is 1-consistent and has no robustness bound.

    Gives (1+lambda)-consistency and (1+1/lambda)-robustness. The policy reads
    the prediction through one bit only -- whether That >= b -- so two
    predictors with very different absolute error behave identically whenever
    they agree on that bit. That is why the evaluation reports the flip rate
    rather than MAE.

    The algorithm is not ours. The contribution is the formulation for KV
    retention and the characterisation under prediction failure.
    """
    if not 0 < lam <= 1:
        raise ValueError(f"lambda must be in (0, 1], got {lam}")

    def policy(d: Decision) -> float:
        b = float(cost.break_even(d.context_tokens))
        predicted_long = d.predicted_duration_s >= b
        return lam * b if predicted_long else b / lam

    return policy


# -- offline --------------------------------------------------------------


def make_clairvoyant(cost: CostModel) -> Policy:
    """Knows T. Holds through a short pause, evicts a long one at once.

    This is the per-pause offline optimum, OPT(T) = min(c*T, B), and the
    denominator of the empirical competitive ratio. It is not achievable
    online; it bounds what any prediction could buy.
    """

    def policy(d: Decision) -> float:
        b = float(cost.break_even(d.context_tokens))
        return 0.0 if d.true_duration_s >= b else float("inf")

    return policy


def build_suite(cost: CostModel, lam: float = 0.5, fixed_ttl_s: float = 30.0) -> dict[str, Policy]:
    """The standard comparison set, minus Continuum-style, which needs fitting."""
    return {
        "immediate_evict": immediate_evict,
        "always_retain": always_retain,
        f"fixed_ttl_{fixed_ttl_s:g}s": make_fixed_ttl(fixed_ttl_s),
        "break_even_ttl": make_break_even_ttl(cost),
        f"lambda_{lam:g}": make_lambda_policy(cost, lam),
        "clairvoyant": make_clairvoyant(cost),
    }
