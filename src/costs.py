"""Cost model for KV-cache retention during agent tool pauses.

For a paused request retained for time tau, with unobserved pause duration T:

    C(tau, T) = c * min(T, tau) + B * 1[T > tau]

B is the eviction penalty (re-prefill the context, plus queueing on return) and c
is the per-second opportunity cost of occupying GPU memory. The prediction-free
break-even retention time is b = B / c.

Open decision #2 (see Readme): is c the same for every request?

Two defensible answers, and they lead to different projects, so both are
implemented here and selected by `CostModel.c_mode`:

  "fixed"         c is a constant per retained session, independent of cache size.
                  Then b = B/c scales with context length, so every request has its
                  own break-even. This is what the proposal assumes, and it is what
                  makes a per-tool TTL blind to B.

  "proportional"  c is proportional to the fraction of the KV budget the cache
                  occupies, because a larger cache displaces proportionally more
                  work. Then B and c both scale with context_tokens and they cancel:

                      b = (context/rate) / (c0 * context/M) = M / (rate * c0)

                  b is identical for every request. A fixed TTL becomes a *strong*
                  baseline and the argument that per-tool TTLs cannot see B
                  collapses.

Which one holds is an empirical question about the serving system, not something
to settle by assertion. Reporting both turns a blocking decision into a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Maps a pause's context length to its eviction penalty and break-even time.

    All costs are expressed in GPU-seconds of useful work lost, so B and
    c * seconds are directly comparable.
    """

    prefill_tokens_per_s: float = 10_000.0
    """Prefill throughput. 10k tok/s is A100-class for an 8B model with GQA."""

    kv_budget_tokens: int = 300_000
    """Total KV pool, in tokens. ~40GB at 128KB/token for Llama-3-8B."""

    c_mode: str = "fixed"
    """"fixed" or "proportional" - see module docstring."""

    c0: float = 0.4
    """Opportunity-cost scale.

    Under "fixed", c = c0 GPU-seconds lost per wall-second, for any cache.
    Under "proportional", c = c0 * context_tokens / kv_budget_tokens.

    The default 0.4 makes the two modes agree at the median context (122k tokens
    of a 300k budget), so a difference between them is a difference in *spread*,
    not in level. That keeps the comparison honest.
    """

    queueing_penalty_s: float = 0.0
    """Flat delay added to B for re-entering the admission queue on return.

    Zero by default: the simulator models queueing explicitly, so charging it
    here too would double-count. Non-zero only for the single-pause analysis,
    where there is no queue to model.
    """

    def __post_init__(self) -> None:
        if self.c_mode not in ("fixed", "proportional"):
            raise ValueError(f"c_mode must be 'fixed' or 'proportional', got {self.c_mode!r}")
        if self.prefill_tokens_per_s <= 0:
            raise ValueError("prefill_tokens_per_s must be positive")
        if self.kv_budget_tokens <= 0:
            raise ValueError("kv_budget_tokens must be positive")
        if self.c0 <= 0:
            raise ValueError("c0 must be positive")

    # -- primitives ---------------------------------------------------------

    def B(self, context_tokens):
        """Eviction penalty: GPU-seconds to re-prefill `context_tokens`."""
        tokens = np.asarray(context_tokens, dtype=float)
        return tokens / self.prefill_tokens_per_s + self.queueing_penalty_s

    def c(self, context_tokens):
        """Per-second opportunity cost of retaining this cache."""
        tokens = np.asarray(context_tokens, dtype=float)
        if self.c_mode == "fixed":
            return np.full(tokens.shape, self.c0, dtype=float)
        return self.c0 * tokens / self.kv_budget_tokens

    def break_even(self, context_tokens):
        """b = B / c, the retention time that costs exactly one eviction."""
        return self.B(context_tokens) / self.c(context_tokens)

    # -- the cost itself ----------------------------------------------------

    def cost(self, tau, T, context_tokens):
        """C(tau, T) = c * min(T, tau) + B * 1[T > tau]."""
        tau = np.asarray(tau, dtype=float)
        T = np.asarray(T, dtype=float)
        held = np.minimum(T, tau)
        evicted = (T > tau).astype(float)
        return self.c(context_tokens) * held + self.B(context_tokens) * evicted

    def clairvoyant(self, T, context_tokens):
        """OPT(T) = min(c*T, B): hold throughout, or evict immediately.

        An oracle that knows T picks whichever is cheaper, so this is the
        denominator of the empirical competitive ratio.
        """
        T = np.asarray(T, dtype=float)
        return np.minimum(self.c(context_tokens) * T, self.B(context_tokens))


def describe(model: CostModel, context_tokens) -> dict:
    """Summary of what a cost model implies for a set of contexts.

    Used to report the spread of b, which is the whole substance of open
    decision #2: under "proportional" the spread is exactly zero.
    """
    b = np.asarray(model.break_even(context_tokens), dtype=float)
    qs = [1, 25, 50, 75, 99]
    out = {
        "c_mode": model.c_mode,
        "c0": model.c0,
        "break_even_mean_s": float(b.mean()),
        "break_even_std_s": float(b.std()),
        "break_even_spread_ratio": float(b.max() / b.min()) if b.min() > 0 else float("inf"),
    }
    for q, v in zip(qs, np.percentile(b, qs)):
        out[f"break_even_p{q}_s"] = float(v)
    return out
