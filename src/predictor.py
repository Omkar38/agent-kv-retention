"""Duration predictions, and deliberate ways of making them worse.

Week 4 established what a predictor can do on clean held-out sessions. The
research question is what happens when it stops being clean, so prediction
quality is treated here as an independent variable rather than a fixed input.

The error models span the range named in the proposal:

    clean          the predictor's own held-out output
    oracle         T itself; the upper bound on what prediction can buy
    noise          multiplicative lognormal jitter, severity in dex
    under / over   systematic bias, That = alpha * T
    shift          train on one provider, test on the other
    tail_heavy     resample long pauses to over-represent the tail
    adversarial    flip the one bit the policy reads, as often as allowed

`adversarial` is the important one. The lambda-policy reads the prediction
only through whether That >= b, so an adversary does not need large errors --
it needs the bit wrong at exactly the durations where being wrong is most
expensive. A predictor can have excellent MAE and be fully adversarial in this
sense, which is the point of reporting flip rate rather than MAE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_error(
    true_duration_s: np.ndarray,
    predicted_duration_s: np.ndarray,
    break_even_s: np.ndarray,
    mode: str,
    severity: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a degraded prediction. Never modifies the true durations.

    `severity` means different things per mode and is documented at each branch.
    All modes are deterministic given `rng`.
    """
    rng = rng or np.random.default_rng(560)
    T = np.asarray(true_duration_s, dtype=float)
    P = np.asarray(predicted_duration_s, dtype=float)
    b = np.asarray(break_even_s, dtype=float)

    if mode == "clean":
        return P.copy()

    if mode == "oracle":
        return T.copy()

    if mode == "noise":
        # severity in dex: multiplicative lognormal, sigma = severity decades.
        return P * np.exp(rng.normal(0.0, severity * np.log(10.0), size=P.shape))

    if mode in ("under", "over"):
        # severity is alpha directly: That = alpha * T. Bias relative to truth,
        # so the failure is systematic rather than merely noisy.
        return T * severity

    if mode == "adversarial":
        # Flip the bit the policy reads, on a `severity` fraction of pauses,
        # chosen where flipping costs most rather than at random.
        #
        # Under-calling a long pause is the expensive direction: the policy
        # holds b/lambda and still pays B. Over-calling a short pause wastes
        # at most B. So we prioritise flipping true-long pauses to "short".
        flipped = P.copy()
        n = P.size
        k = int(round(severity * n))
        if k <= 0:
            return flipped
        # Rank by how costly a flip would be: long pauses first, then by length.
        is_long = T >= b
        harm = np.where(is_long, T, -T)
        victims = np.argsort(-harm)[:k]
        # Send long pauses to just under b, short pauses to just over b.
        flipped[victims] = np.where(
            is_long[victims], b[victims] * 0.999, b[victims] * 1.001
        )
        return flipped

    raise ValueError(f"unknown error mode {mode!r}")


def tail_heavy_resample(
    frame: pd.DataFrame,
    duration_col: str = "pause_s",
    quantile: float = 0.9,
    factor: float = 3.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Over-represent the tail, keeping whole sessions intact.

    Resampling individual pauses would break the session structure the
    simulator replays, so sessions are duplicated as units based on whether
    they contain a long pause.
    """
    rng = rng or np.random.default_rng(560)
    threshold = frame[duration_col].quantile(quantile)
    has_tail = frame.groupby("session_id")[duration_col].max() >= threshold
    tail_sessions = has_tail[has_tail].index.to_numpy()
    if tail_sessions.size == 0:
        return frame.copy()
    extra = rng.choice(tail_sessions, size=int(tail_sessions.size * (factor - 1)), replace=True)
    parts = [frame]
    for i, sid in enumerate(extra):
        dup = frame[frame.session_id == sid].copy()
        dup["session_id"] = f"{sid}__tailcopy{i}"
        parts.append(dup)
    return pd.concat(parts, ignore_index=True)


def decision_flip_rate(
    true_duration_s: np.ndarray,
    predicted_duration_s: np.ndarray,
    break_even_s: np.ndarray,
) -> dict:
    """How often the prediction gets the only bit the policy reads wrong.

    Split by direction, because the two directions have very different costs:
    under-calling a long pause means holding b/lambda *and* paying B.
    """
    T = np.asarray(true_duration_s, dtype=float)
    P = np.asarray(predicted_duration_s, dtype=float)
    b = np.asarray(break_even_s, dtype=float)
    true_long = T >= b
    pred_long = P >= b
    flip = true_long != pred_long
    return {
        "flip_rate": float(flip.mean()),
        "under_called_long": float((true_long & ~pred_long).mean()),
        "over_called_short": float((~true_long & pred_long).mean()),
        "true_long_fraction": float(true_long.mean()),
    }
