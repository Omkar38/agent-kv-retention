"""A predictor trained on the decision the policy actually makes.

Week 4 trained regressors to predict pause *duration* and selected among them
by validation log1p-MAE. The retention policy never consumes a duration. It
consumes one bit: whether the pause will outlast the break-even time `b = B/c`.
Those are different problems, and optimising the first does badly at the
second -- `hgb_q50` reaches 22.3% recall on long pauses because log error is
dominated by the 95% of pauses that are short.

This module trains directly on `1[T >= b]`, with the same decision-time
feature whitelist and the same session-level splits, so the comparison isolates
the change of target.

Two things follow from training on the bit rather than the duration.

**Calibration becomes the headline, not accuracy.** The policy needs a
probability it can act on, so the model is isotonically calibrated on a
held-out slice of the training sessions and reported with a reliability curve.

**The threshold is derived, not set to 0.5.** Over-calling a short pause is
nearly free -- it returns before any deadline -- while under-calling a long one
means holding `b/λ` and paying `B` anyway. A symmetric threshold ignores that.

Working in normalised time `u = T/b` makes the cost dimensionless and the
derivation clean. Dividing `C(τ,T) = c·min(T,τ) + B·1[T>τ]` by `B = c·b`:

    call long  (τ = λb):    g_A(u) = min(u, λ)   + 1[u > λ]
    call short (τ = b/λ):   g_B(u) = min(u, 1/λ) + 1[u > 1/λ]
    clairvoyant:            min(u, 1)

With `p = P(u >= 1)` and the conditional expectations taken over the training
distribution of `u`:

    ΔL = E[g_A - g_B | u >= 1]      (negative: calling long helps long pauses)
    ΔS = E[g_A - g_B | u <  1]      (positive: calling long hurts short ones)

Call long iff `p·ΔL + (1-p)·ΔS < 0`, i.e. iff

    p  >  p* = ΔS / (ΔS - ΔL)

`p*` depends only on λ and the shape of the duration distribution in units of
`b`. On this workload it lands far below 0.5, which is the cost asymmetry
expressed as a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The Week 4 whitelist, unchanged, so the comparison isolates the target.
CATEGORICAL = ["provider", "model", "tool_signature"]
NUMERIC = [
    "feature_n_tools",
    "feature_log1p_context_tokens",
    "feature_log1p_round_index",
    "feature_log1p_tool_input_chars",
    "feature_has_tools",
    "feature_multiple_tools",
]
FEATURES = CATEGORICAL + NUMERIC


def normalised_cost(u: np.ndarray, tau_over_b: float) -> np.ndarray:
    """g(u) = min(u, τ/b) + 1[u > τ/b]: cost in units of B."""
    u = np.asarray(u, dtype=float)
    return np.minimum(u, tau_over_b) + (u > tau_over_b).astype(float)


def cost_optimal_threshold(u_train: np.ndarray, lam: float) -> dict:
    """The probability above which calling a pause 'long' lowers expected cost.

    Derived from the training distribution of `u = T/b`; see module docstring.
    Returns the threshold alongside the two conditional gaps, because those are
    what make it asymmetric and are worth reporting.
    """
    u = np.asarray(u_train, dtype=float)
    u = u[np.isfinite(u) & (u >= 0)]
    long_mask = u >= 1.0

    g_a = normalised_cost(u, lam)
    g_b = normalised_cost(u, 1.0 / lam)
    delta = g_a - g_b

    if not long_mask.any() or long_mask.all():
        return {"threshold": 0.5, "delta_long": float("nan"), "delta_short": float("nan")}

    delta_long = float(delta[long_mask].mean())
    delta_short = float(delta[~long_mask].mean())
    denom = delta_short - delta_long
    threshold = float(delta_short / denom) if denom > 0 else 0.5
    return {
        "threshold": float(np.clip(threshold, 0.0, 1.0)),
        "delta_long": delta_long,
        "delta_short": delta_short,
        "long_base_rate": float(long_mask.mean()),
    }


def build_classifier(seed: int = 560, iterations: int = 200) -> Pipeline:
    """Gradient-boosted classifier over the Week 4 decision-time features."""
    prep = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
                CATEGORICAL,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
        ],
        sparse_threshold=0,
    )
    model = HistGradientBoostingClassifier(
        max_iter=iterations,
        max_leaf_nodes=31,
        learning_rate=0.06,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=seed,
    )
    return Pipeline([("preprocess", prep), ("estimator", model)])


@dataclass
class FittedBitPredictor:
    """A calibrated P(T >= b) model plus the threshold that should act on it."""

    pipeline: object
    threshold: float
    threshold_detail: dict
    lam: float

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[FEATURES])[:, 1]

    def predicts_long(self, frame: pd.DataFrame) -> np.ndarray:
        return self.probability(frame) >= self.threshold


def fit_bit_predictor(
    train: pd.DataFrame,
    break_even_train: np.ndarray,
    lam: float,
    seed: int = 560,
    calibration_fraction: float = 0.25,
) -> FittedBitPredictor:
    """Train and calibrate P(T >= b), and derive the acting threshold.

    The calibration slice is held out **by session**, like every other split in
    this project: calibrating on rows drawn from sessions the model trained on
    would leak later turns of those sessions and flatter the reliability curve.
    """
    train = train.reset_index(drop=True)
    b = np.asarray(break_even_train, dtype=float)
    y = (train["pause_s"].to_numpy(dtype=float) >= b).astype(int)
    u = train["pause_s"].to_numpy(dtype=float) / b

    sessions = train["session_id"].to_numpy()
    unique = pd.unique(sessions)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(unique, size=max(1, int(len(unique) * calibration_fraction)), replace=False))
    is_cal = np.array([s in held for s in sessions])

    if y[~is_cal].sum() == 0 or y[is_cal].sum() == 0:
        # Degenerate split (no positives on one side): fall back to fitting and
        # calibrating on everything rather than producing a silently broken model.
        is_cal = np.zeros(len(train), dtype=bool)

    base = build_classifier(seed=seed)
    fit_rows = train if not is_cal.any() else train[~is_cal]
    fit_y = y if not is_cal.any() else y[~is_cal]
    base.fit(fit_rows[FEATURES], fit_y)

    if is_cal.any():
        # FrozenEstimator replaces the deprecated cv="prefit"; fall back for
        # older scikit-learn so the pipeline runs on the CI image too.
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
        except ImportError:  # pragma: no cover - scikit-learn < 1.6
            calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        calibrated.fit(train[is_cal][FEATURES], y[is_cal])
        pipeline = calibrated
    else:
        pipeline = base

    detail = cost_optimal_threshold(u, lam)
    return FittedBitPredictor(
        pipeline=pipeline,
        threshold=detail["threshold"],
        threshold_detail=detail,
        lam=lam,
    )


def select_threshold(
    probability: np.ndarray,
    true_duration_s: np.ndarray,
    context_tokens: np.ndarray,
    cost,
    lam: float,
    grid: np.ndarray | None = None,
) -> dict:
    """Pick the acting threshold on held-out *validation* sessions.

    `cost_optimal_threshold` derives a threshold analytically from the marginal
    distribution of `u`, which gets the direction right -- far below 0.5 -- but
    not the magnitude, because the relevant conditional distribution is the one
    for pauses the model is unsure about, not the population as a whole. It
    over-corrects: on this workload it calls ~27% of pauses long when 4.5% are.

    So the threshold is chosen empirically instead, by minimising realised cost
    on validation sessions the model never saw. The analytical value is kept
    alongside it as a sanity check on the direction.
    """
    from .policies import make_lambda_policy
    from .evaluate import taus_for

    grid = grid if grid is not None else np.round(np.arange(0.02, 0.86, 0.02), 3)
    T = np.asarray(true_duration_s, dtype=float)
    ctx = np.asarray(context_tokens, dtype=float)
    b = np.asarray(cost.break_even(ctx), dtype=float)
    opt = float(cost.clairvoyant(T, ctx).sum())
    policy = make_lambda_policy(cost, lam)

    rows = []
    for threshold in grid:
        frame = pd.DataFrame(
            {
                "predicted_duration_s": np.where(probability >= threshold, b * 2.0, 0.0),
                "context_tokens": ctx,
                "tool_key": "x",
                "pause_s": T,
            }
        )
        total = float(cost.cost(taus_for(policy, frame), T, ctx).sum())
        rows.append({"threshold": float(threshold), "total_ratio": total / opt})

    table = pd.DataFrame(rows)
    best = table.loc[table.total_ratio.idxmin()]
    return {
        "threshold": float(best.threshold),
        "validation_total_ratio": float(best.total_ratio),
        "curve": table,
    }


def bit_quality(y_true: np.ndarray, predicted_long: np.ndarray) -> dict:
    """Recall, precision and flip rate on the bit the policy reads."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(predicted_long).astype(bool)
    tp = int((y & p).sum())
    return {
        "recall": tp / max(int(y.sum()), 1),
        "precision": tp / max(int(p.sum()), 1),
        "flip_rate": float((y != p).mean()),
        "predicted_long_rate": float(p.mean()),
        "true_long_rate": float(y.mean()),
    }


def reliability(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Calibration curve: predicted probability against observed frequency."""
    y = np.asarray(y_true).astype(float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        m = idx == k
        if not m.any():
            continue
        rows.append(
            {
                "bin_lower": float(edges[k]),
                "bin_upper": float(edges[k + 1]),
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_frequency": float(y[m].mean()),
            }
        )
    return pd.DataFrame(rows)
