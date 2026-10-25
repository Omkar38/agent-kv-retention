"""Assemble the evaluation table: held-out pauses joined to their predictions.

The Week 4 pipeline emits per-pause predictions but not the context length,
because the predictor does not need it. The retention policies do -- B scales
with context -- so the two are rejoined here on the canonical row index.

Everything downstream consumes the frame this module returns, so the join and
the session-level split live in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATASET = Path("analysis/dataset/pause_handoff.parquet")
PREDICTIONS = Path("results/week4_handoff/predictions_test.csv")

PREDICTION_COLUMNS = {
    "hgb_q50": "pred_hgb_q50_s",
    "provider_tool_median": "pred_provider_tool_median_s",
    "global_median": "pred_global_median_s",
    "log_ridge": "pred_log_ridge_s",
    # Upper and lower quantiles of the same gradient-boosted model.
    #
    # Week 4 selected q50 by validation log1p-MAE. That objective is dominated
    # by the ~95% of pauses that are short, while the retention decision turns
    # entirely on the ~4.5% that are long. Feeding the policy an upper quantile
    # trades precision for recall on exactly that bit, which is the trade the
    # cost asymmetry favours: over-calling a short pause is nearly free, since
    # it returns before any deadline, whereas under-calling a long one means
    # holding b/lambda *and* paying B anyway.
    "hgb_q90": "q90_s",
    "hgb_q10": "q10_s",
}


def load_pauses(dataset: Path = DATASET, split: str | None = None) -> pd.DataFrame:
    """The supervised tool-pause cohort, optionally restricted to one split.

    `analysis_track` is a retrospective label used to define the offline
    cohort. It is never a model input and never reaches a policy -- it only
    decides which rows are scorable at all.
    """
    frame = pd.read_parquet(dataset)
    frame = frame[
        (frame["analysis_track"] == "tool_pause_prediction")
        & frame["pause_s"].notna()
        & (frame["pause_s"] > 0)
        & np.isfinite(frame["pause_s"])
    ].copy()
    if split is not None:
        frame = frame[frame["split"] == split].copy()
    return frame.reset_index(drop=True)


def load_evaluation_frame(
    dataset: Path = DATASET,
    predictions: Path = PREDICTIONS,
    model: str = "hgb_q50",
) -> pd.DataFrame:
    """Held-out pauses with a prediction attached, ready for scoring.

    Raises if the predictions file is missing rather than silently falling back
    to a weaker predictor, because a quiet substitution would change every
    downstream number.
    """
    if not Path(predictions).exists():
        raise FileNotFoundError(
            f"{predictions} not found. Regenerate it with:\n"
            f"    bash week4/run_week4.sh"
        )
    if model not in PREDICTION_COLUMNS:
        raise ValueError(f"model must be one of {sorted(PREDICTION_COLUMNS)}, got {model!r}")

    preds = pd.read_csv(predictions)
    full = pd.read_parquet(dataset)

    # source_row_index indexes the untouched 357,161-row dataset.
    context = (
        full[["context_tokens", "pause_start_at", "pause_end_at", "provider"]]
        .reset_index()
        .rename(columns={"index": "source_row_index"})
    )
    merged = preds.merge(context, on="source_row_index", how="left", validate="one_to_one")

    missing = merged["context_tokens"].isna().sum()
    if missing:
        raise ValueError(f"{missing} predictions failed to join to a context length")

    out = pd.DataFrame(
        {
            "session_id": merged["session_id"],
            "round_index": merged["round_index"].astype(int),
            "provider": merged["provider_x"].fillna(merged["provider_y"]),
            "tool_key": merged["provider_x"].astype(str) + "|" + merged["tool_signature"].astype(str),
            "context_tokens": merged["context_tokens"].astype(float),
            "pause_s": merged["target_s"].astype(float),
            "predicted_duration_s": merged[PREDICTION_COLUMNS[model]].astype(float),
            "pause_start_at": merged["pause_start_at"],
            "pause_end_at": merged["pause_end_at"],
        }
    )
    out = out[np.isfinite(out["pause_s"]) & (out["pause_s"] > 0)]
    out["predicted_duration_s"] = out["predicted_duration_s"].clip(lower=0.0)
    return out.sort_values(["session_id", "round_index"]).reset_index(drop=True)


def training_durations(dataset: Path = DATASET) -> pd.DataFrame:
    """Per-tool durations from the *training* split, for fitting Continuum-style.

    Fitting the distribution on anything but the training sessions would leak
    held-out outcomes into a baseline and make the comparison meaningless.
    """
    train = load_pauses(dataset, split="train")
    tool = train["feature_tool_names"].fillna("none").astype(str)
    return pd.DataFrame(
        {
            "tool_key": train["provider"].astype(str) + "|" + tool,
            "pause_s": train["pause_s"].astype(float),
        }
    )
