"""Decision-time features the Week 4 model never used.

Week 4 fed the predictor a description of *this* tool call: who is calling,
which tool, how big the arguments and the context are. It never told the model
what had already happened in the session. That turns out to be the strongest
signal in the dataset:

    lag-1 autocorrelation of log1p(pause)        0.396
    P(long | previous pause was long)            37.5%
    P(long | previous pause was short)            4.1%
    base rate                                     6.1%

A pause that follows a long pause is **6.15x** more likely to be long. The
information was present, leakage-free, and simply not supplied.

**Why this is legitimate.** Every feature here is computed from rows strictly
earlier in the same session. When round k pauses, round k-1's pause has
already completed, so its duration is observable. Nothing reads the current
row's outcome or any future row. `build_history_features` asserts the shift
discipline rather than relying on the caller to get it right.

**Why `user` is included but flagged.** Long-pause rate ranges from 1.9% to
27.1% across the 43 users -- a 14x spread, and a real effect: different
developers run different test suites on different machines. A serving system
does know which tenant a request belongs to, so conditioning on it is fair.
But sessions from the same user appear in both train and test here, so the
measured benefit is an upper bound on what a *new* user would get. The
ablation in `experiments/run_features.py` reports with and without it.

**What is not fixable by features.** Within identical
(provider, tool, n_tools, input-size-decile) groups, the p90/p10 duration
ratio is still 67x. Some of the spread is irreducible given anything knowable
at decision time -- the same `pytest` invocation genuinely takes 2 s or 200 s
depending on what changed. Session history helps because it is a proxy for
that hidden state, not because it removes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features describing this call -- the Week 4 set.
BASE_CATEGORICAL = ["provider", "model", "tool_signature"]
BASE_NUMERIC = [
    "feature_n_tools",
    "feature_log1p_context_tokens",
    "feature_log1p_round_index",
    "feature_log1p_tool_input_chars",
    "feature_has_tools",
    "feature_multiple_tools",
]

# Features describing what already happened in this session.
HISTORY_NUMERIC = [
    "hist_prev_log_pause",
    "hist_prev2_log_pause",
    "hist_mean_log_pause",
    "hist_max_log_pause",
    "hist_prev_was_long",
    "hist_long_count",
    "hist_long_rate",
    "hist_n_prior",
    "hist_elapsed_log_s",
    "hist_context_growth_log",
    "hist_has_history",
]

USER_CATEGORICAL = ["user"]
CLOCK_NUMERIC = ["clock_hour_sin", "clock_hour_cos"]


def build_history_features(
    frame: pd.DataFrame,
    break_even_s: np.ndarray | None = None,
) -> pd.DataFrame:
    """Attach session-history features, computed only from earlier rows.

    `break_even_s` enables the long-pause history features. It depends on the
    cost model, so it is passed in rather than assumed.

    Rows are sorted by (session_id, round_index) and every history column is
    produced by a shift or an expanding window over prior rows only.
    """
    out = frame.sort_values(["session_id", "round_index"]).copy()
    grouped = out.groupby("session_id", sort=False)

    log_pause = np.log1p(out["pause_s"].astype(float))
    out["_log_pause"] = log_pause

    # Durations of earlier pauses in this session.
    out["hist_prev_log_pause"] = grouped["_log_pause"].shift(1)
    out["hist_prev2_log_pause"] = grouped["_log_pause"].shift(2)

    # Expanding statistics over strictly-prior rows: shift first, then expand,
    # so the current row can never enter its own summary.
    prior = grouped["_log_pause"].shift(1)
    out["_prior"] = prior
    out["hist_mean_log_pause"] = out.groupby("session_id")["_prior"].transform(
        lambda s: s.expanding().mean()
    )
    out["hist_max_log_pause"] = out.groupby("session_id")["_prior"].transform(
        lambda s: s.expanding().max()
    )
    out["hist_n_prior"] = grouped.cumcount().astype(float)

    if break_even_s is not None:
        b = pd.Series(np.asarray(break_even_s, dtype=float), index=frame.index).reindex(out.index)
        is_long = (out["pause_s"].astype(float) >= b).astype(float)
        out["_is_long"] = is_long
        out["hist_prev_was_long"] = out.groupby("session_id")["_is_long"].shift(1)
        prior_long = out.groupby("session_id")["_is_long"].shift(1)
        out["_prior_long"] = prior_long
        out["hist_long_count"] = out.groupby("session_id")["_prior_long"].transform(
            lambda s: s.expanding().sum()
        )
        out["hist_long_rate"] = out["hist_long_count"] / out["hist_n_prior"].replace(0, np.nan)
        out = out.drop(columns=["_is_long", "_prior_long"])
    else:
        for column in ("hist_prev_was_long", "hist_long_count", "hist_long_rate"):
            out[column] = np.nan

    # How far into the session we are, in wall-clock terms.
    start = pd.to_datetime(out["pause_start_at"])
    session_start = out.groupby("session_id")["pause_start_at"].transform("min")
    elapsed = (start - pd.to_datetime(session_start)).dt.total_seconds().clip(lower=0)
    out["hist_elapsed_log_s"] = np.log1p(elapsed)

    # Context growth since the previous round: a proxy for how much the last
    # tool call returned, without reading this call's result.
    log_ctx = np.log1p(out["context_tokens"].astype(float))
    out["hist_context_growth_log"] = log_ctx - out.groupby("session_id")["context_tokens"].shift(1).pipe(
        lambda s: np.log1p(s.astype(float))
    )

    out["hist_has_history"] = (out["hist_n_prior"] > 0).astype(float)

    # Time of day, encoded cyclically so 23:00 and 00:00 are adjacent.
    hour = start.dt.hour + start.dt.minute / 60.0
    out["clock_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["clock_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    out = out.drop(columns=["_log_pause", "_prior"])
    _assert_no_leakage(out)
    return out


def _assert_no_leakage(frame: pd.DataFrame) -> None:
    """The first pause of every session must have no history.

    If a shift were mis-specified -- expanding before shifting, or forgetting
    to group -- the first row of a session would carry a value. This is the
    cheapest check that catches it, and it runs on every build.
    """
    first = frame.groupby("session_id").head(1)
    for column in ("hist_prev_log_pause", "hist_prev2_log_pause", "hist_mean_log_pause"):
        leaked = first[column].notna().sum()
        if leaked:
            raise AssertionError(
                f"{column} is populated on {leaked} session-opening rows; "
                "history must be empty before any pause has completed"
            )
    if (frame.groupby("session_id").head(1)["hist_n_prior"] != 0).any():
        raise AssertionError("hist_n_prior must be 0 on the first row of each session")


def feature_sets() -> dict[str, list[str]]:
    """The ablation ladder, from the Week 4 set to everything available."""
    base = BASE_CATEGORICAL + BASE_NUMERIC
    return {
        "week4_base": base,
        "plus_history": base + HISTORY_NUMERIC,
        "plus_history_clock": base + HISTORY_NUMERIC + CLOCK_NUMERIC,
        "plus_history_clock_user": base + HISTORY_NUMERIC + CLOCK_NUMERIC + USER_CATEGORICAL,
    }


def categoricals_in(columns: list[str]) -> list[str]:
    return [c for c in columns if c in BASE_CATEGORICAL + USER_CATEGORICAL]


def numerics_in(columns: list[str]) -> list[str]:
    return [c for c in columns if c not in BASE_CATEGORICAL + USER_CATEGORICAL]
