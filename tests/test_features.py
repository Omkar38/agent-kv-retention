"""Tests for the session-history features.

These features are the only ones in the project computed *across* rows, which
makes them the only place a subtle leak can hide: expand before shifting, or
forget to group by session, and the model silently reads its own answer. Most
of what follows is aimed at that.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.costs import CostModel
from src.features import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    HISTORY_NUMERIC,
    build_history_features,
    categoricals_in,
    feature_sets,
    numerics_in,
)


def frame(n_sessions=4, n_rounds=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2026-01-01")
    for s in range(n_sessions):
        for r in range(n_rounds):
            rows.append(
                {
                    "session_id": f"s{s}",
                    "round_index": r,
                    "pause_s": float(rng.uniform(0.1, 50.0)),
                    "context_tokens": float(rng.uniform(10_000, 200_000)),
                    "pause_start_at": t0 + pd.Timedelta(seconds=s * 1000 + r * 60),
                    "provider": "claude",
                    "model": "m",
                    "tool_signature": "Bash",
                    "user": f"u{s % 2}",
                }
            )
    return pd.DataFrame(rows)


class TestNoLeakage(unittest.TestCase):
    def test_session_opening_rows_have_no_history(self):
        out = build_history_features(frame())
        first = out.groupby("session_id").head(1)
        for column in ("hist_prev_log_pause", "hist_prev2_log_pause", "hist_mean_log_pause"):
            self.assertTrue(first[column].isna().all(), f"{column} leaked into a session's first row")
        self.assertTrue((first["hist_n_prior"] == 0).all())
        self.assertTrue((first["hist_has_history"] == 0).all())

    def test_prev_equals_the_actual_previous_pause(self):
        out = build_history_features(frame()).sort_values(["session_id", "round_index"])
        for _, group in out.groupby("session_id"):
            pauses = group["pause_s"].to_numpy()
            prev = group["hist_prev_log_pause"].to_numpy()
            np.testing.assert_allclose(prev[1:], np.log1p(pauses[:-1]), rtol=1e-12)

    def test_history_never_crosses_sessions(self):
        """Session B's first row must not inherit session A's last pause."""
        out = build_history_features(frame(n_sessions=3, n_rounds=4))
        for _, group in out.groupby("session_id"):
            self.assertTrue(np.isnan(group.iloc[0]["hist_prev_log_pause"]))

    def test_expanding_mean_excludes_the_current_row(self):
        out = build_history_features(frame(n_sessions=1, n_rounds=5)).sort_values("round_index")
        pauses = np.log1p(out["pause_s"].to_numpy())
        means = out["hist_mean_log_pause"].to_numpy()
        for k in range(1, len(out)):
            self.assertAlmostEqual(means[k], pauses[:k].mean(), places=10)

    def test_long_history_uses_only_prior_rows(self):
        f = frame(n_sessions=1, n_rounds=6)
        cost = CostModel()
        b = np.asarray(cost.break_even(f["context_tokens"].to_numpy()))
        out = build_history_features(f, break_even_s=b).sort_values("round_index")
        is_long = (out["pause_s"].to_numpy() >= np.asarray(cost.break_even(out["context_tokens"].to_numpy()))).astype(float)
        counts = out["hist_long_count"].to_numpy()
        for k in range(1, len(out)):
            self.assertAlmostEqual(counts[k], is_long[:k].sum(), places=10)

    def test_assertion_fires_when_history_is_mis_shifted(self):
        """The guard must actually catch a broken build, not just pass silently."""
        out = build_history_features(frame())
        broken = out.copy()
        first_index = broken.groupby("session_id").head(1).index[0]
        broken.loc[first_index, "hist_prev_log_pause"] = 1.23
        from src.features import _assert_no_leakage

        with self.assertRaises(AssertionError):
            _assert_no_leakage(broken)


class TestFeatureValues(unittest.TestCase):
    def test_elapsed_time_starts_at_zero_and_increases(self):
        out = build_history_features(frame(n_sessions=1, n_rounds=5)).sort_values("round_index")
        elapsed = out["hist_elapsed_log_s"].to_numpy()
        self.assertAlmostEqual(elapsed[0], 0.0, places=10)
        self.assertTrue(np.all(np.diff(elapsed) > 0))

    def test_clock_encoding_is_cyclic(self):
        out = build_history_features(frame())
        radius = out["clock_hour_sin"] ** 2 + out["clock_hour_cos"] ** 2
        np.testing.assert_allclose(radius.to_numpy(), 1.0, atol=1e-9)

    def test_all_declared_history_columns_are_produced(self):
        f = frame()
        cost = CostModel()
        out = build_history_features(f, break_even_s=np.asarray(cost.break_even(f["context_tokens"].to_numpy())))
        for column in HISTORY_NUMERIC:
            self.assertIn(column, out.columns)

    def test_row_count_is_preserved(self):
        f = frame(n_sessions=5, n_rounds=7)
        self.assertEqual(len(build_history_features(f)), len(f))


class TestFeatureSets(unittest.TestCase):
    def test_ladder_is_strictly_nested(self):
        sets = feature_sets()
        order = ["week4_base", "plus_history", "plus_history_clock", "plus_history_clock_user"]
        for earlier, later in zip(order, order[1:]):
            self.assertTrue(set(sets[earlier]).issubset(set(sets[later])))

    def test_base_set_matches_week4(self):
        self.assertEqual(set(feature_sets()["week4_base"]), set(BASE_CATEGORICAL + BASE_NUMERIC))

    def test_no_outcome_column_is_ever_a_feature(self):
        banned = {"pause_s", "endpoint_duration_s", "pause_end_at", "target_log1p_pause_s",
                  "label_pause_gt_10s", "label_pause_gt_60s", "analysis_track", "split"}
        for name, columns in feature_sets().items():
            self.assertFalse(banned & set(columns), f"{name} contains an outcome column")

    def test_split_of_categorical_and_numeric_is_exhaustive(self):
        for columns in feature_sets().values():
            self.assertEqual(
                sorted(categoricals_in(columns) + numerics_in(columns)), sorted(columns)
            )


if __name__ == "__main__":
    unittest.main()
