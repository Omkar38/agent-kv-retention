"""Tests for the decision-target classifier and its acting threshold."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.classifier import (
    FEATURES,
    bit_quality,
    cost_optimal_threshold,
    fit_bit_predictor,
    normalised_cost,
    reliability,
    select_threshold,
)
from src.costs import CostModel


class TestNormalisedCost(unittest.TestCase):
    def test_matches_the_dimensional_cost_model(self):
        """g(u) = C(tau, T) / B when u = T/b and tau/b is the same ratio."""
        cost = CostModel(c_mode="fixed")
        ctx = 122_000.0
        b = float(cost.break_even(ctx))
        B = float(cost.B(ctx))
        for lam in (0.25, 0.5, 0.75):
            for T in (0.1 * b, 0.9 * b, 1.5 * b, 10 * b):
                dimensional = float(cost.cost(lam * b, T, ctx)) / B
                dimensionless = float(normalised_cost(T / b, lam))
                self.assertAlmostEqual(dimensional, dimensionless, places=9)

    def test_clairvoyant_is_min_u_one(self):
        cost = CostModel()
        ctx = 122_000.0
        b = float(cost.break_even(ctx))
        B = float(cost.B(ctx))
        for T in (0.3 * b, 5 * b):
            self.assertAlmostEqual(
                float(cost.clairvoyant(T, ctx)) / B, min(T / b, 1.0), places=9
            )


class TestThreshold(unittest.TestCase):
    def test_threshold_is_well_below_one_half(self):
        """Over-calling is cheap and under-calling is not, so p* << 0.5."""
        rng = np.random.default_rng(0)
        # Heavy-tailed, mostly short -- the shape of the real workload.
        u = np.exp(rng.normal(-2.0, 1.6, size=20_000))
        for lam in (0.25, 0.5, 0.75):
            out = cost_optimal_threshold(u, lam)
            self.assertGreater(out["threshold"], 0.0)
            self.assertLess(out["threshold"], 0.5)

    def test_long_and_short_gaps_have_opposite_signs(self):
        rng = np.random.default_rng(1)
        u = np.exp(rng.normal(-2.0, 1.6, size=20_000))
        out = cost_optimal_threshold(u, 0.5)
        self.assertLess(out["delta_long"], 0.0)   # calling long helps long pauses
        self.assertGreater(out["delta_short"], 0.0)  # and hurts short ones

    def test_degenerate_input_falls_back_rather_than_dividing_by_zero(self):
        all_short = np.full(100, 0.1)
        self.assertEqual(cost_optimal_threshold(all_short, 0.5)["threshold"], 0.5)

    def test_select_threshold_returns_a_grid_point_and_a_curve(self):
        rng = np.random.default_rng(2)
        n = 2000
        ctx = rng.uniform(10_000, 200_000, n)
        cost = CostModel()
        b = np.asarray(cost.break_even(ctx))
        T = b * np.exp(rng.normal(-1.5, 1.5, n))
        prob = rng.uniform(0, 1, n)
        out = select_threshold(prob, T, ctx, cost, lam=0.5)
        self.assertIn("curve", out)
        self.assertGreater(len(out["curve"]), 10)
        self.assertGreaterEqual(out["threshold"], 0.0)
        self.assertLessEqual(out["threshold"], 1.0)


class TestBitQualityAndReliability(unittest.TestCase):
    def test_bit_quality_on_a_known_case(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
        q = bit_quality(y, p)
        self.assertAlmostEqual(q["recall"], 2 / 3)
        self.assertAlmostEqual(q["precision"], 2 / 3)
        self.assertAlmostEqual(q["flip_rate"], 2 / 8)

    def test_reliability_bins_sum_to_the_sample(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype(int)
        table = reliability(y, p, bins=10)
        self.assertEqual(int(table.n.sum()), 5000)
        # A perfectly calibrated generator should track the diagonal closely.
        err = (table.mean_predicted - table.observed_frequency).abs().max()
        self.assertLess(err, 0.1)


class TestFitBitPredictor(unittest.TestCase):
    def _frame(self, n=1200, seed=4):
        rng = np.random.default_rng(seed)
        sessions = [f"s{i // 12}" for i in range(n)]
        ctx = rng.uniform(20_000, 250_000, n)
        # A learnable signal: one tool is systematically slow.
        tool = rng.choice(["Bash", "Read", "Edit"], n)
        base = np.where(tool == "Bash", 1.2, -1.8)
        T = np.exp(rng.normal(base, 1.0, n))
        return pd.DataFrame(
            {
                "session_id": sessions,
                "pause_s": T,
                "provider": "claude",
                "model": "m",
                "tool_signature": tool,
                "feature_n_tools": rng.integers(1, 4, n).astype(float),
                "feature_log1p_context_tokens": np.log1p(ctx),
                "feature_log1p_round_index": np.log1p(rng.integers(0, 50, n)),
                "feature_log1p_tool_input_chars": np.log1p(rng.uniform(10, 5000, n)),
                "feature_has_tools": 1.0,
                "feature_multiple_tools": 0.0,
                "context_tokens": ctx,
            }
        )

    def test_produces_probabilities_and_a_usable_threshold(self):
        frame = self._frame()
        cost = CostModel()
        b = np.asarray(cost.break_even(frame.context_tokens.to_numpy()))
        fitted = fit_bit_predictor(frame, b, lam=0.5, seed=560)

        p = fitted.probability(frame)
        self.assertEqual(p.shape, (len(frame),))
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(fitted.threshold, 0.0)
        self.assertLess(fitted.threshold, 1.0)

    def test_learns_the_planted_signal(self):
        """Bash is slow by construction, so it must score higher than Read."""
        frame = self._frame()
        cost = CostModel()
        b = np.asarray(cost.break_even(frame.context_tokens.to_numpy()))
        fitted = fit_bit_predictor(frame, b, lam=0.5, seed=560)
        p = fitted.probability(frame)
        self.assertGreater(
            p[frame.tool_signature == "Bash"].mean(),
            p[frame.tool_signature == "Read"].mean(),
        )

    def test_uses_only_whitelisted_features(self):
        """The target and any outcome column must never reach the model."""
        for banned in ("pause_s", "context_tokens", "session_id"):
            self.assertNotIn(banned, FEATURES)


if __name__ == "__main__":
    unittest.main()
