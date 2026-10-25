"""Tests for the cost model, the policies, and the simulator.

The emphasis is on the properties that would silently corrupt a result rather
than crash: a policy reading the future, lambda pointing the wrong way, the
bound not binding where theory says it must.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.costs import CostModel
from src.evaluate import worst_case_probe
from src.policies import (
    Decision,
    always_retain,
    immediate_evict,
    make_break_even_ttl,
    make_clairvoyant,
    make_fixed_ttl,
    make_lambda_policy,
)
from src.sim import run_one


class TestCostModel(unittest.TestCase):
    def test_break_even_is_B_over_c(self):
        for mode in ("fixed", "proportional"):
            cost = CostModel(c_mode=mode)
            ctx = np.array([10_000.0, 122_000.0, 500_000.0])
            np.testing.assert_allclose(
                cost.break_even(ctx), cost.B(ctx) / cost.c(ctx), rtol=1e-12
            )

    def test_proportional_mode_gives_one_break_even_for_everyone(self):
        """Open decision #2: under proportional c, B and c cancel exactly."""
        cost = CostModel(c_mode="proportional")
        b = cost.break_even(np.array([1_000.0, 122_000.0, 5_000_000.0]))
        self.assertAlmostEqual(float(b.min()), float(b.max()), places=9)

    def test_fixed_mode_break_even_scales_with_context(self):
        cost = CostModel(c_mode="fixed")
        b = cost.break_even(np.array([10_000.0, 100_000.0]))
        self.assertAlmostEqual(float(b[1] / b[0]), 10.0, places=6)

    def test_clairvoyant_is_min_of_hold_and_evict(self):
        cost = CostModel()
        ctx = 122_000.0
        b = float(cost.break_even(ctx))
        # Just below break-even: holding is cheaper.
        self.assertAlmostEqual(
            float(cost.clairvoyant(b * 0.5, ctx)), float(cost.c(ctx) * b * 0.5), places=9
        )
        # Well above: evicting is cheaper, and the cost caps at B.
        self.assertAlmostEqual(float(cost.clairvoyant(b * 100, ctx)), float(cost.B(ctx)), places=9)

    def test_rejects_bad_configuration(self):
        for kwargs in ({"c_mode": "sideways"}, {"c0": 0.0}, {"kv_budget_tokens": 0}):
            with self.assertRaises(ValueError):
                CostModel(**kwargs)


class TestPolicies(unittest.TestCase):
    def setUp(self):
        self.cost = CostModel(c_mode="fixed")
        self.ctx = 122_000.0
        self.b = float(self.cost.break_even(self.ctx))

    def _decide(self, policy, predicted, true=1.0):
        return policy(
            Decision(
                predicted_duration_s=predicted,
                context_tokens=self.ctx,
                tool_key="t",
                true_duration_s=true,
            )
        )

    def test_lambda_one_is_exactly_the_break_even_rule(self):
        """At lambda=1 both branches collapse; the prediction stops mattering."""
        lam1 = make_lambda_policy(self.cost, 1.0)
        brk = make_break_even_ttl(self.cost)
        for predicted in (0.0, self.b * 0.5, self.b, self.b * 100):
            self.assertAlmostEqual(
                self._decide(lam1, predicted), self._decide(brk, predicted), places=9
            )

    def test_lambda_direction_smaller_means_more_trust(self):
        """Smaller lambda must widen the gap between the two branches."""
        spreads = []
        for lam in (0.9, 0.5, 0.1):
            p = make_lambda_policy(self.cost, lam)
            short = self._decide(p, predicted=0.0)
            long_ = self._decide(p, predicted=self.b * 10)
            self.assertGreater(short, long_)
            spreads.append(short / long_)
        self.assertLess(spreads[0], spreads[1])
        self.assertLess(spreads[1], spreads[2])

    def test_lambda_rejects_out_of_range(self):
        for lam in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                make_lambda_policy(self.cost, lam)

    def test_only_clairvoyant_reads_the_future(self):
        """Audit: no online policy may change its answer when only T changes."""
        online = {
            "immediate_evict": immediate_evict,
            "always_retain": always_retain,
            "fixed_ttl": make_fixed_ttl(30.0),
            "break_even": make_break_even_ttl(self.cost),
            "lambda_0.5": make_lambda_policy(self.cost, 0.5),
        }
        for name, policy in online.items():
            a = self._decide(policy, predicted=5.0, true=0.001)
            c = self._decide(policy, predicted=5.0, true=10_000.0)
            self.assertEqual(a, c, f"{name} changed its answer when only T changed")

        clair = make_clairvoyant(self.cost)
        self.assertNotEqual(
            self._decide(clair, predicted=5.0, true=0.001),
            self._decide(clair, predicted=5.0, true=10_000.0),
        )

    def test_worst_case_bound_binds_and_is_never_exceeded(self):
        for lam in (0.1, 0.25, 0.5, 0.75):
            probe = worst_case_probe(self.cost, lam)
            self.assertTrue(probe["within_bound"].all())
            # The probe is constructed to be tight, so it must reach the bound.
            self.assertAlmostEqual(float(probe["ratio"].max()), 1.0 + 1.0 / lam, places=3)


class TestSimulator(unittest.TestCase):
    def _frame(self, n_sessions=6, n_rounds=8, pause_s=2.0, ctx=50_000.0):
        rows = []
        t0 = pd.Timestamp("2026-01-01")
        for s in range(n_sessions):
            for r in range(n_rounds):
                rows.append(
                    {
                        "session_id": f"s{s}",
                        "round_index": r,
                        "tool_key": "t",
                        "context_tokens": ctx,
                        "pause_s": pause_s,
                        "predicted_duration_s": pause_s,
                        "pause_start_at": t0 + pd.Timedelta(seconds=r * 10 + s),
                        "pause_end_at": t0 + pd.Timedelta(seconds=r * 10 + s + pause_s),
                    }
                )
        return pd.DataFrame(rows)

    def test_immediate_evict_recomputes_every_round(self):
        frame = self._frame()
        cost = CostModel()
        r = run_one(frame, immediate_evict, cost, kv_budget_tokens=10_000_000)
        # Every round after the first pause pays a full re-prefill.
        self.assertAlmostEqual(r["recomputed_tokens"], 6 * 8 * 50_000.0, places=0)

    def test_always_retain_recomputes_nothing_when_memory_is_abundant(self):
        frame = self._frame()
        cost = CostModel()
        r = run_one(frame, always_retain, cost, kv_budget_tokens=10_000_000)
        self.assertEqual(r["recomputed_tokens"], 0.0)

    def test_scarce_memory_forces_eviction_even_under_always_retain(self):
        """A tau of infinity is an upper bound on residency, not a guarantee."""
        frame = self._frame(n_sessions=10, ctx=100_000.0)
        cost = CostModel()
        r = run_one(frame, always_retain, cost, kv_budget_tokens=200_000, arrival_scale=1000.0)
        self.assertGreater(r["evictions"], 0)
        self.assertGreater(r["recomputed_tokens"], 0.0)

    def test_tighter_budget_never_reduces_recomputation(self):
        frame = self._frame(n_sessions=10, ctx=100_000.0)
        cost = CostModel()
        prev = -1.0
        for budget in (5_000_000, 1_000_000, 400_000, 200_000):
            r = run_one(
                frame, always_retain, cost, kv_budget_tokens=budget, arrival_scale=1000.0
            )
            self.assertGreaterEqual(r["recomputed_tokens"], prev)
            prev = r["recomputed_tokens"]

    def test_metrics_are_finite(self):
        frame = self._frame()
        cost = CostModel()
        r = run_one(frame, make_break_even_ttl(cost), cost, kv_budget_tokens=300_000)
        for key, value in r.items():
            self.assertTrue(np.isfinite(value), f"{key} is not finite: {value}")


if __name__ == "__main__":
    unittest.main()
