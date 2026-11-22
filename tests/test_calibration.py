"""Pure CPU unit tests for the Week 9 cost-calibration pipeline.

Run:
    python3 -m unittest tests.test_calibration -v
"""
import unittest

import numpy as np
import pandas as pd

from experiments.calibrate_costs import (
    build_scenarios,
    estimate_rate,
    parse_positive_floats,
    validate_measurements,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.measured = pd.DataFrame({
            "context_tokens": [1000, 2000, 3000],
            "prefill_time_s": [0.1, 0.25, 0.3],
        })

    def test_robust_rate_estimate(self):
        observed = estimate_rate(self.measured)
        self.assertEqual(observed["sample_count"], 3)
        self.assertAlmostEqual(observed["tokens_per_s_median"], 10000.0)
        self.assertAlmostEqual(observed["context_tokens_min"], 1000.0)
        self.assertAlmostEqual(observed["context_tokens_max"], 3000.0)

    def test_missing_input_columns_rejected(self):
        with self.assertRaises(ValueError):
            validate_measurements(pd.DataFrame({"context_tokens": [1]}))

    def test_bad_input_measurements_rejected(self):
        for invalid in [0, -1, np.nan, np.inf]:
            with self.subTest(invalid=invalid):
                bad = self.measured.copy()
                bad.loc[0, "prefill_time_s"] = invalid
                with self.assertRaises(ValueError):
                    validate_measurements(bad)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            estimate_rate(pd.DataFrame(columns=["context_tokens", "prefill_time_s"]))

    def test_positive_parser(self):
        self.assertEqual(parse_positive_floats("1, 2,3", "x"), [1, 2, 3])
        for invalid in ["", "0,2", "-1,2", "nan,2"]:
            with self.assertRaises(ValueError):
                parse_positive_floats(invalid, "x")

    def test_fixed_cost_break_even_scales_with_context(self):
        tab = build_scenarios(10000, [1000, 2000], [300000], [0.4])
        fixed = tab[tab.c_mode == "fixed"].sort_values("context_tokens")
        np.testing.assert_allclose(fixed["B_s_modeled"], [0.1, 0.2])
        np.testing.assert_allclose(fixed["break_even_s"], [0.25, 0.5])

    def test_proportional_cost_break_even_cancels_context(self):
        tab = build_scenarios(10000, [1000, 2000], [300000], [0.4])
        proportional = tab[tab.c_mode == "proportional"]
        np.testing.assert_allclose(proportional["break_even_s"], [75.0, 75.0])

    def test_queueing_cost_changes_both_scenarios(self):
        base = build_scenarios(10000, [2000], [300000], [0.4])
        extra = build_scenarios(10000, [2000], [300000], [0.4],
                                queueing_penalty_s=2)
        self.assertTrue((extra["break_even_s"] > base["break_even_s"]).all())

    def test_bad_queue_penalty_rejected(self):
        with self.assertRaises(ValueError):
            build_scenarios(10000, [1000], [300000], [0.4], -2)


if __name__ == "__main__":
    unittest.main()
