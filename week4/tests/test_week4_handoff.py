"""Basic safety and split tests for Week 4."""
import unittest
import numpy as np
from week4_handoff import synthetic_fixture, prepare_handoff, validate_splits, inputs, tool_signature

class Week4Tests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_fixture(sessions=35, rows_per_session=3)

    def test_feature_whitelist(self):
        c,_ = prepare_handoff(self.df)
        x = inputs(c)
        for name in ("pause_s","target_s","analysis_track","split","session_key"):
            self.assertNotIn(name,x.columns)

    def test_sessions_disjoint(self):
        c,_ = prepare_handoff(self.df)
        self.assertTrue(all(validate_splits(c).values()))

    def test_detects_cross_split_session(self):
        self.df.loc[0,"split"]="test"
        c,_ = prepare_handoff(self.df)
        with self.assertRaises(AssertionError):
            validate_splits(c)

    def test_offline_cohort(self):
        n=len(self.df)
        self.df.loc[0,"analysis_track"]="diagnostic_event"
        c,a=prepare_handoff(self.df)
        self.assertEqual(a["source_rows"],n)
        self.assertEqual(len(c),n-1)

    def test_tool_signature(self):
        self.assertEqual(tool_signature(None),"<masked>")
        self.assertEqual(tool_signature(np.array(["Bash","Read","Bash"])),"MULTI:Bash+Read")

if __name__=="__main__":
    unittest.main()
