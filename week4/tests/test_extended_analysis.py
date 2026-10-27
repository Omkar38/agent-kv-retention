import unittest
import numpy as np
from week4_handoff import synthetic_fixture, prepare_handoff, baseline
from week4.extended_analysis import (load_and_verify, audit_targets, make_threshold,
    make_tail, make_calibration, session_bootstrap, ablations)

class ExtendedTests(unittest.TestCase):
    def setUp(self):
        self.raw=synthetic_fixture(sessions=120,rows_per_session=5)
        self.cohort,_=prepare_handoff(self.raw)
        self.test=self.cohort.loc[self.cohort.split.eq("test")].copy().reset_index(drop=True)
        train=self.cohort.loc[self.cohort.split.eq("train")]
        self.test["pred_provider_tool_median_s"]=baseline(train,self.test)["provider_tool_median"]
        self.test["pred_hgb_q50_s"]=self.test.target_s*.91
        self.test["q10_s"]=self.test.target_s*.5
        self.test["q50_s"]=self.test.target_s*.91
        self.test["q90_s"]=self.test.target_s*2.

    def test_zero_records_audited_not_erased(self):
        self.raw.loc[0,"pause_s"]=0.
        a=audit_targets(self.raw)
        self.assertEqual(a.loc[a.label_state.eq("zero"),"n"].sum(),1)

    def test_threshold_error_decomposition(self):
        for r in make_threshold(self.test).itertuples():
            self.assertAlmostEqual(r.total_flip_fraction,r.false_retain_fraction+r.false_evict_fraction)
        self.assertIn("always_keep", set(make_threshold(self.test).model))
        self.assertIn("always_evict", set(make_threshold(self.test).model))

    def test_tail_subgroup_counts(self):
        a=make_tail(self.test)
        self.assertEqual(a.loc[(a.provider=="all")&(a.group=="all")&(a.model=="hgb_q50"),"n"].iloc[0],len(self.test))

    def test_calibration_all(self):
        a=make_calibration(self.test)
        self.assertEqual(a.loc[a.group_by.eq("all"),"n"].iloc[0],len(self.test))

    def test_paired_session_bootstrap(self):
        a=session_bootstrap(self.test,100,560)
        self.assertIn("log1p_mae_gain",set(a.metric))
        self.assertEqual(set(a.weighting),{"event","equal_session"})
        self.assertTrue(np.isfinite(a.ci95_low).all())

    def test_feature_ablation_excludes_tool_metadata(self):
        a=ablations(self.cohort,self.test,max_train=100,iterations=3)
        self.assertEqual(len(a),4)
        self.assertEqual(a.loc[a.variant.eq('provider_model_tool_name_only'),'tool_metadata_included'].iloc[0],True)
        self.assertEqual(int(a.tool_metadata_included.sum()),1)

    def test_event_alignment_rejects_corruption(self):
        import tempfile
        subset=self.test[["source_row_index","session_key","provider","split","target_s",
                          "pred_provider_tool_median_s","pred_hgb_q50_s","q10_s","q50_s","q90_s"]].copy()
        with tempfile.TemporaryDirectory() as temp:
            p=temp+"/test.csv"
            subset.to_csv(p,index=False)
            _,verified,_=load_and_verify(self.raw,p)
            self.assertEqual(len(verified),len(self.test))
            subset.loc[0,"target_s"]+=2.
            subset.to_csv(p,index=False)
            with self.assertRaises(ValueError):load_and_verify(self.raw,p)

if __name__=="__main__":unittest.main()
