#!/usr/bin/env python3
"""Week 4 independent evaluation: audits, tail errors, threshold sensitivity,
group calibration, paired session bootstrap and restricted-feature ablation.

Uses the immutable Week 3 Parquet and the Week 4 HELD-OUT prediction export.
All output categories and cache-cost thresholds are offline diagnostics.
"""
import argparse
import json
import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from week4_handoff import prepare_handoff, validate_splits, from_log

REF="pred_provider_tool_median_s"
HGB="pred_hgb_q50_s"
MODELS={"provider_tool_median":REF,"hgb_q50":HGB}
THRESHOLDS=[0.1,0.3,1.,3.,10.,60.]

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def load_and_verify(raw,pred_file):
    cohort,audit=prepare_handoff(raw)
    validate_splits(cohort)
    pred=pd.read_csv(pred_file)
    required={"source_row_index","session_key","provider","split","target_s",REF,HGB,"q10_s","q50_s","q90_s"}
    missing=required-set(pred.columns)
    if missing: raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if len(pred)!=len(cohort[cohort.split.eq("test")]): raise ValueError("Test-row count mismatch")
    if pred.source_row_index.duplicated().any(): raise ValueError("Duplicate event IDs")
    joined=cohort.loc[cohort.split.eq("test")].merge(
        pred,on="source_row_index",suffixes=("","_saved"),
        indicator=True,validate="one_to_one")
    if len(joined)!=len(pred) or not joined._merge.eq("both").all():
        raise ValueError("Prediction rows differ from test cohort")
    if (joined.split_saved!="test").any():raise ValueError("Non-test prediction row")
    for x in ("session_key","provider","split"):
        if not joined[x].astype(str).equals(joined[x+"_saved"].astype(str)):
            raise ValueError(f"Event alignment mismatch: {x}")
    if not np.allclose(joined.target_s,joined.target_s_saved,atol=1e-6,rtol=1e-9):
        raise ValueError("Observed duration mismatch")
    for x in [REF,HGB,"q10_s","q50_s","q90_s"]:
        if not np.isfinite(joined[x]).all() or (joined[x]<0).any():
            raise ValueError(f"Invalid numerical value in {x}")
    return cohort,joined.reset_index(drop=True),audit

def audit_targets(raw):
    f=raw.copy()
    v=pd.to_numeric(f.pause_s,errors="coerce")
    f["label_state"]=np.select(
        [v.isna(),np.isinf(v),v<0,v==0,v>0],
        ["missing","infinite","negative","zero","positive"],default="other")
    dims=["analysis_track","provider","log_format","timing_category","label_state"]
    return f.groupby(dims,dropna=False).size().rename("n").reset_index()

def audit_features(cohort):
    dims=["split","provider","log_format","feature_category","tool_topology"]
    out=cohort.groupby(dims,dropna=False).size().rename("n").reset_index()
    out["interpretation"]=np.where(out.feature_category.eq("result_logged_tool_metadata"),
        "tool metadata masked: may appear at result time",
        "candidate metadata: exact decision-time availability not verified")
    return out

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if not len(y):return dict(n=0,mae_s=np.nan,log1p_mae=np.nan,underprediction=np.nan,overprediction=np.nan)
    return dict(n=len(y),mae_s=float(np.mean(np.abs(y-p))),
        log1p_mae=float(np.mean(np.abs(np.log1p(y)-np.log1p(p)))),
        underprediction=float(np.mean(p<y)),overprediction=float(np.mean(p>y)))

def make_tail(test):
    cutoff=float(test.target_s.quantile(.9))
    out=[]
    # Use the globally defined top-decile cutoff for all subgroups.
    for provider,f in [("all",test)]+list(test.groupby("provider")):
        groups={
            "all":f,
            "pause_gt_10s":f[f.target_s>10],
            "pause_gt_60s":f[f.target_s>60],
            "test_top_decile":f[f.target_s>=cutoff],
            "pause_le_10s":f[f.target_s<=10],
            "pause_10_to_60s":f[(f.target_s>10)&(f.target_s<=60)]
        }
        for group,g in groups.items():
            for model,col in MODELS.items():
                out.append(dict(provider=provider,group=group,model=model,
                                global_p90_cutoff_s=cutoff,**metrics(g.target_s,g[col])))
    return pd.DataFrame(out)

def make_group(test):
    rows=[]
    for dim in ("provider","log_format","tool_topology","tool_signature","feature_category"):
        for v,g in test.groupby(dim,dropna=False):
            if len(g)<30:continue
            for model,col in MODELS.items():
                rows.append(dict(group_by=dim,value=str(v),model=model,
                     fraction_gt_10s=float((g.target_s>10).mean()),
                     fraction_gt_60s=float((g.target_s>60).mean()),
                     **metrics(g.target_s,g[col])))
    return pd.DataFrame(rows)

def make_threshold(test):
    rows=[]
    y=test.target_s.to_numpy()
    for B in THRESHOLDS:
        actual_keep=y<=B
        near=(y>=.75*B)&(y<=1.25*B)
        for name,col in MODELS.items():
            p=test[col].to_numpy()
            keep=p<=B
            fk=keep&(~actual_keep)
            fe=(~keep)&actual_keep
            flip=fk|fe
            tau=np.where(keep,B,0)
            C=np.minimum(y,tau)+B*(y>tau)
            opt=np.minimum(y,B)
            rows.append(dict(threshold_B_over_c_s=B,model=name,n=len(y),
                actual_keep_fraction=float(actual_keep.mean()),
                predicted_keep_fraction=float(keep.mean()),
                false_retain_fraction=float(fk.mean()),
                false_evict_fraction=float(fe.mean()),
                total_flip_fraction=float(flip.mean()),
                near_threshold_n=int(near.sum()),
                near_threshold_flip_fraction=float(flip[near].mean()) if near.any() else np.nan,
                toy_cost_over_oracle=float(C.sum()/opt.sum()) if opt.sum()>0 else np.nan))
    return pd.DataFrame(rows)

def make_calibration(test):
    rows=[]
    for dim in ("all","provider","log_format","tool_signature"):
        groups=[("all",test)] if dim=="all" else test.groupby(dim,dropna=False)
        for name,g in groups:
            if len(g)<50:continue
            y=g.target_s.to_numpy()
            lo=g.q10_s.to_numpy();mid=g.q50_s.to_numpy();hi=g.q90_s.to_numpy()
            inside=(y>=lo)&(y<=hi)
            rows.append(dict(group_by=dim,value=str(name),n=len(g),
                coverage_p10=float((y<=lo).mean()),
                coverage_p50=float((y<=mid).mean()),
                coverage_p90=float((y<=hi).mean()),
                central_80_coverage=float(inside.mean()),
                median_interval_width_s=float(np.median(hi-lo)),
                conditional_coverage_gt_10s=float(inside[y>10].mean()) if (y>10).any() else np.nan,
                conditional_coverage_gt_60s=float(inside[y>60].mean()) if (y>60).any() else np.nan))
    return pd.DataFrame(rows)

def session_bootstrap(test,replicates=2000,seed=560):
    """Paired cluster bootstrap, sampling sessions not individual pause records."""
    rng=np.random.default_rng(seed)
    out=[]
    for group,frame in [("all",test)]+list(test.groupby("provider")):
        if frame.session_key.nunique()<15:continue
        f=frame.copy()
        y=f.target_s
        f["mae_gain"]=abs(f[REF]-y)-abs(f[HGB]-y)
        f["log_gain"]=abs(np.log1p(f[REF])-np.log1p(y))-abs(np.log1p(f[HGB])-np.log1p(y))
        for B in [1.,10.]:
            keep=y<=B
            f[f"flip_gain_{int(B)}"]=((f[REF]<=B)!=keep).astype(int)-((f[HGB]<=B)!=keep).astype(int)
        ag=f.groupby("session_key").agg(n=("target_s","size"),mae=("mae_gain","sum"),
              log=("log_gain","sum"),f1=("flip_gain_1","sum"),f10=("flip_gain_10","sum"))
        arr=ag[["n","mae","log","f1","f10"]].to_numpy(dtype=float)
        idx=rng.integers(0,len(arr),size=(replicates,len(arr)))
        sums=arr[idx].sum(axis=1)
        for i,metric in [(1,"mae_s_gain"),(2,"log1p_mae_gain"),
                         (3,"flip_1s_gain"),(4,"flip_10s_gain")]:
            sample=sums[:,i]/sums[:,0]
            lo,hi=np.quantile(sample,[.025,.975])
            out.append(dict(provider=group,metric=metric,
                observed_gain=float(arr[:,i].sum()/arr[:,0].sum()),
                ci95_low=float(lo),ci95_high=float(hi),
                fraction_bootstrap_gain_positive=float((sample>0).mean()),
                sessions=len(ag),test_events=len(f),bootstrap_replicates=replicates))
    return pd.DataFrame(out)

def ablations(cohort,test,max_train=80000,iterations=75,seed=560):
    """Drop potentially future-derived tool metadata; context remains provisional."""
    train=cohort[cohort.split.eq("train")]
    val=cohort[cohort.split.eq("val")]
    fit=train.sample(n=min(max_train,len(train)),random_state=seed) if len(train)>max_train else train
    variants=[
      ("provider_model",[]),
      ("provider_model_turn",["feature_log1p_round_index"]),
      ("provider_model_turn_context_provisional",
         ["feature_log1p_round_index","feature_log1p_context_tokens"])]
    out=[]
    for label,num in variants:
        cats=["provider","model"]
        transformers=[("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cats)]
        if num:
            transformers.append(("num",SimpleImputer(strategy="median",add_indicator=True),num))
        prep=ColumnTransformer(transformers,sparse_threshold=0)
        reg=HistGradientBoostingRegressor(loss="quantile",quantile=.5,
             max_iter=iterations,max_leaf_nodes=18,learning_rate=.075,
             min_samples_leaf=25,l2_regularization=4,random_state=seed)
        mdl=Pipeline([("features",prep),("estimator",reg)])
        cols=cats+num
        mdl.fit(fit[cols],np.log1p(fit.target_s))
        mv=metrics(val.target_s,from_log(mdl.predict(val[cols])))
        mt=metrics(test.target_s,from_log(mdl.predict(test[cols])))
        out.append(dict(variant=label,features=",".join(cols),
            validation_log1p_mae=mv["log1p_mae"],test_log1p_mae=mt["log1p_mae"],
            validation_mae_s=mv["mae_s"],test_mae_s=mt["mae_s"],
            fit_rows=len(fit),test_rows=len(test),
            tool_metadata_included=False,context_timing_proven=False))
    return pd.DataFrame(out)

def report(path,audit,elig,tail,threshold,calibration,boot,ablate,manifest):
    pick=lambda df,**kw: df.loc[np.logical_and.reduce([df[k].eq(v) for k,v in kw.items()])].iloc[0]
    a=pick(tail,provider="all",group="all",model="provider_tool_median")
    b=pick(tail,provider="all",group="all",model="hgb_q50")
    q=pick(calibration,group_by="all")
    bt=pick(boot,provider="all",metric="log1p_mae_gain")
    zero=int(elig.loc[(elig.analysis_track=="tool_pause_prediction")&
               (elig.label_state=="zero"),"n"].sum())
    negative=int(elig.loc[(elig.analysis_track=="tool_pause_prediction")&
               (elig.label_state=="negative"),"n"].sum())
    missing=int(elig.loc[(elig.analysis_track=="tool_pause_prediction")&
               elig.label_state.isin(["missing","infinite","other"]),"n"].sum())
    lines=[
     "# Week 4 Final Research Report — Group 9","",
     "## Scope / provenance","",
     f"- Canonical dataset: 357,161 records, 33 columns; supervised observations: {audit['supervised_rows']:,}; held-out test records: {manifest['test_events']:,}.",
     "- Model inputs exclude target, original cause, timing outcomes, session ID and analysis-track labels. Session IDs are for splitting and uncertainty estimates only.",
     "- Trace-derived tool pauses are proxies, not measured GPU KV-cache residency, and do not themselves establish latency improvements.","",
     "## Real-trace duration prediction","",
     f"- Test log1p-MAE: provider/tool median {a.log1p_mae:.4f}, HGB Q50 {b.log1p_mae:.4f}.",
     f"- Test MAE: provider/tool median {a.mae_s:.3f}s, HGB Q50 {b.mae_s:.3f}s.",
     f"- Session-paired bootstrap gain in log1p-MAE (positive favors HGB): {bt.observed_gain:.4f}; 95% interval [{bt.ci95_low:.4f}, {bt.ci95_high:.4f}].",
     "- Significant relative improvement in log error does not prove a systems-level advantage under a GPU memory constraint.","",
     "## Eligibility and future-information restrictions","",
     f"- The supervised-track records with excluded target outcomes: zero={zero:,}, negative={negative:,}, missing/nonfinite={missing:,}. Zero-duration pauses are not automatically invalid; inspect their logging provenance.",
     "- Tool lists and counts may include interactions not known at the first pause; the original features are candidate signals, not formally certified online signals.",
     "- Three minimal-feature ablations remove ALL tool metadata. The context-token feature remains provisionally timed, so even the context ablation is a sensitivity analysis.",
     "- The retrospective offline cohort selector cannot be used in the online policy. Older Codex and multi-tool timing ambiguity must remain disclosed.","",
     "## Tail, calibration, and decision sensitivity","",
     "- Tool durations are strongly heavy-tailed; see tail_error_analysis.csv for >10s, >60s, top decile, provider-specific error and underprediction.",
     f"- Held-out P10/P50/P90 empirical coverage: {q.coverage_p10:.1%} / {q.coverage_p50:.1%} / {q.coverage_p90:.1%}. Central 80% interval coverage: {q.central_80_coverage:.1%}.",
     "- Marginally calibrated quantiles need not be calibrated in the long tail or by tool; see calibration_by_group.csv.",
     "- threshold_analysis.csv reports false-retain, false-evict, near-threshold flips and a two-action cost diagnostic at multiple illustrative B/c settings.",
     "- B/c settings are NOT calibrated, and these diagnostics are NOT the learning-augmented λ-policy, a Continuum-style TTL, or a consistency/robustness guarantee.","",
     "## Reproduction and Week 6 handoff","",
     f"- Dataset SHA256: {manifest['dataset_sha256']}",
     f"- Test prediction SHA256: {manifest['prediction_sha256']}",
     f"- Seed {manifest['seed']}; resamples {manifest['bootstrap_replicates']}; training cap {manifest['max_train']}; boosting iterations {manifest['iterations']}.",
     "- Keep local test predictions and the train-only duration/CDF reference for the later matched-trace, matched-capacity Continuum experiment.",
     "- Week 6 establishes a simplified single-request consistency/robustness theorem; load-sensitive multi-request behavior is a simulation extension.",
     "- Remaining work is future policy development and workload calibration, not an assertion that the current predictor is already an optimal KV retention policy.",""
    ]
    path.write_text("\n".join(lines))

def git_head():
    try:
        r=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,
                         text=True,timeout=5,check=True)
        return r.stdout.strip()
    except Exception:return None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default="analysis/dataset/pause_handoff.parquet")
    parser.add_argument("--predictions",default="results/week4_handoff/predictions_test.csv")
    parser.add_argument("--out",default="results/week4_extended")
    parser.add_argument("--seed",type=int,default=560)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--max-train",type=int,default=80000)
    parser.add_argument("--iterations",type=int,default=75)
    args=parser.parse_args()
    if args.bootstrap<100:parser.error("--bootstrap must be >= 100")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(args.dataset)
    cohort,test,audit=load_and_verify(raw,args.predictions)
    results={
       "target_eligibility":audit_targets(raw),
       "feature_audit":audit_features(cohort),
       "tail_error_analysis":make_tail(test),
       "group_error_extended":make_group(test),
       "threshold_analysis":make_threshold(test),
       "calibration_by_group":make_calibration(test),
       "session_bootstrap":session_bootstrap(test,args.bootstrap,args.seed),
       "feature_ablation":ablations(cohort,test,args.max_train,args.iterations,args.seed)}
    for name,df in results.items():
        df.to_csv(out/(name+".csv"),index=False)
    import sklearn
    meta=dict(dataset_sha256=file_hash(args.dataset),
        prediction_sha256=file_hash(args.predictions),
        dataset_rows=len(raw),dataset_columns=len(raw.columns),
        supervised_rows=audit["supervised_rows"],test_events=len(test),
        seed=args.seed,bootstrap_replicates=args.bootstrap,
        max_train=args.max_train,iterations=args.iterations,
        python=platform.python_version(),pandas=pd.__version__,
        numpy=np.__version__,sklearn=sklearn.__version__,git_commit=git_head(),
        caveat="trace-derived idle proxy, uncalibrated B/c, provisional decision-time features")
    (out/"experiment_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    report(out/"WEEK4_FINAL_REPORT.md",audit,results["target_eligibility"],
        results["tail_error_analysis"],results["threshold_analysis"],
        results["calibration_by_group"],results["session_bootstrap"],
        results["feature_ablation"],meta)
    print(json.dumps({"status":"completed","test_events":len(test),
                      "outputs":str(out)},indent=2))

if __name__=="__main__":main()
