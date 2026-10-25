#!/usr/bin/env python3
"""Group 9 Week 4: prediction from Hangting_week3's 33-column Parquet handoff.

This does NOT measure actual GPU KV-cache idle residency. No repository writes.
All model inputs come from an explicit whitelist. All splits are inherited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

SOURCE = "https://github.com/Omkar38/agent-kv-retention/tree/Hangting_week3"
CATEGORICAL = ["provider", "model", "tool_signature"]
NUMERIC = ["feature_n_tools", "feature_log1p_context_tokens", "feature_log1p_round_index",
           "feature_log1p_tool_input_chars", "feature_has_tools", "feature_multiple_tools"]
FEATURES = CATEGORICAL + NUMERIC
REQUIRED = {
    "provider", "session_id", "model", "pause_s", "analysis_track", "split",
    "log_format", "tool_topology", "timing_category", "feature_category",
    "feature_tool_names", *NUMERIC,
}
POINT = ("global_median", "provider_tool_median", "log_ridge", "hgb_q50")


def tool_signature(value):
    """Normalize the handoff's list-valued feature_tool_names without reading raw tool_names."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "<masked>"
    if isinstance(value, (list, tuple, np.ndarray)):
        tools = sorted({str(x).strip() for x in value if x is not None and str(x).strip()})
    elif isinstance(value, str):
        val = value.strip()
        if not val:
            return "<masked>"
        # Serialized array may be a JSON array, otherwise retain a literal known category.
        if val.startswith("["):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return tool_signature(parsed)
            except ValueError:
                pass
        tools = [val]
    else:
        tools = [str(value)]
    if not tools:
        return "<masked>"
    if len(tools) == 1:
        return tools[0]
    return "MULTI:" + "+".join(tools[:3]) + ("+MORE" if len(tools) > 3 else "")


def load_handoff(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}; check out {SOURCE} first")
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError("Install a Parquet engine: python -m pip install pyarrow") from exc
    return prepare_handoff(frame, source=str(path.resolve()))


def prepare_handoff(frame, source="provided_dataframe"):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Not the 33-column Week 3 handoff; missing: {sorted(missing)}")
    original_rows = len(frame)
    audit = {
        "dataset": source,
        "source_rows": original_rows,
        "source_columns": len(frame.columns),
        "record_category_counts": {str(k): int(v) for k,v in frame.analysis_track.value_counts(dropna=False).items()},
        "timing_category_counts": {str(k): int(v) for k,v in frame.timing_category.value_counts(dropna=False).items()},
        "source_note": "All source rows preserved by the input file; only a clearly defined offline supervised cohort is selected.",
    }
    # This retrospective cohort selection is an EVALUATION definition, NEVER a model input.
    cohort = frame.loc[frame.analysis_track.eq("tool_pause_prediction")].copy()
    cohort["source_row_index"] = cohort.index.to_numpy(dtype="int64")
    y = pd.to_numeric(cohort.pause_s, errors="coerce")
    valid = y.notna() & np.isfinite(y) & (y > 0)
    audit["tool_prediction_track_rows"] = len(cohort)
    audit["nonpositive_or_nonfinite_target_in_track"] = int((~valid).sum())
    cohort = cohort.loc[valid].copy().reset_index(drop=True)
    cohort["target_s"] = pd.to_numeric(cohort.pause_s, errors="coerce")
    cohort["provider"] = cohort.provider.fillna("<missing>").astype(str)
    cohort["model"] = cohort.model.fillna("<missing>").astype(str)
    cohort["tool_signature"] = cohort.feature_tool_names.map(tool_signature)
    for column in NUMERIC:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce").astype("float64")
        cohort.loc[~np.isfinite(cohort[column]), column] = np.nan
    cohort["session_key"] = cohort.provider + "|" + cohort.session_id.fillna("<missing>").astype(str)
    cohort["split"] = cohort["split"].fillna("<missing>").astype(str)
    if not set(cohort["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError("Unexpected partition; do not re-split or train with unpartitioned sessions")
    audit["supervised_rows"] = len(cohort)
    audit["split_counts"] = {str(k):int(v) for k,v in cohort["split"].value_counts().items()}
    audit["supervised_by_provider"] = {str(k):int(v) for k,v in cohort.provider.value_counts().items()}
    audit["masked_tool_name_fraction"] = float(cohort.tool_signature.eq("<masked>").mean()) if len(cohort) else None
    return cohort, audit


def validate_splits(cohort):
    if cohort.empty:
        raise ValueError("No supervised tool_pause_prediction rows")
    sets = {split: set(cohort.loc[cohort.split == split, "session_key"]) for split in ("train", "val", "test")}
    if any(not s for s in sets.values()):
        raise ValueError("Every inherited split must have nonempty supervised sessions")
    if any(sets[a] & sets[b] for a,b in (("train", "val"),("train", "test"),("val", "test"))):
        raise AssertionError("Same session in multiple partitions: data leakage")
    return {k:len(v) for k,v in sets.items()}


def inputs(frame):
    # A whitelist: raw pause_s, retrospective cause, analysis_track, end time,
    # result category, original tool_names, and labels can NEVER enter estimator.
    X = frame[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].fillna("<masked>").astype(str)
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def model(kind, seed, iters):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=40,
                              min_frequency=20, sparse_output=False), CATEGORICAL),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
    ], sparse_threshold=0)
    if kind == "ridge":
        estimator = Ridge(alpha=10.0)
    else:
        estimator = HistGradientBoostingRegressor(
            loss="quantile", quantile=float(kind[1:]), max_iter=iters,
            max_leaf_nodes=18, learning_rate=0.075, min_samples_leaf=25,
            l2_regularization=4, random_state=seed)
    return Pipeline([("preprocess", prep), ("estimator", estimator)])


def from_log(estimate):
    return np.maximum(0., np.expm1(np.clip(np.asarray(estimate, dtype=float), 0, 30)))


def baseline(train, target):
    overall = float(train.target_s.median())
    provider_median = train.groupby("provider").target_s.median()
    grouped = train.groupby(["provider", "tool_signature"]).target_s.agg(["median", "size"])
    tool_median = train.groupby("tool_signature").target_s.agg(["median", "size"])
    vals = []
    for row in target.itertuples(index=False):
        combo = (row.provider, row.tool_signature)
        if combo in grouped.index and grouped.loc[combo, "size"] >= 20:
            v = grouped.loc[combo, "median"]
        elif row.tool_signature in tool_median.index and tool_median.loc[row.tool_signature, "size"] >= 20:
            v = tool_median.loc[row.tool_signature, "median"]
        else:
            v = provider_median.get(row.provider, overall)
        vals.append(float(v))
    return {"global_median":np.full(len(target), overall),
            "provider_tool_median":np.asarray(vals)}


def decision(y, p, horizon):
    # Strictly a diagnostic, not the proposed lambda policy or a hardware result.
    best_keep = (y <= horizon)
    proposed_keep = (p <= horizon)
    ttl = np.where(proposed_keep, horizon, 0.)
    cost = np.minimum(y, ttl) + horizon * (y > ttl)
    opt = np.minimum(y, horizon)
    near = (y >= 0.75*horizon) & (y <= 1.25*horizon)
    return {"horizon_s":float(horizon),
            "decision_flip_rate":float((proposed_keep != best_keep).mean()),
            "near_horizon_count":int(near.sum()),
            "near_horizon_flip_rate":float((proposed_keep[near] != best_keep[near]).mean()) if near.any() else None,
            "sum_cost_over_sum_clairvoyant":float(cost.sum()/opt.sum()) if opt.sum() else None}


def metrics(y, p, horizon):
    y, p = np.asarray(y,dtype=float), np.asarray(p,dtype=float)
    if len(y) == 0: return None
    err = p-y
    tail = y >= np.quantile(y, .9)
    return {"count":int(len(y)),
            "mae_s":float(np.abs(err).mean()),
            "rmse_s":float(np.sqrt(np.square(err).mean())),
            "log1p_mae":float(np.abs(np.log1p(p)-np.log1p(y)).mean()),
            "median_absolute_percentage_error":float(np.median(np.abs(err)/np.maximum(y,1.e-6))),
            "mean_signed_error_s":float(err.mean()),
            "underprediction_fraction":float((err < 0).mean()),
            "tail_top10_underprediction_fraction":float((err[tail] < 0).mean()),
            "tail_top10_mae_s":float(np.abs(err[tail]).mean()),
            "p90_absolute_error_s":float(np.quantile(np.abs(err), .9)),
            "threshold_diagnostic":decision(y,p,horizon)}


def train_and_evaluate(frame, outdir, *, source="input", seed=560, iters=75,
                       max_train=80000, horizon=1.0, synthetic=False):
    cohort, audit = prepare_handoff(frame,source)
    sessions = validate_splits(cohort)
    dest=Path(outdir); dest.mkdir(parents=True,exist_ok=True)
    train=cohort.loc[cohort.split.eq("train")].copy().reset_index(drop=True)
    val=cohort.loc[cohort.split.eq("val")].copy().reset_index(drop=True)
    test=cohort.loc[cohort.split.eq("test")].copy().reset_index(drop=True)
    fit=train.sample(n=min(max_train,len(train)),random_state=seed) if len(train)>max_train else train
    fitted={}; pred_val=baseline(train,val); pred_test=baseline(train,test)
    xv, xt, xf=inputs(val), inputs(test), inputs(fit)
    yfit=np.log1p(fit.target_s.to_numpy())
    for kind,name in (("ridge","log_ridge"),("q0.1","hgb_q10"),("q0.5","hgb_q50"),("q0.9","hgb_q90")):
        m=model(kind,seed,iters)
        m.fit(xf,yfit)
        fitted[name]=m
        pred_val[name]=from_log(m.predict(xv))
        pred_test[name]=from_log(m.predict(xt))
    yv=val.target_s.to_numpy(float); yt=test.target_s.to_numpy(float)
    valmet={n:metrics(yv,pred_val[n],horizon) for n in POINT}
    best=min(POINT,key=lambda n:valmet[n]["log1p_mae"])
    testmet={n:metrics(yt,pred_test[n],horizon) for n in POINT}
    # Crossing quantile curves are sorted but reported (not concealed).
    rawq=np.column_stack([pred_test["hgb_q10"],pred_test["hgb_q50"],pred_test["hgb_q90"]])
    n_cross=int((np.diff(rawq,axis=1)<0).any(axis=1).sum())
    qs=np.sort(rawq,axis=1)
    q10,q50,q90=qs.T
    qc={"p10_fraction_y_le_estimate":float((yt<=q10).mean()),
        "p50_fraction_y_le_estimate":float((yt<=q50).mean()),
        "p90_fraction_y_le_estimate":float((yt<=q90).mean()),
        "central_80pct_coverage":float(((yt>=q10)&(yt<=q90)).mean()),
        "crossing_rows_before_sort":n_cross,
        "median_interval_width_s":float(np.median(q90-q10))}
    # Preserve exact split and event-row order for policy/simulator comparisons.
    cols=["source_row_index","session_key","provider","session_id","round_index","model","log_format","tool_topology",
          "feature_category","tool_signature","target_s","split"]
    # Some original frames may have nullable round_index, but it is export-only.
    export=test[cols].copy()
    export.insert(0,"event_id",[f"source-{x}" for x in test.source_row_index])
    for name in POINT: export[f"pred_{name}_s"]=pred_test[name]
    export["pred_s"]=pred_test[best]
    export["q10_s"]=q10;export["q50_s"]=q50;export["q90_s"]=q90
    export.to_csv(dest/"predictions_test.csv",index=False)
    export_val=val[cols].copy()
    export_val.insert(0,"event_id",[f"source-{x}" for x in val.source_row_index])
    for name in POINT: export_val[f"pred_{name}_s"]=pred_val[name]
    export_val["pred_s"]=pred_val[best]
    export_val.to_csv(dest/"predictions_validation.csv",index=False)
    # Continuum-style per-tool empirical distributions must use training labels ONLY.
    train[["source_row_index","session_key","provider","session_id","tool_signature","target_s"]].to_csv(
        dest/"continuum_train_cdf_reference.csv.gz",index=False,compression="gzip")
    groups=[]
    for attr in ("provider","log_format","tool_topology","feature_category","tool_signature"):
        for key,index in test.groupby(attr,dropna=False).indices.items():
            if len(index)<20:continue
            m=metrics(yt[index],pred_test[best][index],horizon)
            groups.append({"attribute":attr,"value":str(key),"count":len(index),
                           "mae_s":m["mae_s"],"log1p_mae":m["log1p_mae"],
                           "underprediction_fraction":m["underprediction_fraction"],
                           "flip_rate":m["threshold_diagnostic"]["decision_flip_rate"]})
    pd.DataFrame(groups).sort_values(["attribute","count"],ascending=[True,False]).to_csv(
        dest/"group_error_analysis.csv",index=False)
    durations=yt
    result={"status":"synthetic_smoke_test" if synthetic else "real_trace_handoff_evaluation",
            "source":source,"input_sha256":sha256_file(Path(source)) if not synthetic and Path(source).is_file() else None,
            "audit":audit,"split_session_counts":sessions,
            "training_fit_rows":len(fit),"model_selection":"smallest validation log1p-MAE",
            "selected_model":best,"break_even_s":horizon,"break_even_is_calibrated":False,
            "test_duration_quantiles_s":{"p50":float(np.quantile(durations,.5)),
                                         "p90":float(np.quantile(durations,.9)),
                                         "p99":float(np.quantile(durations,.99))},
            "test_slowest_10pct_share_of_duration":float(np.sort(yt)[-max(1,int(np.ceil(.1*len(yt)))):].sum()/yt.sum()),
            "validation_metrics":valmet,"test_metrics":testmet,"quantile_calibration":qc,
            "diagnostic_horizon_sweep":{str(th):decision(yt,pred_test[best],th)
                                         for th in (0.1,1.,10.,60.)},
            "method_note":"Offline filtered tool-wait cohort; trace-derived pause, not GPU idle time. No test-target tuning. Cost ratios are simplified diagnostics, not end-to-end policy guarantees."}
    (dest/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    joblib.dump(fitted,dest/"fitted_models.joblib")
    write_report(result,dest)
    make_plots(yt,pred_test[best],q10,q50,q90,dest)
    return result


def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(2*1024*1024),b""):h.update(block)
    return h.hexdigest()


def write_report(r, dest):
    lines=["# Group 9 — Week 4 duration prediction", "",
           f"**STATUS:** \x60{r['status']}\x60",
           f"**Dataset:** \x60{r['source']}\x60",
           f"**Offline tool-pause cohort:** {r['audit']['supervised_rows']:,} / {r['audit']['source_rows']:,} original records",
           f"**Session groups:** {r['split_session_counts']}",
           f"**Validation-selected model:** \x60{r['selected_model']}\x60", "",
           "| Model | Test MAE (s) | Test log1p-MAE | Underprediction | Threshold-flip rate |",
           "|---|---:|---:|---:|---:|"]
    for name,m in r["test_metrics"].items():
        lines.append(f"| {name} | {m['mae_s']:.4f} | {m['log1p_mae']:.4f} | {m['underprediction_fraction']:.1%} | {m['threshold_diagnostic']['decision_flip_rate']:.1%} |")
    q=r["quantile_calibration"]
    lines += ["",f"Empirical P10/P50/P90 coverage: {q['p10_fraction_y_le_estimate']:.1%}, {q['p50_fraction_y_le_estimate']:.1%}, {q['p90_fraction_y_le_estimate']:.1%}.",
              f"Central 80% interval coverage: {q['central_80pct_coverage']:.1%}.",
              "",f"**Important:** B/c={r['break_even_s']} seconds is an illustrative, uncalibrated sensitivity threshold.",
              "Track filtering uses retrospective data only to define an offline supervised cohort; it is never an online model input.",
              "The data are trace-derived pause estimates, not GPU-resident cache inactivity measurements.",
              "The simplified threshold diagnostic is neither the new λ-policy nor the Continuum algorithm.",
              "Predictions and train-only empirical-CDF reference records are exported for later matched-trace policy comparisons."]
    (Path(dest)/"analysis.md").write_text("\n".join(lines)+"\n")


def make_plots(y,p,q10,q50,q90,dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i=np.linspace(0,len(y)-1,min(10000,len(y)),dtype=int)
    fig,ax=plt.subplots(figsize=(6.5,5.5))
    a,b=np.log1p(y[i]),np.log1p(p[i])
    ax.scatter(a,b,s=5,alpha=.17)
    bound=max(a.max(),b.max())
    ax.plot([0,bound],[0,bound],"--",linewidth=1)
    ax.set(xlabel="log(1 + observed pause, seconds)",ylabel="log(1 + predicted pause, seconds)",title="Held-out tool-pause prediction")
    fig.tight_layout();fig.savefig(Path(dest)/"prediction_scatter.png",dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,5.5))
    ax.plot([0,1],[0,1],"--",linewidth=1)
    ax.plot([.1,.5,.9],[(y<=x).mean() for x in (q10,q50,q90)],"o-")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Nominal quantile",ylabel="Fraction of observed pauses ≤ predicted quantile",title="Held-out quantile calibration")
    fig.tight_layout();fig.savefig(Path(dest)/"quantile_calibration.png",dpi=155);plt.close(fig)


def synthetic_fixture(seed=560, sessions=48, rows_per_session=8):
    rng=np.random.default_rng(seed)
    tools=["Read","Edit","Bash","exec_command","Grep"]
    rows=[]
    for s in range(sessions):
        provider="claude" if s%2 else "codex"
        split="train" if s%5<3 else "val" if s%5==3 else "test"
        for r in range(rows_per_session):
            tool=tools[int(rng.integers(0,len(tools)))]
            t=float(rng.lognormal(np.log({"Read":.15,"Edit":.3,"Bash":4.,"exec_command":2.,"Grep":.5}[tool]),1.1))
            rows.append(dict(provider=provider,session_id=f"session-{s}",model="synthetic-model",
                round_index=r,pause_s=t,analysis_track="tool_pause_prediction",split=split,
                log_format="claude" if provider=="claude" else "codex",tool_topology="single_tool_call",
                timing_category="timing_consistent",feature_category="source_round_tool_metadata",
                feature_tool_names=[tool],feature_n_tools=1,
                feature_log1p_context_tokens=np.log1p(8000+r*200),
                feature_log1p_round_index=np.log1p(r),
                feature_log1p_tool_input_chars=np.log1p(float(rng.integers(5,500))),
                feature_has_tools=True,feature_multiple_tools=False))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",default="analysis/dataset/pause_handoff.parquet")
    p.add_argument("--output-dir",default="results/week4_handoff")
    p.add_argument("--break-even-s",type=float,default=1.)
    p.add_argument("--max-train",type=int,default=80000)
    p.add_argument("--iterations",type=int,default=75)
    p.add_argument("--seed",type=int,default=560)
    p.add_argument("--demo",action="store_true",help="Only validate code with synthetic observations")
    args=p.parse_args()
    if args.break_even_s<=0: p.error("--break-even-s must be positive")
    if args.demo:
        df=synthetic_fixture(args.seed)
        res=train_and_evaluate(df,args.output_dir,synthetic=True,source="synthetic",seed=args.seed,
                               max_train=args.max_train,iters=args.iterations,horizon=args.break_even_s)
    else:
        # Load the actual input only once. The 33-column handoff is kept unchanged on disk.
        df=pd.read_parquet(args.input)
        res=train_and_evaluate(df,args.output_dir,source=args.input,seed=args.seed,
                               max_train=args.max_train,iters=args.iterations,horizon=args.break_even_s)
    print(json.dumps({"status":res["status"],"selected_model":res["selected_model"],
                      "rows":res["audit"]["supervised_rows"],"outputs":args.output_dir},indent=2))


if __name__=="__main__":main()
