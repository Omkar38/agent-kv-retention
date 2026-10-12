# Week 4 datasets — Brian v2, lossless audited handoff

**Source of truth:** Brian's [Week 3 Version 2 pause extractor](https://github.com/Omkar38/agent-kv-retention/blob/Brian/week3/build_pause_table.py). All v2 records and columns are preserved, including `pause_s`, `cause`, `usable`, `context_tokens`, `n_tools`, `tool_names`, `tool_input_chars`, `pause_start_at`, `pause_end_at`, and `log_format`.

**Integration:** Hangting's [build_pause_handoff.py](../validation/build_pause_handoff.py) adds `qa_endpoint_duration_s`, `qa_boundary_status`, `qa_feature_origin`, `qa_tool_topology`, `evaluation_group`, `eligible_for_prediction`, masked `feature_*` columns, and deterministic `split`, without dropping Brian's columns.

## Current and historical artifacts

| Filename | Meaning |
|---|---|
| `pause_handoff.parquet` **(to regenerate)** | All **357,161** Brian v2 records, including unusable and unknown causes; v2 columns unchanged, additional diagnostics |
| `week4_pause_model.parquet` **(to regenerate)** | Every v2 record passing basic integrity and usability requirements; retains single- and multi-tool, and human waits as distinguishable groups |
| `pauses_v2.parquet` **(legacy)** | Historical handoff generated from Brian **v1**; **not** Brian's corrected v2 |
| `week4_pause_train.parquet` **(legacy)** | Historical 219,465-record, single-tool v1 benchmark; **not** the current Week 4 training input |

**Do not train on legacy Parquet artifacts or misidentify them as v2.** The new two files have not yet been run and uploaded from the user's TraceLab environment.

## Prediction-safe feature policy

Every Brian v2 column is preserved in both outputs for provenance, analysis, and auditing, **but this does not mean every column can be used as a predictor feature**. For initial training, use only `provider`, `feature_n_tools`, `feature_tool_names`, and `feature_tool_input_chars` (and missingness indicators if needed). `session_id` is a group identifier; `split` is a train/val/test assignment; `pause_s` is the target. `cause` and `evaluation_group` are retrospective annotations, **not** prediction features. Exclude `pause_end_at`, `qa_*`, and other future-outcome fields. Treat `context_tokens`, `model`, and `round_index` as additional features only after verifying their decision-time availability.

Older Codex logs (`log_format=codex_old`) may record tool metadata at result time; the handoff **retains original Brian tool columns** but masks corresponding `feature_*` values. A v2-valid duration is not proof that every associated feature was available at decision time. Missing tool features may be imputed or old Codex evaluated separately.

Both outputs use the same SHA-256, session-grouped 80/10/10 assignment preserved from the previous handoff. Split membership is deterministic and rows of the same provider/session do not cross splits.

## Generate

From the TraceLab repository root on Mac:

```bash
cd ~/kv-week3-work/agent-kv-retention-hangting
git switch Hangting_week3
git pull --ff-only origin Hangting_week3
git fetch origin Brian

cd ~/kv-week3-work/TraceLab
REPO=~/kv-week3-work/agent-kv-retention-hangting
mkdir -p our_work group_work

git -C "$REPO" show origin/Brian:week3/build_pause_table.py > our_work/build_pause_table.py
python our_work/build_pause_table.py
cp our_work/pauses.parquet group_work/pauses.parquet

python "$REPO/analysis/validation/build_pause_handoff.py"
python "$REPO/analysis/validation/check_handoff_bias.py"

cp group_work/pause_handoff.parquet "$REPO/analysis/dataset/"
cp group_work/week4_pause_model.parquet "$REPO/analysis/dataset/"
```

Only after inspecting the builder's checks and the new file schemas, commit and push **both** files. Remove the old v1-named Parquet files only after verifying the replacement files, to avoid confusing classmates. Data is based on TraceLab release v0.0.1; cite its published license and paper.

**Scientific limitation:** Brian v2 produces agent *pause-duration proxies*, not observed physical GPU KV-cache residency. The Week 4 dataset supports prediction and sensitivity analysis, not measured GPU savings.
