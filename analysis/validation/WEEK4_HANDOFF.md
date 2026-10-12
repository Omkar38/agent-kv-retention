# Week 4 handoff — Brian's Version 2 + independent validation

**Current implementation:** [`build_pause_handoff.py`](build_pause_handoff.py) accepts the unmodified corrected **Brian Version 2** pause table. Brian's own definition and all his fields are preserved; the handoff only **adds** auditing, grouping and reproducible model-split information.

### Required Brian input

Get `week3/build_pause_table.py` and its resulting `pauses.parquet` from the current [Brian branch](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3), not its earlier commit. The input must contain `pause_start_at`, `pause_end_at`, and `log_format`. A Version 1 table causes the new builder to fail with a schema error instead of silently producing misleading results.

Brian's README reports **357,161** source rows and **351,497 usable** v2 records. These are upstream expected totals; this branch **has not yet reproduced the v2 handoff counts**.

### Changes relative to our historical v1 pilot

- The v1 single-tool filter, old endpoint reconstruction, and interleaved-next-round veto have been **replaced**. The handoff trusts Brian v2's pause endpoints and labels, checks them for internal consistency, and does not reinterpret parallel Claude tool calls as bad waiting intervals.
- Tool counts, names and input size are Brian's **v2 original columns**, kept unchanged. Audit `qa_tool_topology` distinguishes none, one and multiple. **Multi-tool pauses remain eligible** if the original target is usable and coherent.
- Human waits remain in the full table and in the usable model/evaluation table, with `cause`/`evaluation_group` for **offline stratification only**.
- All Brian v2 columns are also available in the usable table, but feature leakage is prevented by a documented training whitelist, not by deleting source columns.
- Older Codex records, marked `codex_old`, retain v2 original metadata, while `feature_n_tools`, `feature_tool_names` and `feature_tool_input_chars` are **masked** because result-time logging makes their online availability uncertain.
- The deterministic **SHA-256 provider/session split** is unchanged from the v1 pilot. No per-row random split.

### Output names

| Artifact | Purpose |
|---|---|
| `group_work/pause_handoff.parquet` | Lossless Brian v2 data plus QA, provenance and session splits |
| `group_work/week4_pause_model.parquet` | All original v2 rows passing integrity and Brian usability checks, **including multi-tool and human waits** |
| `group_work/pauses_v2.parquet` | **Legacy only**, produced from Brian v1; do not confuse with Brian's own v2 |
| `group_work/week4_pause_train.parquet` | **Legacy only**, restricted to single-tool v1 records |

### Run on Mac from TraceLab root

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
```

Additional reproduction and upload instructions: [`../dataset/README.md`](../dataset/README.md).

### Audits, metrics, and future use

The handoff reports total-row preservation, original-column preservation, v2 endpoint arithmetic checks, log-format and cause distribution, tool topology, and stable session-level splits. The bias script computes P50/P90/P99 and fractions over 10 and 60 seconds **separately by provider, log format and original cause**; no comparisons against contaminated v1 durations.

**Important:** Internal endpoint arithmetic is necessary but not independent validation of inference timestamps. `pause_s` is a TraceLab proxy, not observed GPU cache-idle residency; `cause` is only known retrospectively. Training and evaluation should stratify by cause and log format, while the online policy cannot condition on future cause information.
