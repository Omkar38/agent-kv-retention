> **Historical Version 1 validation record:** This document reports initial analyses, not the final Version 2 handoff. The sole current dataset is [`../dataset/pause_handoff.parquet`](../dataset/pause_handoff.parquet); new record and timing categories supersede binary screening for Week 4.

# Week 3 TraceLab Validation — Findings and Limitations

**Project:** Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents
**Audit date:** October 11, 2026
**Branch:** `Hangting_week3`
**Data source:** local TraceLab release `trace/syfi_coding_trace.duckdb`
**Scope:** workload characterization, extraction QA, and preliminary tool-duration proxy preparation.

## Executive summary

The team's original round-gap analysis was reproduced. Its tool attribution is not reliably decision-time-safe for Codex: a documented example attaches a tool issued in the next round to a roughly 24-day preceding user-inactivity gap. An alternative audit extracts observed tool execution intervals directly from `tool_calls`, merges overlapping intervals within a session, and flags simpler candidate episodes. These candidates are useful for **preliminary tool-duration prediction**, but they are not yet established as the LLM's exact KV-cache retention opportunities.

## 1. Original round-gap table: reproduced counts

| Cause | Total | Usable | Unusable |
|---|---:|---:|---:|
| `tool` | 310,628 | 309,673 | 955 |
| `user` | 31,992 | 31,924 | 68 |
| `tool+user` | 5,230 | 5,223 | 7 |
| `unknown` | 5,046 | 0 | 5,046 |
| `end` | 4,265 | 0 | 4,265 |
| **All** | **357,161** | **346,820** | **10,341** |

The audit found **531 negative-duration** round-gap records. A separate check found **514 duplicate `(session_id, round_index)` groups** in **16 sessions**, covering 1,028 round records. These checks do not, by themselves, fully reconcile all exclusion reasons or establish a correction to the round-ordering logic. Negative gaps should not be converted to positive values or silently repaired.

The original original duration distributions and figures were reproduced, but they are **preliminary** for the final KV-cache idle-time evaluation because the source attribution and endpoint definitions require revision.

## 2. Decision-time leakage and Codex round structure

In a result-driven join from next-round `tool_result` events to `tool_calls`, the audit counted **430,452 matched call records**, of which **63,122** had `emitted_at > estimated_decision_ts` (14.66%). This is a **match-record count**, not a count of distinct affected pauses.

| Provider | Source round of matched tool | Matched call records | Emitted after decision |
|---|---|---:|---:|
| Claude | Current | 140,719 | 0 |
| Claude | Next | 7 | 7 |
| Claude | Other | 112 | 1 |
| Codex | Current | 225,087 | 0 |
| Codex | Next | 64,426 | 63,114 |
| Codex | Other | 101 | 0 |

**Example:** Session `codex:019dd611-dd7c-7750-915b-910d54eaccbc`, round 383 has last output at `2026-05-01 22:57:09.163`. Round 384 begins with a user message at `2026-05-25 20:45:16.217`; only after new reasoning/text does `exec_command` emit at `20:45:21.802` and return at `20:45:22.245` (443 ms). The latter tool call cannot explain the preceding roughly 24-day inactivity interval, despite being associated with it by next-round result matching.

A within-round audit found that `tool_result` occurred after the first reasoning/text/tool-call event in **8 of 141,888 Claude results** and **58,031 of 289,930 Codex results** (with 28 Codex cases of unknown comparison timing). These findings demonstrate that Codex tool-result placement frequently differs from a simple "inputs then generation" per-round assumption.

**Interpretation:** Next-round result matching may leak information that was unavailable at the proposed decision time and may misclassify user inactivity as tool waiting. The exact number of distinct affected pauses remains unquantified. Raw TraceLab tool records should be preserved.

## 3. Tool intervals: independently derived measurements

Tool-call records were extracted using `tool_calls.emitted_at` and `tool_calls.result_at`.

| Provider | Valid intervals | Missing timestamps | Median duration | P99 duration |
|---|---:|---:|---:|---:|
| Claude | 141,824 | 564 | 0.125 s | 309.314 s |
| Codex | 290,081 | 41 | 0.237 s | 120.002 s |
| **All** | **431,905** | **605** | — | — |

Total tool-call records: **432,510**. Under the implemented audit, all timestamp-complete intervals were nonnegative. For all 431,905 comparable intervals, `result_at - emitted_at` agreed exactly (at reported precision) with `tool_wall_latency_ms`; the mean and P99 absolute differences were zero. This demonstrates internal consistency, **not independent validation** of how the trace constructs those fields.

A per-session overlap audit identified **11,542 Claude** and **88,963 Codex** calls beginning before an earlier call's maximum end time: **100,505 potentially overlapping calls** in total (23.27% of valid tool calls). This does not prove independent concurrent GPU work; overlap may reflect multiple tools, command polling, or trace representation.

## 4. Non-double-counted tool activity episodes

Using the union of overlapping valid tool intervals within a provider/session:

| Provider | Episodes | Multi-call episodes | Median episode duration | P99 episode duration |
|---|---:|---:|---:|---:|
| Claude | 130,282 | 7,030 | 0.112 s | 316.423 s |
| Codex | 201,118 | 46,444 | 0.328 s | 126.733 s |
| **All** | **331,400** | **53,474** | — | — |

The reduction from 431,905 tool calls to 331,400 episodes equals **100,505**, consistent with the overlap diagnostic and the unioning convention used here. A merged episode is a span of recorded tool activity; it is **not automatically** a period of LLM/GPU idleness.

A separate cross-event check found **109 episodes containing a recorded reasoning/text event**, **124 containing a user-message event**, and **331,175 containing neither**. The two signal classes overlap; they should not simply be added to calculate an exclusion count. The absence of sparse events is not proof of continuous inference inactivity.

## 5. Candidate episodes and audit flags

Every episode was associated with a first emitted tool and a positive context-token proxy; the audit found **zero missing first-tool names**, **zero missing first-tool input character counts**, and **zero nonpositive context-token counts**. It did not prove that the context-token proxy was observable exactly at the proposed retention decision.

The boundary audit found reasoning/text after first tool emission in **201 Claude** and **23 Codex** candidates, and later tool-call events in the same round for **16,434 Claude** and **44,440 Codex** candidates. A later event somewhere in the round does not necessarily occur inside the episode.

The final audit classifies an episode as `simple_tool_wait_proxy` when it contains one recorded tool call and no recorded reasoning/text/user message strictly inside its observed interval:

| Provider | All candidates | Simple proxy | Complex/flagged |
|---|---:|---:|---:|
| Claude | 130,282 | 123,112 | 7,170 |
| Codex | 201,118 | 154,665 | 46,453 |
| **All** | **331,400** | **277,777 (83.82%)** | **53,623 (16.18%)** |

The classifications are descriptive flags, **not a claim that all 277,777 records represent correctly delimited KV-cache pauses**. All complex cases remain in the dataset for later analysis.

## 6. Leakage-safe modeling boundary

At the candidate episode's tool-emission time, the first emitted call's name and input characters can serve as **provisional** prediction features. The final episode duration, eventual number of calls, full set of tool names, and later results are **labels or retrospective information**, not decision-time features. Even the first-call and round-context fields require checking against a defensible inference-stop/retention-decision time before deployment-style evaluation.

Do **not** train on `observed_n_calls`, `observed_n_rounds`, `observed_tool_names`, or the target duration as contemporaneous features. The current `tool_activity_duration_s` is a proxy for tool activity, **not a verified wall-clock gap until the next LLM decoding step**.

## 7. Corrections to related work

- `analysis/trace_facts/check_tool_latency.py`: fixed a mismatch between SQL output ordering and printed percentile columns; the corrected local run reported expected mean/P50/P90/P95/P99/P99.9 columns.
- `analysis/trace_facts/check_prediction_fields.py`: replaced references to unavailable tool-call columns with fields present in the released schema. Review its output and retain only features available at the policy's decision point.
- The original original pause extraction and figures were independently reproduced, not replaced or overwritten. Preserve attribution and the original team branches.

## 8. Open issues and handoff

**Outstanding for correct KV retention modeling:**

1. Locate the actual **end of LLM generation / start of cache retention** and the next LLM continuation, including cases where tool calls/results occur within one round.
2. Reclassify human-driven gaps separately from tool-driven waits; avoid next-round future-result attribution.
3. Determine how to treat multi-call episodes, polling, session breaks, missing timestamps, duplicate rounds, and negative gaps.
4. Rebuild and test a distinct `pauses_v2.parquet` with absolute timestamps and explicit exclusion flags; validate prediction features *only* from information known at the chosen decision instant.
5. Recompute CDF, tool-type, and context-size figures from the corrected target. Use the same traces and memory constraints for comparisons against fixed TTL, eviction, and Continuum-style policies.
6. Treat prefill speeds/eviction recomputation parameters used in initial figures as *illustrative sensitivity assumptions* until calibrated, not measured hardware results.

**Week 4 handoff:** `tool_episode_candidates_audited.parquet` can be used for a clearly labeled preliminary **tool-activity-duration** predictor. Keep simple and complex subsets separate in evaluation, split at the session level to avoid leakage, and do not claim that performance directly establishes optimal cache-retention TTLs.

**Project-scope alignment:** Keep the theoretical consistency/robustness analysis simple and single-request; reserve load-dependent multi-request effects for simulation. Compare directly against Continuum-style TTL under matched workloads and memory constraints, consistent with instructor feedback.

## 9. Artifact locations

| Artifact | Intended relative location when run from TraceLab root |
|---|---|
| Original reproduced round-gap table | `group_work/pauses.parquet` |
| Tool intervals | `group_work/tool_intervals.parquet` |
| Tool activity episodes | `group_work/tool_activity_episodes.parquet` |
| First-tool episode candidates | `group_work/tool_episode_candidates.parquet` |
| Audited episode candidates | `group_work/tool_episode_candidates_audited.parquet` |

Raw TraceLab files and generated Parquet outputs should be treated as separately managed data artifacts unless the team explicitly chooses a permitted distribution method. Scripts and this report are the primary source-controlled deliverables.
