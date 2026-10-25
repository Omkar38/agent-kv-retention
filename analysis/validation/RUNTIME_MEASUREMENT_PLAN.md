# Runtime Measurement Plan — From Agent Waits to GPU KV-Cache Idle Intervals

**Project:** Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents  
**Purpose:** Establish empirically observed retention-decision times, inference resumptions, and KV-cache residency on a controlled local inference server.

This plan follows the TraceLab processing and classification in [README.md](README.md) and [WEEK3_VALIDATION.md](WEEK3_VALIDATION.md). The existing `agent_waits_classified.parquet` contains **trace-level tool-activity proxies**. It must not be relabeled as exact GPU KV-cache idle time.

## 1. What can and cannot be identified

- **TraceLab (Claude Code / Codex, closed-model serving):** tool-call dispatch/completion, round and event timestamps, approximate continuation signals. The actual back-end GPU scheduler, KV memory allocation and cache lifecycle are **not observed**. Thus original-trace GPU-cache idle times are not identifiable exactly.
- **Controlled replay on an instrumented vLLM/SGLang deployment:** per-request scheduler transitions, cache block allocation/free, tool completion, and resume times can be measured. This provides **measured GPU KV-cache idle intervals for the replay**, not retrospective ground truth for the original Claude/Codex GPU clusters.
- **Trace-driven simulator:** uses a declared mapping from TraceLab intervals to simulated pauses plus hardware-calibrated costs. This gives an explicitly **modeled** workload; do not call it exact original-cluster KV residency.

## 2. Three distinct time quantities

For each session continuation, use a stable `session_id` and `step_id` (or `pause_id`).

| Quantity | Boundaries | Interpretation |
|---|---|---|
| External ready time | `generation_finished` → `continuation_ready` | Tool/user/application work until the next request can be submitted; ideally separate tool finish from application handling |
| Inference inter-step gap | `generation_finished` → `resume_compute_started` | Includes external wait **plus** server queue delay; **not** itself a measure of cache residency |
| Actual held-idle GPU KV | Scheduler not actively using the session **and** cache blocks physically allocated/retained for that session | May comprise zero or multiple subintervals. Measure resident **KV byte-seconds** as well as elapsed time |

A cache-evicted pause can have substantial external waiting time and **zero held-idle KV time after eviction**. Shared/prefix-cache blocks need ownership/refcount accounting rather than naively charging their whole size to every session.

## 3. Minimum event contract

Emit line-delimited JSON records from the **agent orchestrator** and **inference server**, with a consistent correlation ID. Include `schema_version`, `session_id`, `step_id`, `event`, a monotonic timestamp (e.g., `monotonic_ns`) when recorded in the same clock domain, and a wall timestamp for cross-process correlation. Do not directly subtract monotonic timestamps from different machines without synchronizing/calibrating their clocks.

Required event types (actual instrumentation names may vary by engine):

| Event | Emitter | Why |
|---|---|---|
| `generation_finished` | Server scheduler | True point that model processing finishes and a retention decision can begin |
| `kv_retention_decision` | Scheduler / retention policy | TTL, predicted duration, cache allocation, and chosen keep/offload/evict action |
| `kv_block_state_changed` | KV memory allocator / scheduler | Resident/evicted/offloaded blocks, bytes and block ownership (including shared/refcounted blocks) |
| `tool_dispatched`, `tool_finished` | Agent orchestrator | External action timing, with `tool_call_id` |
| `continuation_ready`, `resume_submitted` | Agent orchestrator | Separates tool ready time and enqueue time |
| `resume_compute_started` | Server scheduler | Start of actual prefill/decode for next step |
| `prefill_finished`, `generation_finished` | Server scheduler | Recompute and generation timing |
| `queue_size`, `gpu_kv_blocks_used` | Scheduler / allocator | Context for load- and memory-dependent opportunity cost |

Also record model ID, context/prefix token count, engine version and config, cache block size, KV dtype, GPU hardware, concurrency, and cache policy. These are necessary for comparing runs.

An illustrative event (format contract, not output of TraceLab):

```json
{"schema_version":1,"clock_id":"server-0","monotonic_ns":123456789000,"session_id":"demo-1","step_id":"3","event":"generation_finished","resident_kv_bytes":1073741824}
```

## 4. Instrumentation and controlled replay workflow

1. **Choose the serving engine and fixed GPU setup.** Start with one GPU, one model, one concurrency level, and a single explicit cache-retention implementation. Record engine/commit versions. Confirm whether a *session-resident KV cache* actually exists: generic prefix caching may instead reuse shared blocks without preserving a session-owned allocation.
2. **Add server hooks** for generation completion, request admission/resume, and block-level cache allocation/eviction; use a stable session/pause correlation identifier. Add orchestrator hooks for tool dispatch, completion and resume request submission.
3. **Controlled single-session tests:** run a known tool (e.g. a sleep operation) for several prescribed durations; first choose an always-retain policy, then immediate-eviction, then a fixed TTL. Verify the logged chronology and byte residency against the policy. Introduce multi-tool concurrency and a second competing request only after single-session validation.
4. **Construct measured pause records** from event streams, distinguishing `external_wait_s`, `enqueue_wait_s`, `idle_kv_resident_s`, `idle_kv_byte_seconds`, `eviction_time_s`, and `recompute_s`. Preserve absolute timestamps and exclusion flags rather than silently repairing missing/negative records.
5. **Replay representative TraceLab tool-delay and context-size distributions** via the orchestrator, keeping the same workload and memory capacity across immediate eviction, fixed TTL, Continuum-style TTL and the learning-augmented policy. Clearly label replay results as local engine measurements, not measurements of TraceLab's original back-end GPUs.

## 5. Acceptance checks before labeling a cache interval “measured”

- A pause has explicit server-observed `generation_finished` and the relevant subsequent `resume_compute_started` or a documented terminal/censored outcome.
- Resident allocation, release/eviction and scheduler-active intervals are measurable and associated with the correct request/session. Shared KV blocks are handled consistently.
- All compared durations lie in comparable clock domains; event ordering discrepancies, missing events, duplicate IDs, and clock discontinuities are reported.
- The observed cache residence matches policy semantics in controlled always-keep, eviction and TTL tests; measured byte-seconds can be reconstructed from recorded allocation transitions.
- Tool wait, application handoff, server queueing and recompute are **separately** attributed.
- Hardware, engine version, block size, concurrency, workload seeds and policy parameters are recorded for reproducibility.

If any check fails, label the record `unverified` and retain the exclusion reason. Do not interpret a TraceLab `generation_signal_ts` as the server's actual `generation_finished`.

## 6. Outputs and downstream use

Suggested additional deliverables (only after runtime instrumentation is implemented):

- `runtime/events.jsonl` — structured orchestrator/scheduler/cache events.
- `runtime/measured_cache_idle.parquet` — per-pause held-idle durations, byte-seconds, queueing/recompute, and quality flags.
- `runtime/calibration.json` — model/GPU/cost calibration and engine configuration.
- `group_work/pauses_v2.parquet` — **trace-reconstructed simulator inputs**, which should remain distinguishable from measured GPU observations.

**Week 4:** use audited tool-activity durations for preliminary prediction and session-level splitting, excluding retrospective features from model inputs.  
**Weeks 6–7:** choose a validated simulation pause definition and calibrate the (c) and (B) terms on measured server behavior; compare policies under identical traces and memory budgets.  
**Weeks 8–10:** repeat with concurrent requests, prediction shift and long-tail workloads, reporting tail latency, recomputation and byte-second memory opportunity cost.

## 7. Immediate next decision

Confirm the available GPU, target model, and runtime engine (vLLM or SGLang). Once known, implement the **minimum event hooks** for one orchestrated tool call and perform the three single-session smoke tests before generating new datasets.
