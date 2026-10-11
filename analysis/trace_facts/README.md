# TraceLab workload statistics

This folder contains exploratory descriptive analyses originally written by **David Mei** and consolidated here for reproducible execution. These complement Brian's round-gap processing and Hangting's timing-validation and Week 4 pause-handoff workflow.

## Script

Run from the TraceLab root after downloading `trace/syfi_coding_trace.duckdb` (TraceLab v0.0.1):

```bash
REPO=~/kv-week3-work/agent-kv-retention-hangting
python "$REPO/analysis/trace_facts/trace_statistics.py" --step all
```

Or use `--step tool_latency`, `--step session_latency`, `--step tool_thresholds`, or `--step prediction_fields`. The script accepts `--trace-root` if invoked from another directory. Dependency: Python with `duckdb` and `pandas`.

| Step | Analysis |
|---|---|
| `tool_latency` | Distribution (mean, p50, p90, p95, p99, p99.9) of valid wall latencies for selected execution tools |
| `session_latency` | Per-session counts, threshold frequencies, **sums of individual call durations** and maxima |
| `tool_thresholds` | Valid-call frequencies above 5, 10, 30 and 60 seconds by execution tool |
| `prediction_fields` | TraceLab field samples and coverage; does **not** establish that values are available at retention-decision time |

**Important:** Summing per-tool durations within sessions double-counts overlap; the resulting figure is **not elapsed time spent waiting**. Post-result fields (`result_at`, `is_error`, result sizes, latency) must never be supplied to a decision-time predictor. Even available round fields require timing validation.

The original four small scripts are consolidated into `trace_statistics.py` **on this branch**; their prior content remains available in Git history and David's original work. See [../validation/README.md](../validation/README.md) for the Week 3 pipeline and [../dataset/README.md](../dataset/README.md) for Week 4 handoff data.
