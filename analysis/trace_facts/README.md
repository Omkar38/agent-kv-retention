# TraceLab exploratory statistics

Descriptive measurements of tool-call latency, per-session aggregates, tail thresholds, and
available trace fields, computed against the [TraceLab release](https://github.com/uw-syfi/TraceLab).

Two implementations of the same analysis live here. They were written independently in Week 3
and are kept both because they cross-check each other.

## Unified script

`trace_statistics.py` — run from the TraceLab root with `--step all`, or select one of
`tool_latency`, `session_latency`, `tool_thresholds`, `prediction_fields`.

## Per-step scripts

| Script | What it examines |
| --- | --- |
| `check_tool_latency.py` | Individual tool-call latency distributions |
| `check_session_latency.py` | Latency at the session level |
| `check_tool_thresholds.py` | How often tool calls exceed selected duration thresholds |
| `check_prediction_fields.py` | Available fields and candidate features for duration prediction |

## Data requirements

The TraceLab database is not included in this repository and must be obtained separately.
Verify the expected database path and Python environment against each script before reproducing.

## Limitations

Summing individual tool durations can double-count concurrent calls; totals are **not** wall-clock
session idle time. Recorded field presence alone does not prove decision-time feature availability.
These analyses motivate further investigation but do not by themselves establish that
prediction-aware KV-cache retention improves performance.

They supplement the unified [Version 2 handoff](../dataset/README.md), which is the authoritative
dataset for Week 4 onward.
