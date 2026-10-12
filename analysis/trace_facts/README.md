# TraceLab exploratory statistics

Run `trace_statistics.py` from the TraceLab root with `--step all` or select `tool_latency`, `session_latency`, `tool_thresholds`, or `prediction_fields`. These are descriptive measurements of tool-call latency, per-session aggregates, tail thresholds, and available trace fields.

Summing individual tool durations can double-count concurrent calls; totals are not wall-clock session idle time. Recorded field presence alone does not prove decision-time feature availability. These analyses supplement the unified [Version 2 handoff](../dataset/README.md). The raw dataset is the [TraceLab release](https://github.com/uw-syfi/TraceLab).
