# TraceLab Workload Analysis

## Purpose

These scripts perform exploratory workload analysis for the agent KV-cache retention project using the TraceLab dataset. The link to the dataset can be found here: https://github.com/uw-syfi/TraceLab.

## Scripts

- `check_tool_latency.py` — Examines individual tool-call latency distributions.
- `check_session_latency.py` — Examines latency at the session level.
- `check_tool_thresholds.py` — Examines how frequently tool calls exceed selected duration thresholds.
- `check_prediction_fields.py` — Inspects available data fields and potential features for predicting tool-call duration.

## Data Requirements

These scripts were developed using the TraceLab dataset. The database is not included in this repository and must be obtained separately.

## Limitations

These scripts support exploratory workload analysis. Their findings motivate further investigation but do not, by themselves, establish that prediction-aware KV-cache retention improves performance.

## Reproducibility

The expected database path, required Python environment, and execution instructions should be verified against each script before reproducing the results.