# Unified Version 2 handoff for Week 4

The pipeline is intentionally **one dataset, no deleted source rows**. The authoritative artifact is [`../dataset/pause_handoff.parquet`](../dataset/pause_handoff.parquet). The build script [`build_pause_handoff.py`](build_pause_handoff.py) preserves all original Version 2 columns and source rows, and annotates records with cause, timing, tool topology, feature-origin, and analysis-track categories, plus Week 4 numeric transforms and offline long-pause labels. The legacy source `usable` flag remains unchanged for exact upstream provenance; no added field uses the terms "usable" or "unusable".

See [dataset README](../dataset/README.md) for the exact schema, latest regeneration command, leakage precautions, and the single-artifact workflow.

The feature-engineered output contains 33 columns (15 original and 18 additional) after regeneration. Source rows are never discarded. The first Version 2 extraction produced **357,161** total rows, with upstream completeness criteria met for **351,497** and remaining **5,664** needing separate record-category or timing treatment. Those figures are **from the prior run**. The newly categorized artifact must be regenerated and checked before reporting the new category breakdown.

**Do not interpret all rows as supervised training examples.** Session-ending and unknown-continuation cases can lack an observed pause length. The data remains complete for auditing, while analysts choose an appropriate `analysis_track` within the same Parquet file. Human, mixed, and tool pauses are differentiated; older Codex metadata is masked from the derived `feature_*` tool fields when uncertain. All logging formats remain available for Week 4; investigation of Codex pause discrepancies is deferred. Retention decisions never know the retrospective cause in advance.

Train/validation/test are assigned deterministically by provider and session using the original SHA-256 rule. No source rows or source columns are overwritten, and the dataset is not a physical GPU KV-cache residency measurement.
