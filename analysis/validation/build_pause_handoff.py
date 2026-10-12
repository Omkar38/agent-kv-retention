#!/usr/bin/env python3
"""Build a lossless Week 4 handoff from Brian's TraceLab pause table VERSION 2.

All Brian v2 rows and columns are retained verbatim in pause_handoff.parquet;
the model table keeps *all original columns*, plus QA and split columns, for
usable rows. Predictors MUST use the explicit feature_* columns/whitelist,
not the retrospective ground-truth, cause, or quality columns.

Run from the TraceLab root, with Brian v2 saved as group_work/pauses.parquet.
No raw TraceLab database is needed for this step.
"""
import argparse
import hashlib
from pathlib import Path


V2_REQUIRED = {
    "provider", "user", "session_id", "round_index", "model",
    "pause_s", "cause", "usable", "context_tokens", "n_tools",
    "tool_names", "tool_input_chars", "pause_start_at",
    "pause_end_at", "log_format",
}


def split_for(provider, session_id):
    """Keep the original Week 3 deterministic session-level 80/10/10 split."""
    key = f"tracelab-v0.0.1|{provider}|{session_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def resolved(base, p):
    p = Path(p).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def quote_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--brian-pauses", type=Path, default=Path("group_work/pauses.parquet"),
        help="Brian VERSION 2 pauses.parquet (NOT Version 1)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("group_work"),
        help="Output directory (default: TraceLab/group_work/)",
    )
    args = parser.parse_args()
    root = args.trace_root.expanduser().resolve()
    source = resolved(root, args.brian_pauses)
    dest = resolved(root, args.output_dir)
    if not source.is_file():
        parser.error(f"Missing {source}. Export Brian's VERSION 2 file first.")
    dest.mkdir(parents=True, exist_ok=True)
    if dest == source.parent and source.name in (
        "pause_handoff.parquet", "week4_pause_model.parquet"
    ):
        parser.error("Input would be overwritten by the output")

    import duckdb
    con = duckdb.connect()
    con.execute(f"CREATE TEMP VIEW brian_v2 AS SELECT * FROM read_parquet({quote_path(source)})")

    col_info = con.execute("DESCRIBE brian_v2").fetchall()
    columns = [c[0] for c in col_info]
    missing = sorted(V2_REQUIRED.difference(columns))
    if missing:
        raise ValueError(
            "Brian VERSION 2 schema required. Missing: " + ", ".join(missing)
            + ". Do not use the prior Version 1 pauses.parquet."
        )
    if any(c.startswith(("qa_", "feature_")) or c in (
        "split", "evaluation_group", "eligible_for_prediction",
        "handoff_record_id",
    ) for c in columns):
        raise ValueError("Unexpected handoff-specific columns in source; supply original Brian v2.")

    # Brian's 'usable' is retained as published. The QA indicators below are
    # additive; none overrides Brian's original labels or measurements.
    con.execute("""
      CREATE TEMP TABLE handoff_base AS
      SELECT b.*,
             ROW_NUMBER() OVER () AS handoff_record_id,
             EPOCH(b.pause_end_at - b.pause_start_at)
                AS qa_endpoint_duration_s,
             CASE
               WHEN b.pause_start_at IS NULL OR b.pause_end_at IS NULL
                 THEN 'missing_boundary'
               WHEN b.pause_end_at < b.pause_start_at THEN 'negative_boundary'
               WHEN b.pause_s IS NULL THEN 'missing_target'
               WHEN ABS(EPOCH(b.pause_end_at - b.pause_start_at) - b.pause_s) > 0.001
                 THEN 'target_endpoint_disagreement'
               ELSE 'consistent_with_v2_endpoints'
             END AS qa_boundary_status,
             CASE
               WHEN b.log_format = 'codex_old'
                 THEN 'old_codex_result_logged_tool_metadata'
               WHEN b.log_format IN ('codex','claude')
                 THEN 'source_round_tool_metadata'
               ELSE 'unknown_log_format'
             END AS qa_feature_origin,
             CASE
               WHEN b.n_tools > 1 THEN 'multiple_recorded_tools'
               WHEN b.n_tools = 1 THEN 'single_recorded_tool'
               WHEN b.n_tools = 0 THEN 'no_recorded_tools'
               ELSE 'tool_count_unknown'
             END AS qa_tool_topology,
             -- Cause is an OFFLINE label from the later resumption.
             -- It must never be fed to the actual online predictor.
             b.cause AS evaluation_group,
             -- Older Codex tool metadata can be recorded alongside results
             -- after the retention decision. Never expose it as a model input.
             CASE WHEN b.log_format IN ('codex','claude')
                  THEN b.n_tools ELSE NULL END AS feature_n_tools,
             CASE WHEN b.log_format IN ('codex','claude')
                  THEN b.tool_names ELSE NULL END AS feature_tool_names,
             CASE WHEN b.log_format IN ('codex','claude')
                  THEN b.tool_input_chars ELSE NULL END
                  AS feature_tool_input_chars,
             CASE WHEN b.usable IS TRUE AND b.pause_s IS NOT NULL
                       AND b.pause_s >= 0
                       AND b.pause_start_at IS NOT NULL
                       AND b.pause_end_at IS NOT NULL
                       AND b.pause_end_at >= b.pause_start_at
                       AND ABS(EPOCH(b.pause_end_at - b.pause_start_at)-b.pause_s) <= 0.001
                       AND b.provider IS NOT NULL AND b.session_id IS NOT NULL
                  THEN TRUE ELSE FALSE END AS eligible_for_prediction
      FROM brian_v2 b
    """)

    splits = con.execute("""
        SELECT DISTINCT provider, session_id FROM handoff_base
        WHERE provider IS NOT NULL AND session_id IS NOT NULL
    """).fetchall()
    con.execute("CREATE TEMP TABLE session_split(provider VARCHAR, session_id VARCHAR, split VARCHAR)")
    con.executemany("INSERT INTO session_split VALUES (?, ?, ?)", [
        (p, s, split_for(p, s)) for p, s in splits
    ])
    con.execute("""
      CREATE TEMP TABLE handoff AS
      SELECT h.*, s.split
      FROM handoff_base h LEFT JOIN session_split s USING (provider, session_id)
    """)
    n_source = con.execute("SELECT COUNT(*) FROM brian_v2").fetchone()[0]
    n_all = con.execute("SELECT COUNT(*) FROM handoff").fetchone()[0]
    if n_all != n_source:
        raise RuntimeError(f"Row count changed: Brian v2 {n_source} / handoff {n_all}")

    # Every Brian v2 column survives in BOTH exported datasets, including
    # cause, n_tools, model, context_tokens and original timestamps.
    full_columns = [c[0] for c in con.execute("DESCRIBE handoff").fetchall()]
    assert all(c in full_columns for c in columns)
    con.execute("""
      CREATE TEMP TABLE model_data AS
      SELECT * FROM handoff WHERE eligible_for_prediction
    """)
    n_eligible = con.execute("SELECT COUNT(*) FROM model_data").fetchone()[0]
    if n_eligible == 0:
        raise RuntimeError("No usable v2 records; wrong source version or invalid timestamps?")
    # Same key in different branches must never cross splits.
    crossing = con.execute("""
       SELECT COUNT(*) FROM (
           SELECT provider, session_id
           FROM model_data
           GROUP BY provider, session_id
           HAVING COUNT(DISTINCT split) > 1
       )
    """).fetchone()[0]
    if crossing:
        raise RuntimeError(f"{crossing} sessions cross a split boundary")
    ineligible_in_model = con.execute("""
       SELECT COUNT(*) FROM model_data
       WHERE pause_s IS NULL OR pause_s < 0 OR split IS NULL
         OR qa_boundary_status <> 'consistent_with_v2_endpoints'
    """).fetchone()[0]
    if ineligible_in_model:
        raise RuntimeError(f"{ineligible_in_model} noneligible rows in model_data")

    # Write only after all assertions, and never overwrite Brian's source.
    handoff_path = dest / "pause_handoff.parquet"
    model_path = dest / "week4_pause_model.parquet"
    if source in (handoff_path, model_path):
        raise RuntimeError("Refusing to overwrite Brian's source file")
    con.execute(f"COPY handoff TO {quote_path(handoff_path)} (FORMAT PARQUET)")
    con.execute(f"COPY model_data TO {quote_path(model_path)} (FORMAT PARQUET)")

    print("\n=== Version 2 source preservation and prediction coverage ===")
    print(f"Brian v2 original: {n_source:,} rows, {len(columns)} columns")
    print(f"Lossless augmented handoff: {n_all:,} rows; all source columns preserved")
    print(f"Week 4 model/evaluation table: {n_eligible:,} usable rows")
    print("\n=== Provider, log format, retrospective cause and eligibility ===")
    print(con.execute("""
       SELECT provider, log_format, cause,
              COUNT(*) AS all_rows,
              COUNT(*) FILTER (WHERE eligible_for_prediction) AS usable_rows
       FROM handoff GROUP BY provider, log_format, cause
       ORDER BY provider, log_format, all_rows DESC
    """).df().to_string(index=False))
    print("\n=== Cohort by split, including multiple tools ===")
    print(con.execute("""
       SELECT provider, split, qa_tool_topology, COUNT(*) AS rows,
              COUNT(DISTINCT session_id) AS sessions
       FROM model_data GROUP BY provider, split, qa_tool_topology
       ORDER BY provider, split, rows DESC
    """).df().to_string(index=False))
    print("\n=== Boundary checks ===")
    print(con.execute("""
        SELECT provider, qa_boundary_status, COUNT(*) AS rows
        FROM handoff GROUP BY provider, qa_boundary_status
        ORDER BY provider, rows DESC
    """).df().to_string(index=False))
    print("\n=== Masking of tool attributes from older Codex training inputs ===")
    print(con.execute("""
       SELECT log_format, COUNT(*) AS rows,
              COUNT(*) FILTER (WHERE feature_n_tools IS NULL) AS masked_n_tools,
              COUNT(*) FILTER (WHERE feature_tool_names IS NULL) AS missing_tool_names_feature
       FROM model_data GROUP BY log_format ORDER BY log_format
    """).df().to_string(index=False))
    print("\nSaved:", handoff_path)
    print("Saved:", model_path)
    print("IMPORTANT: cause, usable, duration endpoints, quality flags and original")
    print("old-Codex tool metadata are OFFLINE audit/evaluation fields, not model inputs.")
    print("No claim of observed inference-server KV-cache residency.")
    con.close()


if __name__ == "__main__":
    main()
