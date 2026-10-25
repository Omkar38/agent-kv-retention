#!/usr/bin/env python3
"""Build ONE lossless, feature-engineered Week 4 dataset from TraceLab pause Version 2.

Preserves ALL source rows and ALL 15 source columns unchanged (including the
legacy source field named 'usable', which is retained only for provenance).
Preserve categorical context and add decision-time candidate features and
separate offline target transformations for Week 4 regression/classification.

Output: group_work/pause_handoff.parquet. No second filtered parquet.
Usage: from TraceLab root, after obtaining Brian v2 as group_work/pauses.parquet:
    python path/to/build_pause_handoff.py

No physical KV-cache state or inference-server idle period is observed.
"""
import argparse
import hashlib
from pathlib import Path


REQUIRED_SOURCE_COLUMNS = {
    "provider", "user", "session_id", "round_index", "model",
    "pause_s", "cause", "usable", "context_tokens", "n_tools",
    "tool_names", "tool_input_chars", "pause_start_at",
    "pause_end_at", "log_format",
}


def session_split(provider, session_id):
    """Stable split compatible with the existing SHA-256 session split."""
    token = f"tracelab-v0.0.1|{provider}|{session_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def absolute_path(root, path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path.cwd())
    parser.add_argument("--source", "--brian-pauses", type=Path,
                        default=Path("group_work/pauses.parquet"),
                        help="Unmodified Version 2 pause table")
    parser.add_argument("--output-dir", type=Path, default=Path("group_work"))
    args = parser.parse_args()

    root = args.trace_root.expanduser().resolve()
    source = absolute_path(root, args.source)
    dest = absolute_path(root, args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / "pause_handoff.parquet"

    if not source.is_file():
        parser.error(f"Missing {source}. Regenerate Version 2 pause table first.")
    if output == source:
        parser.error("Source and output cannot be the same file.")

    import duckdb
    con = duckdb.connect()
    con.execute(f"CREATE TEMP VIEW source_pause AS SELECT * FROM read_parquet({sql_path(source)})")
    cols = [x[0] for x in con.execute("DESCRIBE source_pause").fetchall()]
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(cols))
    if missing:
        raise ValueError(
            "Expected source Version 2, not the old Version 1 table; missing "
            + ", ".join(missing)
        )
    extras = set(cols).intersection({
        "record_category", "timing_category", "feature_category", "analysis_track",
        "tool_topology", "split", "feature_n_tools", "feature_tool_names",
        "feature_tool_input_chars",
    })
    if extras:
        raise ValueError("Pass unmodified Version 2, not a previous handoff: " + str(extras))

    # No original field is removed or modified. Keep provenance diagnostics
    # but do not use the legacy source eligibility flag to discard records.
    # Retrospective pause cause and duration-derived labels are never inputs
    # to an online retention predictor.
    con.execute("""
    CREATE TEMP TABLE categorized AS
    WITH basic AS (
        SELECT b.*,
               EPOCH(b.pause_end_at - b.pause_start_at) AS endpoint_duration_s,
               CASE b.cause
                    WHEN 'tool' THEN 'tool_wait'
                    WHEN 'user' THEN 'human_wait'
                    WHEN 'tool+user' THEN 'mixed_tool_human_wait'
                    WHEN 'end' THEN 'session_end'
                    WHEN 'unknown' THEN 'unresolved_continuation'
                    ELSE 'other_cause'
               END AS record_category,
               CASE
                    WHEN b.n_tools > 1 THEN 'multiple_tool_calls'
                    WHEN b.n_tools = 1 THEN 'single_tool_call'
                    WHEN b.n_tools = 0 THEN 'no_tool_call'
                    ELSE 'tool_count_not_recorded'
               END AS tool_topology,
               CASE
                    WHEN b.pause_start_at IS NULL OR b.pause_end_at IS NULL
                         THEN 'boundary_missing'
                    WHEN b.pause_end_at < b.pause_start_at OR b.pause_s < 0
                         THEN 'negative_duration'
                    WHEN b.pause_s IS NULL THEN 'duration_missing'
                    WHEN ABS(EPOCH(b.pause_end_at - b.pause_start_at) - b.pause_s) > 0.001
                         THEN 'endpoint_disagreement'
                    WHEN b.usable IS NOT TRUE
                         THEN 'source_record_review'
                    ELSE 'timing_consistent'
               END AS timing_category,
               CASE
                    WHEN b.log_format = 'codex_old' AND b.n_tools > 0
                         THEN 'result_logged_tool_metadata'
                    WHEN b.log_format = 'codex_old'
                         THEN 'older_codex_no_tool'
                    WHEN b.log_format IN ('claude','codex') AND b.n_tools > 0
                         THEN 'source_round_tool_metadata'
                    WHEN b.log_format IN ('claude','codex') AND b.n_tools = 0
                         THEN 'source_round_no_tool'
                    ELSE 'feature_availability_unknown'
               END AS feature_category
        FROM source_pause b
    )
    SELECT basic.*,
           CASE
             WHEN timing_category <> 'timing_consistent' THEN 'boundary_or_source_review'
             WHEN record_category = 'tool_wait' THEN 'tool_pause_prediction'
             WHEN record_category = 'human_wait' THEN 'human_wait_prediction'
             WHEN record_category = 'mixed_tool_human_wait'
                  THEN 'mixed_wait_prediction'
             ELSE 'diagnostic_event'
           END AS analysis_track,
           CASE WHEN feature_category = 'source_round_tool_metadata'
                  OR feature_category = 'source_round_no_tool'
                THEN n_tools ELSE NULL END AS feature_n_tools,
           CASE WHEN feature_category = 'source_round_tool_metadata'
                THEN tool_names ELSE NULL END AS feature_tool_names,
           CASE WHEN feature_category = 'source_round_tool_metadata'
                THEN tool_input_chars ELSE NULL END AS feature_tool_input_chars,
           -- Numeric features derived only from original decision-time
           -- candidate fields. Negative/absent values remain NULL.
           CASE WHEN context_tokens >= 0
                THEN LN(1.0 + context_tokens) END AS feature_log1p_context_tokens,
           CASE WHEN round_index >= 0
                THEN LN(1.0 + round_index) END AS feature_log1p_round_index,
           CASE WHEN feature_category = 'source_round_tool_metadata'
                     AND tool_input_chars >= 0
                THEN LN(1.0 + tool_input_chars)
           END AS feature_log1p_tool_input_chars,
           CASE WHEN feature_category IN (
                      'source_round_tool_metadata','source_round_no_tool'
                    ) THEN n_tools > 0 END AS feature_has_tools,
           CASE WHEN feature_category IN (
                      'source_round_tool_metadata','source_round_no_tool'
                    ) THEN n_tools > 1 END AS feature_multiple_tools,
           -- Offline target transforms/labels for evaluation only:
           -- NEVER include target_* or label_* in model features.
           CASE WHEN pause_s >= 0
                THEN LN(1.0 + pause_s) END AS target_log1p_pause_s,
           CASE WHEN pause_s >= 0
                THEN pause_s > 10 END AS label_pause_gt_10s,
           CASE WHEN pause_s >= 0
                THEN pause_s > 60 END AS label_pause_gt_60s
    FROM basic
    """)

    # Session split is group-based: all records (including review-only records)
    # in one provider/session get the same partition.
    sessions = con.execute("""
      SELECT DISTINCT provider, session_id FROM categorized
      WHERE provider IS NOT NULL AND session_id IS NOT NULL
    """).fetchall()
    con.execute("""
      CREATE TEMP TABLE session_partition(
          provider VARCHAR, session_id VARCHAR, split VARCHAR
      )
    """)
    if sessions:
        con.executemany("INSERT INTO session_partition VALUES (?, ?, ?)", [
            (provider, sid, session_split(provider, sid))
            for provider, sid in sessions
        ])
    con.execute("""
      CREATE TEMP TABLE handoff AS
      SELECT c.*, p.split
      FROM categorized c LEFT JOIN session_partition p USING (provider, session_id)
    """)

    expected = con.execute("SELECT COUNT(*) FROM source_pause").fetchone()[0]
    observed = con.execute("SELECT COUNT(*) FROM handoff").fetchone()[0]
    if observed != expected:
        raise RuntimeError(f"Lost or duplicated source rows ({expected} -> {observed}).")

    result_columns = [c[0] for c in con.execute("DESCRIBE handoff").fetchall()]
    if not set(cols).issubset(result_columns):
        raise RuntimeError("At least one original source column is missing.")
    for field in ["record_category", "timing_category", "feature_category",
                  "analysis_track", "tool_topology"]:
        no_label = con.execute(
            f'SELECT COUNT(*) FROM handoff WHERE "{field}" IS NULL'
        ).fetchone()[0]
        if no_label:
            raise RuntimeError(f"Missing {field} label in {no_label} rows")

    crossover = con.execute("""
      SELECT COUNT(*) FROM (
        SELECT provider, session_id FROM handoff
        WHERE provider IS NOT NULL AND session_id IS NOT NULL
        GROUP BY provider, session_id
        HAVING COUNT(DISTINCT split) > 1
      )
    """).fetchone()[0]
    if crossover:
        raise RuntimeError(f"{crossover} sessions assigned to multiple partitions.")

    # Verify exact source identity for source columns: b.* is not mutated.
    # DuckDB's EXCEPT ALL catches loss, extra copies, or modified source rows,
    # including duplicate records.
    source_list = ", ".join('"' + x.replace('"', '""') + '"' for x in cols)
    mismatch = con.execute(f"""
      SELECT COUNT(*) FROM (
        (SELECT {source_list} FROM source_pause
           EXCEPT ALL SELECT {source_list} FROM handoff)
        UNION ALL
        (SELECT {source_list} FROM handoff
           EXCEPT ALL SELECT {source_list} FROM source_pause)
      )
    """).fetchone()[0]
    if mismatch:
        raise RuntimeError(f"{mismatch} source-column rows changed; aborting.")

    con.execute(f"COPY handoff TO {sql_path(output)} (FORMAT PARQUET)")
    print("\n=== Complete categorized v2 handoff ===")
    print(f"Preserved {observed:,} source records and all {len(cols)} v2 columns unchanged.")
    print("One output:", output)
    print("\n=== Event categories and timing categories ===")
    print(con.execute("""
       SELECT record_category, timing_category, COUNT(*) AS rows
       FROM handoff GROUP BY ALL ORDER BY rows DESC
    """).df().to_string(index=False))
    print("\n=== Prediction/evaluation tracks ===")
    print(con.execute("""
       SELECT provider, log_format, analysis_track,
              COUNT(*) AS rows
       FROM handoff GROUP BY ALL ORDER BY provider, log_format, rows DESC
    """).df().to_string(index=False))
    print("\n=== Tool-call topology within prediction tracks ===")
    print(con.execute("""
       SELECT analysis_track, tool_topology, COUNT(*) AS rows
       FROM handoff GROUP BY ALL ORDER BY analysis_track, rows DESC
    """).df().to_string(index=False))
    print("\n=== Session-based partitions ===")
    print(con.execute("""
       SELECT split, COUNT(*) AS rows, COUNT(DISTINCT session_id) AS sessions
       FROM handoff GROUP BY split ORDER BY split
    """).df().to_string(index=False))
    print("\nIMPORTANT: model inputs must exclude pause_s, target_*, label_*,"
          "\n cause/record_category/analysis_track/timing_category/split,"
          "\n session/user identifiers, and outcome timestamps."
          "\n The original Version 2 source schema remains unchanged."
          "\n No GPU KV-cache residency is measured.")
    con.close()


if __name__ == "__main__":
    main()
