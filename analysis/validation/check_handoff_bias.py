#!/usr/bin/env python3
"""Summarize coverage, tails, and cause/format differences in Brian v2 handoff."""
import argparse
from pathlib import Path
import duckdb

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path.cwd())
    parser.add_argument("--handoff", type=Path, default=Path("group_work/pause_handoff.parquet"))
    args = parser.parse_args()
    root = args.trace_root.expanduser().resolve()
    path = args.handoff if args.handoff.is_absolute() else root / args.handoff
    if not path.exists():
        parser.error(f"Missing {path}; run build_pause_handoff.py on Brian v2 first.")
    db = duckdb.connect()
    db.execute("CREATE TEMP VIEW pauses AS SELECT * FROM read_parquet(\'" + str(path).replace("\'", "\'\'") + "\')")
    print("\n=== Original v2 records retained by provider, log format, and cause ===")
    print(db.execute("""
        SELECT provider, log_format, cause, COUNT(*) AS all_records,
               COUNT(*) FILTER (WHERE eligible_for_prediction) AS eligible_records,
               ROUND(100.0*COUNT(*) FILTER (WHERE eligible_for_prediction)/COUNT(*), 2)
                   AS eligible_pct
        FROM pauses GROUP BY provider, log_format, cause
        ORDER BY provider, log_format, all_records DESC
    """).df().to_string(index=False))
    print("\n=== Usable pause duration by log format and cause ===")
    print(db.execute("""
        SELECT provider, log_format, cause, COUNT(*) AS rows,
               ROUND(MEDIAN(pause_s), 3) AS p50_s,
               ROUND(QUANTILE_CONT(pause_s, 0.90), 3) AS p90_s,
               ROUND(QUANTILE_CONT(pause_s, 0.99), 3) AS p99_s,
               ROUND(100.0*COUNT(*) FILTER (WHERE pause_s > 10)/COUNT(*), 2)
                   AS pct_gt_10s,
               ROUND(100.0*COUNT(*) FILTER (WHERE pause_s > 60)/COUNT(*), 2)
                   AS pct_gt_60s
        FROM pauses WHERE eligible_for_prediction
        GROUP BY provider, log_format, cause
        ORDER BY provider, log_format, rows DESC
    """).df().to_string(index=False))
    print("\n=== Single and multi-tool coverage ===")
    print(db.execute("""
        SELECT provider, log_format, qa_tool_topology,
               COUNT(*) AS original_rows,
               COUNT(*) FILTER (WHERE eligible_for_prediction) AS eligible_rows
        FROM pauses GROUP BY provider, log_format, qa_tool_topology
        ORDER BY provider, log_format, original_rows DESC
    """).df().to_string(index=False))
    print("\n=== Exclusion reasons for originally usable rows ===")
    print(db.execute("""
        SELECT provider, log_format, qa_boundary_status,
               COUNT(*) AS excluded_rows
        FROM pauses WHERE usable AND NOT eligible_for_prediction
        GROUP BY provider, log_format, qa_boundary_status
        ORDER BY excluded_rows DESC
    """).df().to_string(index=False))
    db.close()

if __name__ == "__main__":
    main()
