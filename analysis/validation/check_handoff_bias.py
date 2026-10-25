#!/usr/bin/env python3
"""Analyze the complete categorized Version 2 handoff without dropping rows."""
import argparse
from pathlib import Path
import duckdb

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-root", type=Path, default=Path.cwd())
    ap.add_argument("--handoff", type=Path, default=Path("group_work/pause_handoff.parquet"))
    args = ap.parse_args()
    root = args.trace_root.expanduser().resolve()
    file = args.handoff if args.handoff.is_absolute() else root / args.handoff
    if not file.is_file():
        ap.error(f"Missing {file}; run build_pause_handoff.py first.")
    db = duckdb.connect()
    src = "'" + str(file).replace("'", "''") + "'"
    db.execute(f"CREATE TEMP VIEW p AS SELECT * FROM read_parquet({src})")
    print("\n=== All records: category, log format, and timing ===")
    print(db.execute("""
      SELECT provider, log_format, record_category, timing_category,
             COUNT(*) AS records
      FROM p GROUP BY ALL ORDER BY provider, log_format, records DESC
    """).df().to_string(index=False))
    print("\n=== Duration distributions by analysis track ===")
    print(db.execute("""
      SELECT provider, log_format, analysis_track, COUNT(*) AS records,
             ROUND(MEDIAN(pause_s), 3) AS p50_s,
             ROUND(QUANTILE_CONT(pause_s, 0.90), 3) AS p90_s,
             ROUND(QUANTILE_CONT(pause_s, 0.99), 3) AS p99_s,
             ROUND(100.0*COUNT(*) FILTER (WHERE pause_s > 10)
                   / NULLIF(COUNT(pause_s),0),2) AS pct_gt_10s,
             ROUND(100.0*COUNT(*) FILTER (WHERE pause_s > 60)
                   / NULLIF(COUNT(pause_s),0),2) AS pct_gt_60s
      FROM p GROUP BY ALL ORDER BY provider, log_format, records DESC
    """).df().to_string(index=False))
    print("\n=== Parallel/multi-tool topology and feature origin ===")
    print(db.execute("""
      SELECT provider, log_format, analysis_track, tool_topology,
             feature_category, COUNT(*) AS records
      FROM p GROUP BY ALL ORDER BY records DESC
    """).df().to_string(index=False))
    print("\n=== Total records and train/validation/test ===")
    print(db.execute("""
      SELECT split, COUNT(*) AS records,
             COUNT(DISTINCT session_id) AS sessions
      FROM p GROUP BY split ORDER BY split
    """).df().to_string(index=False))
    db.close()

if __name__ == "__main__":
    main()
