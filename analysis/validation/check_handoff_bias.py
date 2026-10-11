#!/usr/bin/env python3
"""Report selection bias in Brian/Hangting Week 4 pause handoff."""
import argparse
from pathlib import Path
import duckdb

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    path = args.trace_root.expanduser().resolve() / "group_work/pauses_v2.parquet"
    if not path.exists():
        parser.error(f"Missing {path}; first run build_pause_handoff.py")
    con = duckdb.connect()
    q = str(path).replace("'", "''")
    con.execute(f"CREATE TEMP VIEW p AS SELECT * FROM read_parquet('{q}')")
    print("\n=== Provider, original cause and strict eligibility ===")
    print(con.execute("""
      SELECT provider, brian_cause,
             COUNT(*) AS all_rows,
             COUNT(*) FILTER (WHERE week4_strict_eligible) AS strict_rows,
             ROUND(100.0*COUNT(*) FILTER (WHERE week4_strict_eligible)/COUNT(*), 2) AS strict_pct
      FROM p GROUP BY provider, brian_cause
      ORDER BY provider, all_rows DESC
    """).df().to_string(index=False))
    print("\n=== Pause duration distribution: strict versus excluded original-usable tools ===")
    print(con.execute("""
      SELECT provider, week4_strict_eligible AS strict,
             COUNT(*) AS rows,
             ROUND(MEDIAN(brian_pause_s), 3) AS p50_s,
             ROUND(QUANTILE_CONT(brian_pause_s, .90), 3) AS p90_s,
             ROUND(QUANTILE_CONT(brian_pause_s, .99), 3) AS p99_s,
             ROUND(100.0*AVG(CASE WHEN brian_pause_s > 10 THEN 1 ELSE 0 END), 2) AS pct_over_10s,
             ROUND(100.0*AVG(CASE WHEN brian_pause_s > 60 THEN 1 ELSE 0 END), 2) AS pct_over_60s
      FROM p
      WHERE brian_usable AND brian_cause = 'tool' AND brian_pause_s >= 0
      GROUP BY provider, week4_strict_eligible
      ORDER BY provider, strict DESC
    """).df().to_string(index=False))
    print("\n=== Quality reason: duration and long-tail frequency ===")
    print(con.execute("""
      SELECT provider, quality_reason, COUNT(*) AS rows,
             ROUND(MEDIAN(brian_pause_s) FILTER (WHERE brian_pause_s >= 0), 3) AS median_nonnegative_s,
             ROUND(100.0*COUNT(*) FILTER (WHERE brian_pause_s > 10)
                / NULLIF(COUNT(*) FILTER (WHERE brian_pause_s >= 0), 0), 2) AS pct_gt_10s_nonnegative
      FROM p GROUP BY provider, quality_reason
      ORDER BY provider, rows DESC
    """).df().to_string(index=False))
    con.close()

if __name__ == "__main__":
    main()
