#!/usr/bin/env python3
"""TraceLab descriptive tool, session, and field analyses.

Consolidated from four original Week 3 exploratory scripts on this branch.
Run from the TraceLab root directory. Usage:
    python trace_statistics.py --step all
    python trace_statistics.py --step session_latency

All results are descriptive. Summed tool-call durations double count concurrent
and overlapping calls; never equate the sum with actual wall-clock agent wait.
"""
import argparse
import os
from pathlib import Path

def tool_latency():
    import duckdb
    
    connection = duckdb.connect(
        "trace/syfi_coding_trace.duckdb",
        read_only=True,
    )
    
    query = """
    SELECT
        tool_name,
        COUNT(*) AS calls,
        ROUND(AVG(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END)) / 1000.0, 3) AS mean_seconds,
        ROUND(MEDIAN(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END)) / 1000.0, 3) AS median_seconds,
    
        ROUND(
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END), 0.50) / 1000.0,
            3
        ) AS p50_seconds,
    
        ROUND(
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END), 0.90) / 1000.0,
            3
        ) AS p90_seconds,
    
        ROUND(
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END), 0.95) / 1000.0,
            3
        ) AS p95_seconds,
    
        ROUND(
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END), 0.99) / 1000.0,
            3
        ) AS p99_seconds,
    
        ROUND(
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END), 0.999) / 1000.0,
            3
        ) AS p999_seconds
    
    FROM tool_calls
    
    WHERE tool_name IN (
        'Bash',
        'exec_command',
        'write_stdin',
        'exec',
        'shell_command'
    )
    
    AND COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) IS NOT NULL
    AND COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) >= 0
    
    GROUP BY tool_name
    
    ORDER BY calls DESC
    """
    
    results = connection.execute(query).fetchall()
    
    print("\nAutomated execution tools:\n")
    
    print(
        f"{'Tool':20} "
        f"{'Calls':>10} "
        f"{'Mean':>10} "
        f"{'P50':>10} "
        f"{'P90':>10} "
        f"{'P95':>10} "
        f"{'P99':>10} "
        f"{'P99.9':>10}"
    )
    
    print("-" * 100)
    
    for row in results:
        (
            tool, calls, mean, median,
            p50, p90, p95, p99, p999
        ) = row
        print(
    	f"{str(tool):20} "
            f"{calls:10,} "
            f"{mean:10} "
            f"{p50:10} "
            f"{p90:10} "
            f"{p95:10} "
            f"{p99:10} "
            f"{p999:10}"
        )
    
    connection.close()


def session_latency():
    print("\\n[Note] Session total is a SUM of per-call durations; overlaps are double-counted. It is NOT elapsed session wait.")
    import duckdb
    
    
    # Connect to the TraceLab DuckDB database.
    connection = duckdb.connect(
        "trace/syfi_coding_trace.duckdb",
        read_only=True,
    )
    
    
    # ------------------------------------------------------------
    # SESSION-LEVEL TOOL LATENCY
    #
    # Effective duration uses internal runner latency when available; wall fallback:
    #
    #     tool_internal_latency_ms first; otherwise result_at - emitted_at
    #
    # This is a runner-first execution-duration estimate for individual
    # tool calls.
    #
    # We keep Claude and Codex separate because their tool
    # semantics are different.
    # ------------------------------------------------------------
    
    query = """
    WITH tool_data AS (
        SELECT
            r.provider,
            r.session_id,
            tc.tool_name,
            COALESCE(CASE WHEN tc.tool_internal_latency_ms >= 0 THEN tc.tool_internal_latency_ms END, CASE WHEN tc.tool_wall_latency_ms >= 0 THEN tc.tool_wall_latency_ms END) / 1000.0 AS latency_seconds
        FROM tool_calls AS tc
        JOIN rounds AS r
            USING (round_pk)
    
        WHERE COALESCE(CASE WHEN tc.tool_internal_latency_ms >= 0 THEN tc.tool_internal_latency_ms END, CASE WHEN tc.tool_wall_latency_ms >= 0 THEN tc.tool_wall_latency_ms END) IS NOT NULL
          AND COALESCE(CASE WHEN tc.tool_internal_latency_ms >= 0 THEN tc.tool_internal_latency_ms END, CASE WHEN tc.tool_wall_latency_ms >= 0 THEN tc.tool_wall_latency_ms END) >= 0
    ),
    
    session_stats AS (
        SELECT
            provider,
            session_id,
    
            COUNT(*) AS tool_calls,
    
            SUM(latency_seconds) AS total_tool_wait_seconds,
    
            MEDIAN(latency_seconds) AS median_tool_latency_seconds,
    
            MAX(latency_seconds) AS max_tool_latency_seconds,
    
            COUNT(*) FILTER (
                WHERE latency_seconds >= 5
            ) AS calls_over_5s,
    
            COUNT(*) FILTER (
                WHERE latency_seconds >= 10
            ) AS calls_over_10s,
    
            COUNT(*) FILTER (
                WHERE latency_seconds >= 30
            ) AS calls_over_30s,
    
            COUNT(*) FILTER (
                WHERE latency_seconds >= 60
            ) AS calls_over_60s
    
        FROM tool_data
    
        GROUP BY
            provider,
            session_id
    )
    
    SELECT
        provider,
    
        COUNT(*) AS sessions,
    
        ROUND(AVG(tool_calls), 2)
            AS avg_tool_calls_per_session,
    
        ROUND(MEDIAN(tool_calls), 2)
            AS median_tool_calls_per_session,
    
        ROUND(AVG(total_tool_wait_seconds), 2)
            AS avg_total_tool_wait_seconds,
    
        ROUND(MEDIAN(total_tool_wait_seconds), 2)
            AS median_total_tool_wait_seconds,
    
        ROUND(
            quantile_cont(total_tool_wait_seconds, 0.90),
            2
        ) AS p90_total_tool_wait_seconds,
    
        ROUND(
            quantile_cont(total_tool_wait_seconds, 0.99),
            2
        ) AS p99_total_tool_wait_seconds,
    
        ROUND(
            AVG(median_tool_latency_seconds),
            3
        ) AS avg_session_median_tool_latency_seconds,
    
        ROUND(
            quantile_cont(max_tool_latency_seconds, 0.50),
            3
        ) AS median_of_session_max_latency_seconds,
    
        ROUND(
            quantile_cont(max_tool_latency_seconds, 0.90),
            3
        ) AS p90_of_session_max_latency_seconds,
    
        ROUND(
            quantile_cont(max_tool_latency_seconds, 0.99),
            3
        ) AS p99_of_session_max_latency_seconds,
    
        ROUND(
            100.0 * AVG(
                CASE
                    WHEN calls_over_5s > 0 THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS pct_sessions_with_call_over_5s,
    
        ROUND(
            100.0 * AVG(
                CASE
                    WHEN calls_over_10s > 0 THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS pct_sessions_with_call_over_10s,
    
        ROUND(
            100.0 * AVG(
                CASE
                    WHEN calls_over_30s > 0 THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS pct_sessions_with_call_over_30s,
    
        ROUND(
            100.0 * AVG(
                CASE
                    WHEN calls_over_60s > 0 THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS pct_sessions_with_call_over_60s
    
    FROM session_stats
    
    GROUP BY provider
    
    ORDER BY provider
    """
    
    
    results = connection.execute(query).fetchall()
    
    
    # ------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------
    
    print("\nSession-level tool latency:\n")
    
    for row in results:
    
        (
            provider,
            sessions,
            avg_tool_calls,
            median_tool_calls,
            avg_total_wait,
            median_total_wait,
            p90_total_wait,
            p99_total_wait,
            avg_session_median,
            median_session_max,
            p90_session_max,
            p99_session_max,
            pct_over_5,
            pct_over_10,
            pct_over_30,
            pct_over_60,
        ) = row
    
        print("=" * 75)
    
        print(f"Provider: {provider}")
        print(f"Sessions: {sessions:,}")
    
        print("\nTool calls per session:")
        print(f"  Mean:   {avg_tool_calls}")
        print(f"  Median: {median_tool_calls}")
    
        print("\nTotal tool latency per session:")
        print(f"  Mean:   {avg_total_wait} seconds")
        print(f"  Median: {median_total_wait} seconds")
        print(f"  P90:    {p90_total_wait} seconds")
        print(f"  P99:    {p99_total_wait} seconds")
    
        print("\nSession maximum tool latency:")
        print(f"  Median: {median_session_max} seconds")
        print(f"  P90:    {p90_session_max} seconds")
        print(f"  P99:    {p99_session_max} seconds")
    
        print("\nPercentage of sessions containing at least one long call:")
        print(f"  > 5 seconds:  {pct_over_5}%")
        print(f"  > 10 seconds: {pct_over_10}%")
        print(f"  > 30 seconds: {pct_over_30}%")
        print(f"  > 60 seconds: {pct_over_60}%")
    
    
    connection.close()


def tool_thresholds():
    import duckdb
    
    DB = "trace/syfi_coding_trace.duckdb"
    
    TOOLS = [
        "Bash",
        "exec_command",
        "write_stdin",
        "exec",
        "shell_command",
    ]
    
    con = duckdb.connect(DB)
    
    print("\nAutomated tool latency thresholds:")
    print("=" * 110)
    
    for tool in TOOLS:
        query = """
        SELECT
            COUNT(*) AS calls,
            COUNT(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END)) AS valid_calls,
    
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) / 1000.0, 0.50) AS median,
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) / 1000.0, 0.90) AS p90,
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) / 1000.0, 0.95) AS p95,
            quantile_cont(COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) / 1000.0, 0.99) AS p99,
    
            100.0 * AVG(
                CASE WHEN COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) > 5000
                THEN 1 ELSE 0 END
            ) AS pct_gt_5s,
    
            100.0 * AVG(
                CASE WHEN COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) > 10000
                THEN 1 ELSE 0 END
            ) AS pct_gt_10s,
    
            100.0 * AVG(
                CASE WHEN COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) > 30000
                THEN 1 ELSE 0 END
            ) AS pct_gt_30s,
    
            100.0 * AVG(
                CASE WHEN COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) > 60000
                THEN 1 ELSE 0 END
            ) AS pct_gt_60s
    
        FROM tool_calls
        WHERE tool_name = ?
          AND COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) IS NOT NULL
          AND COALESCE(CASE WHEN tool_internal_latency_ms >= 0 THEN tool_internal_latency_ms END, CASE WHEN tool_wall_latency_ms >= 0 THEN tool_wall_latency_ms END) >= 0
        """
    
        row = con.execute(query, [tool]).fetchone()
    
        print(f"\n{tool}")
        print(f"  Calls:       {row[0]:,}")
        print(f"  Valid:       {row[1]:,}")
        print(f"  Median:      {row[2]:.3f} s")
        print(f"  P90:         {row[3]:.3f} s")
        print(f"  P95:         {row[4]:.3f} s")
        print(f"  P99:         {row[5]:.3f} s")
        print(f"  >5s:         {row[6]:.2f}%")
        print(f"  >10s:        {row[7]:.2f}%")
        print(f"  >30s:        {row[8]:.2f}%")
        print(f"  >60s:        {row[9]:.2f}%")
    
    con.close()


def prediction_fields():
    
    import duckdb
    
    DB = "trace/syfi_coding_trace.duckdb"
    
    TOOLS = [
        "Bash",
        "exec_command",
        "write_stdin",
        "exec",
        "shell_command",
    ]
    
    con = duckdb.connect(DB, read_only=True)
    
    # Inspect fields from the actual TraceLab schema.
    query = """
    SELECT
        r.provider,
        r.model,
        r.session_id,
        r.round_index,
    
        r.input_tokens_total,
        r.output_tokens,
    
        tc.tool_name,
        tc.tool_call_id,
        tc.emitted_at,
        tc.input_chars,
    
        -- Post-result fields: analysis only.
        tc.result_at,
        tc.result_chars,
        tc.is_error,
        tc.tool_wall_latency_ms,
        tc.tool_internal_latency_ms
    
    FROM tool_calls AS tc
    JOIN rounds AS r USING (round_pk)
    
    WHERE tc.tool_name IN (
        'Bash',
        'exec_command',
        'write_stdin',
        'exec',
        'shell_command'
    )
    
    ORDER BY tc.emitted_at NULLS LAST
    LIMIT 20
    """
    
    df = con.execute(query).df()
    
    print("\n=== Sample prediction fields ===")
    print(df.to_string(index=False))
    
    # Examine availability across all selected tool records.
    coverage_query = """
    SELECT
        r.provider,
        tc.tool_name,
        COUNT(*) AS total_calls,
    
        COUNT(tc.emitted_at) AS has_emitted_at,
        COUNT(tc.input_chars) AS has_input_chars,
        COUNT(r.input_tokens_total) AS has_context_tokens,
        COUNT(tc.result_at) AS has_result_at,
        COUNT(tc.tool_wall_latency_ms) AS has_wall_latency
    
    FROM tool_calls AS tc
    JOIN rounds AS r USING (round_pk)
    
    WHERE tc.tool_name IN (
        'Bash',
        'exec_command',
        'write_stdin',
        'exec',
        'shell_command'
    )
    
    GROUP BY r.provider, tc.tool_name
    ORDER BY total_calls DESC
    """
    
    coverage = con.execute(coverage_query).df()
    
    print("\n=== Feature availability ===")
    print(coverage.to_string(index=False))
    
    con.close()


STEPS = {
    "tool_latency": tool_latency,
    "session_latency": session_latency,
    "tool_thresholds": tool_thresholds,
    "prediction_fields": prediction_fields,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path.cwd(),
                        help="TraceLab directory containing trace/syfi_coding_trace.duckdb")
    parser.add_argument("--step", choices=["all", *STEPS], default="all")
    args = parser.parse_args()
    root = args.trace_root.expanduser().resolve()
    if not (root / "trace/syfi_coding_trace.duckdb").is_file():
        parser.error("No TraceLab DuckDB at --trace-root/trace/syfi_coding_trace.duckdb")
    os.chdir(root)
    for name, func in STEPS.items():
        if args.step in ("all", name):
            print(f"\\n=== Trace statistics: {name} ===", flush=True)
            func()


if __name__ == "__main__":
    main()
