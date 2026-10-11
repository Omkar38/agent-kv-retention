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
        COUNT(tool_wall_latency_ms) AS valid_calls,

        quantile_cont(tool_wall_latency_ms / 1000.0, 0.50) AS median,
        quantile_cont(tool_wall_latency_ms / 1000.0, 0.90) AS p90,
        quantile_cont(tool_wall_latency_ms / 1000.0, 0.95) AS p95,
        quantile_cont(tool_wall_latency_ms / 1000.0, 0.99) AS p99,

        100.0 * AVG(
            CASE WHEN tool_wall_latency_ms > 5000
            THEN 1 ELSE 0 END
        ) AS pct_gt_5s,

        100.0 * AVG(
            CASE WHEN tool_wall_latency_ms > 10000
            THEN 1 ELSE 0 END
        ) AS pct_gt_10s,

        100.0 * AVG(
            CASE WHEN tool_wall_latency_ms > 30000
            THEN 1 ELSE 0 END
        ) AS pct_gt_30s,

        100.0 * AVG(
            CASE WHEN tool_wall_latency_ms > 60000
            THEN 1 ELSE 0 END
        ) AS pct_gt_60s

    FROM tool_calls
    WHERE tool_name = ?
      AND tool_wall_latency_ms IS NOT NULL
      AND tool_wall_latency_ms >= 0
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