import duckdb

connection = duckdb.connect(
    "trace/syfi_coding_trace.duckdb",
    read_only=True,
)

query = """
SELECT
    tool_name,
    COUNT(*) AS calls,
    ROUND(AVG(tool_wall_latency_ms) / 1000.0, 3) AS mean_seconds,
    ROUND(MEDIAN(tool_wall_latency_ms) / 1000.0, 3) AS median_seconds,

    ROUND(
        quantile_cont(tool_wall_latency_ms, 0.50) / 1000.0,
        3
    ) AS p50_seconds,

    ROUND(
        quantile_cont(tool_wall_latency_ms, 0.90) / 1000.0,
        3
    ) AS p90_seconds,

    ROUND(
        quantile_cont(tool_wall_latency_ms, 0.95) / 1000.0,
        3
    ) AS p95_seconds,

    ROUND(
        quantile_cont(tool_wall_latency_ms, 0.99) / 1000.0,
        3
    ) AS p99_seconds,

    ROUND(
        quantile_cont(tool_wall_latency_ms, 0.999) / 1000.0,
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

AND tool_wall_latency_ms IS NOT NULL
AND tool_wall_latency_ms >= 0

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
    print(
        f"{str(row[0]):20} "
        f"{row[1]:10,} "
        f"{row[2]:10} "
        f"{row[3]:10} "
        f"{row[4]:10} "
        f"{row[5]:10} "
        f"{row[6]:10} "
        f"{row[7]:10}"
    )

connection.close()