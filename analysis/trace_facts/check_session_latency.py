import duckdb


# Connect to the TraceLab DuckDB database.
connection = duckdb.connect(
    "trace/syfi_coding_trace.duckdb",
    read_only=True,
)


# ------------------------------------------------------------
# SESSION-LEVEL TOOL LATENCY
#
# For now, we use tool_wall_latency_ms:
#
#     result_at - emitted_at
#
# This measures the observed wall-clock latency of individual
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
        tc.tool_wall_latency_ms / 1000.0 AS latency_seconds
    FROM tool_calls AS tc
    JOIN rounds AS r
        USING (round_pk)

    WHERE tc.tool_wall_latency_ms IS NOT NULL
      AND tc.tool_wall_latency_ms >= 0
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