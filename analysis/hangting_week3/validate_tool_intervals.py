
import duckdb

con = duckdb.connect()

DATA = "group_work/tool_intervals.parquet"

print("\n=== 1. Timestamp consistency ===")

result = con.execute("""
SELECT
    provider,
    COUNT(*) AS compared_calls,
    ROUND(
        AVG(ABS(
            tool_duration_s -
            reported_tool_duration_s
        )), 6
    ) AS mean_difference_s,
    ROUND(
        QUANTILE_CONT(ABS(
            tool_duration_s -
            reported_tool_duration_s
        ), 0.99), 6
    ) AS p99_difference_s,
    COUNT(*) FILTER (
        WHERE ABS(
            tool_duration_s -
            reported_tool_duration_s
        ) > 0.1
    ) AS differences_over_100ms
FROM read_parquet(?)
WHERE timing_status = 'valid'
  AND reported_tool_duration_s IS NOT NULL
GROUP BY provider
ORDER BY provider
""", [DATA]).df()

print(result.to_string(index=False))

print("\n=== 2. Overlapping tool calls ===")

overlap = con.execute("""
WITH ordered AS (
    SELECT
        provider,
        session_id,
        tool_start_ts,
        tool_end_ts,

        MAX(tool_end_ts) OVER (
            PARTITION BY provider, session_id
            ORDER BY
                tool_start_ts,
                tool_end_ts,
                round_pk,
                tool_index
            ROWS BETWEEN
                UNBOUNDED PRECEDING AND
                1 PRECEDING
        ) AS previous_max_end

    FROM read_parquet(?)
    WHERE timing_status = 'valid'
)
SELECT
    provider,
    COUNT(*) AS valid_calls,
    COUNT(*) FILTER (
        WHERE tool_start_ts < previous_max_end
    ) AS potentially_overlapping_calls
FROM ordered
GROUP BY provider
ORDER BY provider
""", [DATA]).df()

print(overlap.to_string(index=False))

con.close()
