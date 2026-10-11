
import duckdb

con = duckdb.connect()

SOURCE = "group_work/tool_intervals.parquet"
OUTPUT = "group_work/tool_activity_episodes.parquet"

query = f"""
COPY (
    WITH valid_calls AS (
        SELECT *
        FROM read_parquet('{SOURCE}')
        WHERE timing_status = 'valid'
    ),

    ordered AS (
        SELECT *,
            MAX(tool_end_ts) OVER (
                PARTITION BY provider, session_id
                ORDER BY
                    tool_start_ts,
                    tool_end_ts,
                    round_pk,
                    tool_index
                ROWS BETWEEN
                    UNBOUNDED PRECEDING
                    AND 1 PRECEDING
            ) AS previous_max_end
        FROM valid_calls
    ),

    marked AS (
        SELECT *,
            CASE
                WHEN previous_max_end IS NULL
                  OR tool_start_ts >= previous_max_end
                    THEN 1
                ELSE 0
            END AS new_episode
        FROM ordered
    ),

    numbered AS (
        SELECT *,
            SUM(new_episode) OVER (
                PARTITION BY provider, session_id
                ORDER BY
                    tool_start_ts,
                    tool_end_ts,
                    round_pk,
                    tool_index
                ROWS BETWEEN
                    UNBOUNDED PRECEDING
                    AND CURRENT ROW
            ) AS episode_id
        FROM marked
    ),

    episodes AS (
        SELECT
            provider,
            session_id,
            episode_id,

            MIN(tool_start_ts) AS start_ts,
            MAX(tool_end_ts) AS end_ts,

            COUNT(*) AS n_calls,
            COUNT(DISTINCT round_pk) AS n_rounds,

            STRING_AGG(
                DISTINCT tool_name,
                ',' ORDER BY tool_name
            ) AS tool_names

        FROM numbered
        GROUP BY provider, session_id, episode_id
    )

    SELECT *,
        EPOCH(end_ts - start_ts)
            AS tool_activity_duration_s,

        n_calls > 1 AS multiple_calls

    FROM episodes
)
TO '{OUTPUT}' (FORMAT PARQUET)
"""

con.execute(query)

print(f"Saved: {OUTPUT}")

summary = con.execute(f"""
SELECT
    provider,
    COUNT(*) AS episodes,
    COUNT(*) FILTER (
        WHERE multiple_calls
    ) AS multi_call_episodes,
    SUM(n_calls) AS total_calls,
    ROUND(
        MEDIAN(tool_activity_duration_s), 3
    ) AS median_s,
    ROUND(
        QUANTILE_CONT(
            tool_activity_duration_s, 0.99
        ), 3
    ) AS p99_s
FROM read_parquet('{OUTPUT}')
GROUP BY provider
ORDER BY provider
""").df()

print("\n=== Tool activity episodes ===")
print(summary.to_string(index=False))

con.close()
