
import duckdb

DB = "trace/syfi_coding_trace.duckdb"
INPUT = "group_work/tool_episode_candidates.parquet"
OUTPUT = "group_work/tool_episode_candidates_audited.parquet"

con = duckdb.connect(DB, read_only=True)

query = f"""
COPY (
    WITH signals AS (
        SELECT
            r.provider,
            r.session_id,
            e.event_type,
            e.timestamp
        FROM timing_events e
        JOIN rounds r USING (round_pk)
        WHERE e.event_type IN (
            'reasoning', 'text', 'user_message'
        )
    ),
    flags AS (
        SELECT
            c.provider,
            c.session_id,
            c.episode_id,

            COUNT(*) FILTER (
                WHERE s.event_type IN ('reasoning', 'text')
            ) AS generation_events_inside,

            COUNT(*) FILTER (
                WHERE s.event_type = 'user_message'
            ) AS user_messages_inside

        FROM read_parquet('{INPUT}') c

        LEFT JOIN signals s
          ON s.provider = c.provider
         AND s.session_id = c.session_id
         AND s.timestamp > c.candidate_start_ts
         AND s.timestamp < c.tool_activity_end_ts

        GROUP BY
            c.provider, c.session_id, c.episode_id
    )
    SELECT
        c.*,

        f.generation_events_inside,
        f.user_messages_inside,

        (
            c.observed_n_calls = 1
            AND f.generation_events_inside = 0
            AND f.user_messages_inside = 0
        ) AS simple_tool_wait_proxy

    FROM read_parquet('{INPUT}') c
    JOIN flags f
      USING (provider, session_id, episode_id)
)
TO '{OUTPUT}' (FORMAT PARQUET)
"""

con.execute(query)

summary = con.execute(f"""
SELECT
    provider,
    COUNT(*) AS total_candidates,

    COUNT(*) FILTER (
        WHERE simple_tool_wait_proxy
    ) AS simple_candidates,

    COUNT(*) FILTER (
        WHERE NOT simple_tool_wait_proxy
    ) AS complex_candidates,

    COUNT(*) FILTER (
        WHERE generation_events_inside > 0
    ) AS generation_overlap,

    COUNT(*) FILTER (
        WHERE user_messages_inside > 0
    ) AS user_input_overlap

FROM read_parquet('{OUTPUT}')
GROUP BY provider
ORDER BY provider
""").df()

print("\n=== Audited Week 3 candidates ===")
print(summary.to_string(index=False))

con.close()
