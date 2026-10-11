
import duckdb

DB = "trace/syfi_coding_trace.duckdb"
EPISODES = "group_work/tool_activity_episodes.parquet"

con = duckdb.connect(DB, read_only=True)

query = """
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
episode_flags AS (
    SELECT
        ep.provider,
        ep.session_id,
        ep.episode_id,

        COUNT(s.timestamp) FILTER (
            WHERE s.event_type IN ('reasoning', 'text')
        ) AS generation_events_inside,

        COUNT(s.timestamp) FILTER (
            WHERE s.event_type = 'user_message'
        ) AS user_messages_inside

    FROM read_parquet(?) ep
    LEFT JOIN signals s
      ON s.provider = ep.provider
     AND s.session_id = ep.session_id
     AND s.timestamp > ep.start_ts
     AND s.timestamp < ep.end_ts

    GROUP BY
        ep.provider,
        ep.session_id,
        ep.episode_id
)
SELECT
    provider,
    COUNT(*) AS total_episodes,

    COUNT(*) FILTER (
        WHERE generation_events_inside > 0
    ) AS episodes_with_generation,

    COUNT(*) FILTER (
        WHERE user_messages_inside > 0
    ) AS episodes_with_user_input,

    COUNT(*) FILTER (
        WHERE generation_events_inside = 0
          AND user_messages_inside = 0
    ) AS episodes_without_these_signals

FROM episode_flags
GROUP BY provider
ORDER BY provider
"""

result = con.execute(query, [EPISODES]).df()

print("\n=== Episode generation audit ===")
print(result.to_string(index=False))

con.close()
