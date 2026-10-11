
import duckdb

con = duckdb.connect(
    "trace/syfi_coding_trace.duckdb",
    read_only=True
)

query = """
WITH round_times AS (
    SELECT
        r.round_pk,
        r.session_id,
        r.round_index,

        MIN(e.timestamp) AS first_event_at,

        MAX(e.timestamp) FILTER (
            WHERE e.event_type IN (
                'reasoning', 'text',
                'tool_call', 'usage_report'
            )
        ) AS decision_ts

    FROM rounds r
    JOIN timing_events e USING (round_pk)

    GROUP BY
        r.round_pk, r.session_id, r.round_index
),

paired AS (
    SELECT *,
        LEAD(round_pk) OVER (
            PARTITION BY session_id
            ORDER BY round_index, first_event_at, round_pk
        ) AS next_round_pk

    FROM round_times
),

valid_pauses AS (
    SELECT p.*
    FROM paired p

    JOIN read_parquet('group_work/pauses.parquet') q
      ON q.session_id = p.session_id
     AND q.round_index = p.round_index

    WHERE q.usable IS TRUE
      AND q.cause IN ('tool', 'tool+user')
),

matched_calls AS (
    SELECT
        p.session_id,
        p.round_index,
        p.decision_ts,

        tc.tool_name,
        tc.tool_call_id,
        tc.emitted_at

    FROM valid_pauses p

    JOIN timing_events e
      ON e.round_pk = p.next_round_pk
     AND e.event_type = 'tool_result'

    JOIN tool_calls tc
      ON tc.tool_call_id = e.tool_call_id

    JOIN rounds source_round
      ON source_round.round_pk = tc.round_pk
     AND source_round.session_id = p.session_id
)

SELECT
    COUNT(*) AS matched_call_records,

    COUNT(*) FILTER (
        WHERE emitted_at <= decision_ts
    ) AS emitted_before_decision,

    COUNT(*) FILTER (
        WHERE emitted_at > decision_ts
    ) AS emitted_after_decision,

    COUNT(*) FILTER (
        WHERE emitted_at IS NULL
           OR decision_ts IS NULL
    ) AS cannot_verify

FROM matched_calls
"""

result = con.execute(query).df()

print("\n=== Decision-time feature validation ===")
print(result.to_string(index=False))

con.close()
