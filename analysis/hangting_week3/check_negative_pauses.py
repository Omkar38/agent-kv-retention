
import duckdb

con = duckdb.connect(
    "trace/syfi_coding_trace.duckdb",
    read_only=True
)

session = (
    "claude:b2145a8e-6787-4940-9a56-"
    "ccd7a3f46934/subagents/agent-a4a42d80d5baeb377"
)

query = """
WITH per_round AS (
    SELECT
        r.round_pk,
        r.session_id,
        r.round_index,
        MIN(e.timestamp) AS first_event_at,
        MAX(e.timestamp) FILTER (
            WHERE e.event_type IN
            ('reasoning', 'text', 'tool_call', 'usage_report')
        ) AS last_output_at,
        MAX(e.timestamp) FILTER (
            WHERE e.event_type IN ('user_message', 'tool_result')
        ) AS last_input_at
    FROM rounds r
    JOIN timing_events e USING (round_pk)
    WHERE r.session_id = ?
    GROUP BY ALL
),
paired AS (
    SELECT *,
        LEAD(round_index) OVER w AS next_round,
        LEAD(first_event_at) OVER w AS next_first_event,
        LEAD(last_input_at) OVER w AS next_input
    FROM per_round
    WINDOW w AS (
        PARTITION BY session_id
        ORDER BY round_index, first_event_at, round_pk
    )
)
SELECT
    round_index,
    last_output_at,
    next_round,
    next_first_event,
    next_input,
    EPOCH(next_input - last_output_at) AS pause_s
FROM paired
WHERE round_index BETWEEN 7 AND 9
ORDER BY round_index
"""

result = con.execute(query, [session]).df()
print(result.to_string(index=False))

con.close()
