
import duckdb

DB = "trace/syfi_coding_trace.duckdb"
CANDIDATES = "group_work/tool_episode_candidates.parquet"

con = duckdb.connect(DB, read_only=True)

query = """
WITH round_outputs AS (
    SELECT
        round_pk,

        MAX(timestamp) FILTER (
            WHERE event_type IN ('reasoning', 'text')
        ) AS last_generation_event,

        MAX(timestamp) FILTER (
            WHERE event_type = 'tool_call'
        ) AS last_tool_call_event

    FROM timing_events
    GROUP BY round_pk
)

SELECT
    c.provider,

    COUNT(*) AS candidates,

    COUNT(*) FILTER (
        WHERE r.last_generation_event >
              c.candidate_start_ts
    ) AS generation_after_first_call,

    COUNT(*) FILTER (
        WHERE r.last_tool_call_event >
              c.candidate_start_ts
    ) AS later_tool_calls_same_round,

    COUNT(*) FILTER (
        WHERE c.first_tool_input_chars IS NULL
    ) AS missing_input_chars,

    COUNT(*) FILTER (
        WHERE c.context_tokens <= 0
    ) AS nonpositive_context

FROM read_parquet(?) c
LEFT JOIN round_outputs r
    ON c.first_call_round_pk = r.round_pk

GROUP BY c.provider
ORDER BY c.provider
"""

result = con.execute(query, [CANDIDATES]).df()

print("\n=== Candidate boundary validation ===")
print(result.to_string(index=False))

con.close()
