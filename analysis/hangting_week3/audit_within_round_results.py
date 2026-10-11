
import duckdb

con = duckdb.connect(
    "trace/syfi_coding_trace.duckdb",
    read_only=True
)

query = """
WITH generation_start AS (
    SELECT
        round_pk,
        MIN(timestamp) FILTER (
            WHERE event_type IN (
                'reasoning',
                'text',
                'tool_call'
            )
        ) AS first_generation_at
    FROM timing_events
    GROUP BY round_pk
)
SELECT
    r.provider,
    COUNT(*) AS total_tool_results,

    COUNT(*) FILTER (
        WHERE e.timestamp > g.first_generation_at
    ) AS results_after_generation_started,

    COUNT(*) FILTER (
        WHERE e.timestamp <= g.first_generation_at
    ) AS results_before_or_at_generation,

    COUNT(*) FILTER (
        WHERE g.first_generation_at IS NULL
           OR e.timestamp IS NULL
    ) AS timing_unknown

FROM timing_events e
JOIN rounds r USING (round_pk)
JOIN generation_start g USING (round_pk)

WHERE e.event_type = 'tool_result'
GROUP BY r.provider
ORDER BY r.provider
"""

result = con.execute(query).df()

print("\n=== Within-round tool-result timing ===")
print(result.to_string(index=False))

con.close()
