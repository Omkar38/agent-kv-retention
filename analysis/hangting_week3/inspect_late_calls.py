
import duckdb

con = duckdb.connect(
    "trace/syfi_coding_trace.duckdb",
    read_only=True
)

con.execute("""
CREATE TEMP TABLE matched AS
WITH per_round AS (
    SELECT
        r.round_pk,
        r.provider,
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
        r.round_pk, r.provider,
        r.session_id, r.round_index
),
paired AS (
    SELECT *,
        LEAD(round_pk) OVER (
            PARTITION BY session_id
            ORDER BY round_index,
                     first_event_at,
                     round_pk
        ) AS next_round_pk
    FROM per_round
),
valid AS (
    SELECT p.*
    FROM paired p
    JOIN read_parquet(
        'group_work/pauses.parquet'
    ) q
      ON q.session_id = p.session_id
     AND q.round_index = p.round_index
    WHERE q.usable IS TRUE
      AND q.cause IN ('tool', 'tool+user')
)
SELECT
    p.provider,
    p.session_id,
    p.round_index,
    p.decision_ts,
    tc.tool_name,
    tc.tool_call_id,
    tc.emitted_at,
    source.round_index AS call_round_index,
    CASE
        WHEN tc.round_pk = p.round_pk
            THEN 'current_round'
        WHEN tc.round_pk = p.next_round_pk
            THEN 'next_round'
        ELSE 'other_round'
    END AS call_origin
FROM valid p
JOIN timing_events e
  ON e.round_pk = p.next_round_pk
 AND e.event_type = 'tool_result'
JOIN tool_calls tc
  ON tc.tool_call_id = e.tool_call_id
JOIN rounds source
  ON source.round_pk = tc.round_pk
 AND source.session_id = p.session_id
""")

print("\n=== Late calls by provider and origin ===")

print(con.execute("""
SELECT
    provider,
    call_origin,
    COUNT(*) AS matched,
    COUNT(*) FILTER (
        WHERE emitted_at > decision_ts
    ) AS emitted_after,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE emitted_at > decision_ts
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS pct_after
FROM matched
GROUP BY provider, call_origin
ORDER BY provider, call_origin
""").df().to_string(index=False))

print("\n=== Examples of late calls ===")

print(con.execute("""
SELECT
    provider,
    session_id,
    round_index,
    call_round_index,
    call_origin,
    tool_name,
    decision_ts,
    emitted_at,
    ROUND(
        EPOCH(emitted_at - decision_ts), 3
    ) AS seconds_after
FROM matched
WHERE emitted_at > decision_ts
ORDER BY seconds_after DESC
LIMIT 10
""").df().to_string(index=False))

con.close()
