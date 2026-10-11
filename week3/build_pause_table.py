"""Build the pause table: one row per pause (the gap after each LLM round).

A pause runs from the last thing the model produced in round k to the last
input that arrives for round k+1. During that stretch the session's KV cache
sits idle in GPU memory, and our policy has to decide whether to keep it.
"""
import duckdb

DB = "trace/syfi_coding_trace.duckdb"
OUT = "our_work/pauses.parquet"

con = duckdb.connect(DB, read_only=True)

pauses = con.sql("""
WITH per_round AS (
    -- One row per LLM round, with its key timestamps.
    SELECT r.round_pk, r.provider, r.user, r.session_id, r.round_index, r.model,
           r.input_tokens_total + r.output_tokens AS context_tokens,
           min(e.timestamp) AS first_event_at,
           max(e.timestamp) FILTER (WHERE e.event_type IN
               ('reasoning', 'text', 'tool_call', 'usage_report')) AS last_output_at,
           max(e.timestamp) FILTER (WHERE e.event_type IN
               ('user_message', 'tool_result')) AS last_input_at,
           count(*) FILTER (WHERE e.event_type = 'user_message') AS n_user_msgs,
           count(*) FILTER (WHERE e.event_type = 'tool_result') AS n_tool_results
    FROM rounds r JOIN timing_events e USING (round_pk)
    GROUP BY ALL
),
paired AS (
    -- Line each round up with the round that follows it in the same session.
    SELECT *,
           count(*) OVER (PARTITION BY session_id, round_index) > 1 AS dup_index,
           lead(round_pk)       OVER w AS next_round_pk,
           lead(last_input_at)  OVER w AS next_last_input_at,
           lead(n_user_msgs)    OVER w AS next_user_msgs,
           lead(n_tool_results) OVER w AS next_tool_results
    FROM per_round
    WINDOW w AS (PARTITION BY session_id ORDER BY round_index, first_event_at, round_pk)
),
tools AS (
    -- Every tool call, tagged with its session (ids are unique per session).
    SELECT r.session_id, t.tool_call_id,
           any_value(t.tool_name) AS tool_name,
           any_value(t.input_chars) AS input_chars
    FROM tool_calls t JOIN rounds r USING (round_pk)
    GROUP BY ALL
),
pause_tools AS (
    -- The tools running during the pause = the tools whose results end it.
    -- Matched by the result's id, because older Codex logs file the call
    -- in a different round than Claude does.
    SELECT e.round_pk AS next_round_pk,
           count(*) AS n_tools,
           string_agg(DISTINCT t.tool_name, ',' ORDER BY t.tool_name) AS tool_names,
           sum(t.input_chars) AS tool_input_chars
    FROM timing_events e
    JOIN rounds r USING (round_pk)
    JOIN tools t ON t.session_id = r.session_id AND t.tool_call_id = e.tool_call_id
    WHERE e.event_type = 'tool_result'
    GROUP BY 1
),
labeled AS (
    SELECT p.*, pt.n_tools, pt.tool_names, pt.tool_input_chars,
           epoch(p.next_last_input_at - p.last_output_at) AS pause_s,
           CASE WHEN p.next_round_pk IS NULL THEN 'end'
                WHEN p.next_user_msgs = 0 AND p.next_tool_results = 0 THEN 'unknown'
                WHEN p.next_user_msgs > 0 AND p.next_tool_results > 0 THEN 'tool+user'
                WHEN p.next_user_msgs > 0 THEN 'user'
                ELSE 'tool' END AS cause,
           lead(dup_index) OVER (PARTITION BY session_id
               ORDER BY round_index, first_event_at, round_pk) AS next_dup_index
    FROM paired p
    LEFT JOIN pause_tools pt USING (next_round_pk)
)
SELECT provider, user, session_id, round_index, model,
       -- the target: how long the cache sat idle, and why
       pause_s,
       cause,
       -- false for rows we can't trust
       (cause NOT IN ('end', 'unknown') AND pause_s >= 0
        AND NOT dup_index AND NOT coalesce(next_dup_index, false)) AS usable,
       -- size of the idle cache (drives the eviction penalty B)
       context_tokens,
       -- decision-time features: all known the moment the pause starts
       coalesce(n_tools, 0) AS n_tools,
       tool_names,
       tool_input_chars
FROM labeled
ORDER BY session_id, round_index
""")

pauses.write_parquet(OUT)
print(f"Wrote {OUT}\n")
con.sql("""
    SELECT provider, cause, count(*) AS pauses,
           count(*) FILTER (WHERE usable) AS usable,
           round(quantile_cont(pause_s, 0.5) FILTER (WHERE usable), 2) AS median_s,
           round(quantile_cont(pause_s, 0.9) FILTER (WHERE usable), 1) AS p90_s,
           round(quantile_cont(pause_s, 0.99) FILTER (WHERE usable), 0) AS p99_s
    FROM read_parquet(?)
    GROUP BY ALL ORDER BY provider, pauses DESC
""", params=[OUT]).show()