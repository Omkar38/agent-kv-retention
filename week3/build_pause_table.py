"""Build the pause table (v2): one row per pause, the gap after each LLM round.

A pause starts when the model finishes generating in round k. It ends when the
next LLM call can start: when the last result of the tools round k called has
come back, or, for a human wait, when the person's message arrives. During
that stretch the session's KV cache sits idle in GPU memory.

What changed from v1 (see README):
  1. Tools belong to the round that called them, not the round where their
     results happen to be logged. Newer Codex logs sometimes log a round's own
     tool results inside that same round; v1 then credited those tools to the
     previous pause and counted the model's thinking time as idle time.
  2. In those newer Codex logs, usage_report is written after the tools
     finish, so it marks the end of generation only when it comes before the
     round's first tool result.
  3. Older Codex logs write each tool call together with its result when the
     tool finishes. Those sessions keep the v1 rule, which already matches
     the tools' own runtimes there.
"""
import duckdb

DB = "trace/syfi_coding_trace.duckdb"
OUT = "our_work/pauses.parquet"

con = duckdb.connect(DB, read_only=True)

pauses = con.sql("""
WITH old_codex_sessions AS (
    -- Older Codex logs: a tool call and its result share a timestamp
    -- (logged together when the tool finishes). Detected per session.
    SELECT r.session_id
    FROM tool_calls t JOIN rounds r USING (round_pk)
    WHERE r.provider = 'codex'
    GROUP BY 1
    HAVING avg((t.tool_wall_latency_ms <= 5)::INT) >= 0.5
),
per_round AS (
    -- One row per LLM round, with its key timestamps.
    SELECT r.round_pk, r.provider, r.user, r.session_id, r.round_index, r.model,
           r.input_tokens_total + r.output_tokens AS context_tokens,
           min(e.timestamp) AS first_event_at,
           min(e.timestamp) FILTER (WHERE e.event_type IN
               ('reasoning', 'text', 'tool_call')) AS first_gen_at,
           max(e.timestamp) FILTER (WHERE e.event_type IN
               ('reasoning', 'text', 'tool_call')) AS last_gen_at,
           max(e.timestamp) FILTER (WHERE e.event_type = 'usage_report') AS usage_at,
           max(e.timestamp) FILTER (WHERE e.event_type IN
               ('user_message', 'tool_result')) AS last_input_at,
           count(*) FILTER (WHERE e.event_type = 'user_message') AS n_user_msgs,
           count(*) FILTER (WHERE e.event_type = 'tool_result') AS n_tool_results
    FROM rounds r JOIN timing_events e USING (round_pk)
    GROUP BY ALL
),
paired AS (
    -- Line each round up with the round that follows it in the same session.
    SELECT p.*,
           o.session_id IS NOT NULL AS old_codex_log,
           count(*) OVER (PARTITION BY p.session_id, p.round_index) > 1 AS dup_index,
           lead(p.round_pk)       OVER w AS next_round_pk,
           lead(p.first_gen_at)   OVER w AS next_first_gen_at,
           lead(p.last_input_at)  OVER w AS next_last_input_at,
           lead(p.n_user_msgs)    OVER w AS next_user_msgs,
           lead(p.n_tool_results) OVER w AS next_tool_results
    FROM per_round p LEFT JOIN old_codex_sessions o USING (session_id)
    WINDOW w AS (PARTITION BY p.session_id ORDER BY p.round_index, p.first_event_at, p.round_pk)
),
result_events AS (
    -- When each tool's result was logged. Used as a fallback, because
    -- tool_calls.result_at is blank for a few hundred Claude calls.
    SELECT r.session_id, e.tool_call_id, max(e.timestamp) AS result_ts
    FROM timing_events e JOIN rounds r USING (round_pk)
    WHERE e.event_type = 'tool_result'
    GROUP BY ALL
),
one_call AS (
    -- Each tool call once (a few hundred call ids are duplicated).
    SELECT t.round_pk, t.tool_call_id,
           any_value(t.tool_name) AS tool_name,
           any_value(t.input_chars) AS input_chars,
           any_value(coalesce(t.result_at, re.result_ts)) AS result_at
    FROM tool_calls t
    JOIN rounds r USING (round_pk)
    LEFT JOIN result_events re ON re.session_id = r.session_id AND re.tool_call_id = t.tool_call_id
    GROUP BY 1, 2
),
called AS (
    -- The tools the model called in each round. Known the moment it stops.
    SELECT round_pk,
           count(*) AS n_tools,
           count(result_at) AS n_results,
           string_agg(DISTINCT tool_name, ',' ORDER BY tool_name) AS tool_names,
           sum(input_chars) AS tool_input_chars,
           min(result_at) AS first_result_at,
           max(result_at) AS last_result_at
    FROM one_call
    GROUP BY 1
),
next_user AS (
    -- User messages the next LLM call read: logged in round k+1
    -- before that round started generating.
    SELECT p.round_pk, max(e.timestamp) AS user_msg_at
    FROM paired p
    JOIN timing_events e ON e.round_pk = p.next_round_pk AND e.event_type = 'user_message'
    WHERE p.next_first_gen_at IS NULL OR e.timestamp <= p.next_first_gen_at
    GROUP BY 1
),
tools_by_session AS (
    SELECT r.session_id, t.tool_call_id,
           any_value(t.tool_name) AS tool_name,
           any_value(t.input_chars) AS input_chars
    FROM tool_calls t JOIN rounds r USING (round_pk)
    GROUP BY ALL
),
result_tools AS (
    -- v1 attribution, used only for older Codex logs: the tools whose
    -- results end the pause (their calls are logged late, with the result).
    SELECT e.round_pk AS next_round_pk,
           count(*) AS n_tools,
           string_agg(DISTINCT t.tool_name, ',' ORDER BY t.tool_name) AS tool_names,
           sum(t.input_chars) AS tool_input_chars
    FROM timing_events e
    JOIN rounds r USING (round_pk)
    JOIN tools_by_session t ON t.session_id = r.session_id AND t.tool_call_id = e.tool_call_id
    WHERE e.event_type = 'tool_result'
    GROUP BY 1
),
measured AS (
    SELECT p.*,
           nu.user_msg_at,
           CASE WHEN p.old_codex_log THEN coalesce(rt.n_tools, 0)
                ELSE coalesce(c.n_tools, 0) END AS n_tools,
           CASE WHEN p.old_codex_log THEN rt.tool_names ELSE c.tool_names END AS tool_names,
           CASE WHEN p.old_codex_log THEN rt.tool_input_chars ELSE c.tool_input_chars END AS tool_input_chars,
           -- Pause start: the end of generation in round k.
           CASE WHEN p.old_codex_log THEN greatest(p.last_gen_at, p.usage_at)
                ELSE greatest(p.last_gen_at,
                              CASE WHEN p.usage_at < coalesce(c.first_result_at, 'infinity'::TIMESTAMP)
                                   THEN p.usage_at END)
           END AS pause_start_at,
           -- Pause end: when the next LLM call has everything it needs.
           CASE WHEN p.next_round_pk IS NULL THEN NULL
                WHEN p.old_codex_log THEN p.next_last_input_at
                ELSE greatest(CASE WHEN c.n_results = c.n_tools THEN c.last_result_at END,
                              nu.user_msg_at)
           END AS pause_end_at,
           -- Cause: what the next LLM call was waiting for.
           CASE WHEN p.next_round_pk IS NULL THEN 'end'
                WHEN p.old_codex_log THEN
                     CASE WHEN p.next_user_msgs = 0 AND p.next_tool_results = 0 THEN 'unknown'
                          WHEN p.next_user_msgs > 0 AND p.next_tool_results > 0 THEN 'tool+user'
                          WHEN p.next_user_msgs > 0 THEN 'user'
                          ELSE 'tool' END
                WHEN coalesce(c.n_tools, 0) > 0 AND nu.user_msg_at IS NOT NULL THEN 'tool+user'
                WHEN nu.user_msg_at IS NOT NULL THEN 'user'
                WHEN coalesce(c.n_tools, 0) > 0 THEN 'tool'
                ELSE 'unknown'
           END AS cause,
           lead(p.dup_index) OVER (PARTITION BY p.session_id
               ORDER BY p.round_index, p.first_event_at, p.round_pk) AS next_dup_index
    FROM paired p
    LEFT JOIN called c USING (round_pk)
    LEFT JOIN next_user nu USING (round_pk)
    LEFT JOIN result_tools rt USING (next_round_pk)
)
SELECT provider, user, session_id, round_index, model,
       -- the target: how long the cache sat idle, and why
       epoch(pause_end_at - pause_start_at) AS pause_s,
       cause,
       -- false for rows we can't trust (see README)
       (cause NOT IN ('end', 'unknown')
        AND pause_end_at IS NOT NULL AND pause_start_at IS NOT NULL
        AND pause_end_at >= pause_start_at
        AND NOT dup_index AND NOT coalesce(next_dup_index, false)) AS usable,
       -- size of the idle cache (drives the eviction penalty B)
       context_tokens,
       -- decision-time features: the tools the model just called
       n_tools,
       tool_names,
       tool_input_chars,
       -- when the pause starts and ends, for replaying sessions in order
       pause_start_at,
       pause_end_at,
       CASE WHEN provider = 'claude' THEN 'claude'
            WHEN old_codex_log THEN 'codex_old'
            ELSE 'codex' END AS log_format
FROM measured
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
