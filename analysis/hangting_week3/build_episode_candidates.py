
import duckdb

DB = "trace/syfi_coding_trace.duckdb"
EPISODES = "group_work/tool_activity_episodes.parquet"
CALLS = "group_work/tool_intervals.parquet"
OUT = "group_work/tool_episode_candidates.parquet"

con = duckdb.connect(DB, read_only=True)

query = f"""
COPY (
    SELECT
        ep.provider,
        ep.session_id,
        ep.episode_id,

        -- Candidate timing, not yet verified KV idle timing
        ep.start_ts AS candidate_start_ts,
        ep.end_ts AS tool_activity_end_ts,
        ep.tool_activity_duration_s,

        -- Information from the first emitted tool call
        tc.tool_name AS first_tool_name,
        tc.input_chars AS first_tool_input_chars,
        tc.tool_call_id AS first_tool_call_id,
        tc.round_pk AS first_call_round_pk,
        tc.round_index AS first_call_round_index,
        tc.model,
        r.input_tokens_total AS context_tokens,

        -- Retrospective fields: NOT prediction features
        ep.n_calls AS observed_n_calls,
        ep.n_rounds AS observed_n_rounds,
        ep.tool_names AS observed_tool_names

    FROM read_parquet('{EPISODES}') ep

    LEFT JOIN read_parquet('{CALLS}') tc
      ON tc.provider = ep.provider
     AND tc.session_id = ep.session_id
     AND tc.tool_start_ts = ep.start_ts
     AND tc.timing_status = 'valid'

    LEFT JOIN rounds r
      ON r.round_pk = tc.round_pk

    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
            ep.provider,
            ep.session_id,
            ep.episode_id
        ORDER BY
            tc.round_pk,
            tc.tool_index,
            tc.tool_call_id
    ) = 1
)
TO '{OUT}' (FORMAT PARQUET)
"""

con.execute(query)

result = con.execute(f"""
SELECT
    provider,
    COUNT(*) AS episodes,
    COUNT(*) FILTER (
        WHERE first_tool_name IS NULL
    ) AS missing_first_tool,
    COUNT(*) FILTER (
        WHERE context_tokens IS NULL
    ) AS missing_context,
    COUNT(*) FILTER (
        WHERE observed_n_calls > 1
    ) AS multi_call_episodes
FROM read_parquet('{OUT}')
GROUP BY provider
ORDER BY provider
""").df()

print("\n=== Candidate episode dataset ===")
print(result.to_string(index=False))

con.close()
