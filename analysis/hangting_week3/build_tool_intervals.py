
import duckdb

DB = "trace/syfi_coding_trace.duckdb"
OUT = "group_work/tool_intervals.parquet"

con = duckdb.connect(DB, read_only=True)

con.execute(f"""
COPY (
    SELECT
        r.provider,
        r.session_id,
        r.round_pk,
        r.round_index,
        r.model,

        tc.tool_index,
        tc.tool_call_id,
        tc.tool_name,
        tc.input_chars,

        tc.emitted_at AS tool_start_ts,
        tc.result_at AS tool_end_ts,

        EPOCH(
            tc.result_at - tc.emitted_at
        ) AS tool_duration_s,

        tc.tool_wall_latency_ms / 1000.0
            AS reported_tool_duration_s,

        CASE
            WHEN tc.emitted_at IS NULL
              OR tc.result_at IS NULL
                THEN 'missing_timestamp'
            WHEN tc.result_at < tc.emitted_at
                THEN 'negative_duration'
            ELSE 'valid'
        END AS timing_status

    FROM tool_calls tc
    JOIN rounds r USING (round_pk)
)
TO '{OUT}' (FORMAT PARQUET)
""")

print(f"Saved: {OUT}")

result = con.execute(f"""
SELECT
    provider,
    timing_status,
    COUNT(*) AS records,
    COUNT(DISTINCT session_id) AS sessions,
    ROUND(MEDIAN(tool_duration_s), 3)
        AS median_duration_s,
    ROUND(
        QUANTILE_CONT(tool_duration_s, 0.99), 3
    ) AS p99_duration_s
FROM read_parquet('{OUT}')
GROUP BY provider, timing_status
ORDER BY provider, timing_status
""").df()

print("\n=== Tool interval summary ===")
print(result.to_string(index=False))

con.close()
