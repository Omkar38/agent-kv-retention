
import duckdb

DB = "trace/syfi_coding_trace.duckdb"

TOOLS = [
    "Bash",
    "exec_command",
    "write_stdin",
    "exec",
    "shell_command",
]

con = duckdb.connect(DB, read_only=True)

# Inspect fields from the actual TraceLab schema.
query = """
SELECT
    r.provider,
    r.model,
    r.session_id,
    r.round_index,

    r.input_tokens_total,
    r.output_tokens,

    tc.tool_name,
    tc.tool_call_id,
    tc.emitted_at,
    tc.input_chars,

    -- Post-result fields: analysis only.
    tc.result_at,
    tc.result_chars,
    tc.is_error,
    tc.tool_wall_latency_ms,
    tc.tool_internal_latency_ms

FROM tool_calls AS tc
JOIN rounds AS r USING (round_pk)

WHERE tc.tool_name IN (
    'Bash',
    'exec_command',
    'write_stdin',
    'exec',
    'shell_command'
)

ORDER BY tc.emitted_at NULLS LAST
LIMIT 20
"""

df = con.execute(query).df()

print("\n=== Sample prediction fields ===")
print(df.to_string(index=False))

# Examine availability across all selected tool records.
coverage_query = """
SELECT
    r.provider,
    tc.tool_name,
    COUNT(*) AS total_calls,

    COUNT(tc.emitted_at) AS has_emitted_at,
    COUNT(tc.input_chars) AS has_input_chars,
    COUNT(r.input_tokens_total) AS has_context_tokens,
    COUNT(tc.result_at) AS has_result_at,
    COUNT(tc.tool_wall_latency_ms) AS has_wall_latency

FROM tool_calls AS tc
JOIN rounds AS r USING (round_pk)

WHERE tc.tool_name IN (
    'Bash',
    'exec_command',
    'write_stdin',
    'exec',
    'shell_command'
)

GROUP BY r.provider, tc.tool_name
ORDER BY total_calls DESC
"""

coverage = con.execute(coverage_query).df()

print("\n=== Feature availability ===")
print(coverage.to_string(index=False))

con.close()
