import duckdb

DB = "trace/syfi_coding_trace.duckdb"

con = duckdb.connect(DB)

query = """
SELECT
    r.provider,
    r.model,
    r.session_id,
    r.round_index,

    tc.tool_name,
    tc.tool_call_id,
    tc.emitted_at,
    tc.result_at,

    tc.input_chars,
    tc.result_chars,

    tc.command_skeleton,
    tc.executables,

    tc.command_status,
    tc.command_exit_code,

    tc.tool_wall_latency_ms,
    tc.tool_internal_latency_ms

FROM tool_calls AS tc
JOIN rounds AS r
    USING (round_pk)

WHERE tc.tool_name IN (
    'Bash',
    'exec_command',
    'write_stdin',
    'exec',
    'shell_command'
)

ORDER BY tc.emitted_at

LIMIT 20
"""

rows = con.execute(query).fetchall()

columns = [
    "provider",
    "model",
    "session_id",
    "round_index",
    "tool_name",
    "tool_call_id",
    "emitted_at",
    "result_at",
    "input_chars",
    "result_chars",
    "command_skeleton",
    "executables",
    "command_status",
    "command_exit_code",
    "tool_wall_latency_ms",
    "tool_internal_latency_ms",
]

for i, row in enumerate(rows, start=1):
    print(f"\n{'=' * 80}")
    print(f"ROW {i}")
    print(f"{'=' * 80}")

    for column, value in zip(columns, row):
        print(f"{column}: {value}")

con.close()