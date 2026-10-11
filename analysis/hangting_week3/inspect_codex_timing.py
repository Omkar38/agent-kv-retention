
import duckdb

con = duckdb.connect(
    "trace/syfi_coding_trace.duckdb",
    read_only=True
)

session = "codex:019dd611-dd7c-7750-915b-910d54eaccbc"

print("\n=== Events in rounds 383 and 384 ===")

events = con.execute("""
    SELECT
        r.round_index,
        e.event_index,
        e.event_type,
        e.timestamp,
        e.tool_name,
        e.tool_call_id
    FROM rounds r
    JOIN timing_events e USING (round_pk)
    WHERE r.session_id = ?
      AND r.round_index IN (383, 384)
    ORDER BY r.round_index, e.event_index
""", [session]).df()

print(events.to_string(index=False))

print("\n=== Tool calls in rounds 383 and 384 ===")

tools = con.execute("""
    SELECT
        r.round_index,
        tc.tool_name,
        tc.tool_call_id,
        tc.emitted_at,
        tc.result_at,
        tc.tool_wall_latency_ms
    FROM tool_calls tc
    JOIN rounds r USING (round_pk)
    WHERE r.session_id = ?
      AND r.round_index IN (383, 384)
    ORDER BY r.round_index, tc.emitted_at
""", [session]).df()

print(tools.to_string(index=False))

con.close()
