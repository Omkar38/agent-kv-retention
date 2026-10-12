import sys
import duckdb

DB = "trace/syfi_coding_trace.duckdb"
session = sys.argv[1] if len(sys.argv) > 1 else "claude:9dda6cbb-174c-4536-b993-c79c836ac161"

con = duckdb.connect(DB, read_only=True)
con.sql("""
    SELECT r.round_index        AS rnd,
           r.input_tokens_total AS context_tokens,
           e.event_type,
           strftime(e.timestamp, '%H:%M:%S.%g') AS time,
           e.tool_name
    FROM rounds r
    JOIN timing_events e USING (round_pk)
    WHERE r.session_id = ?
    ORDER BY r.round_index, e.event_index
""", params=[session]).show(max_rows=100)
