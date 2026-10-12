#!/usr/bin/env python3
"""Independent, non-destructive audit of Version 2 pause versus tool evidence.

Uses TraceLab's raw tool_calls and timing_events; does not change the official
pause_handoff.parquet or relabel ground truth. In particular, runner-internal
duration cannot by itself reconstruct an unlogged tool start/LLM stop time.
"""
import argparse
from pathlib import Path
import duckdb


def qpath(p):
    return "'" + str(p).replace("'", "''") + "'"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.trace_root.expanduser().resolve()
    trace = root / "trace/syfi_coding_trace.duckdb"
    handoff = root / "group_work/pause_handoff.parquet"
    for p in (trace, handoff):
        if not p.is_file():
            ap.error(f"Missing {p}")
    con = duckdb.connect(str(trace), read_only=True)
    source = qpath(handoff)

    print("\n=== Independent event evidence for Version 2 pauses ===")
    # Match source rounds and their immediate next round within a session.
    # Duplicate round indexes are surfaced separately rather than multiplied
    # into a join. Next-round results are matched only to the source round's
    # tool calls (or next-round co-logged calls for legacy Codex).
    query = f"""
    WITH ordered_rounds AS (
        SELECT r.provider, r.session_id, r.round_index, r.round_pk,
               ROW_NUMBER() OVER (
                 PARTITION BY r.provider,r.session_id,r.round_index
                 ORDER BY r.round_pk
               ) AS occurrence,
               LEAD(r.round_pk) OVER (
                 PARTITION BY r.provider,r.session_id
                 ORDER BY r.round_index,r.round_pk
               ) AS next_round_pk,
               COUNT(*) OVER (
                 PARTITION BY r.provider,r.session_id,r.round_index
               ) AS index_multiplicity
        FROM rounds r
    ),
    base AS (
        SELECT h.provider, h.session_id, h.round_index,
               h.log_format, h.record_category, h.timing_category,
               h.pause_s, h.pause_start_at, h.pause_end_at,
               r.round_pk, r.next_round_pk, r.index_multiplicity
        FROM read_parquet({source}) h
        LEFT JOIN ordered_rounds r
          ON h.provider=r.provider AND h.session_id=r.session_id
         AND h.round_index=r.round_index AND r.occurrence=1
    ),
    source_calls AS (
        SELECT b.provider,b.session_id,b.round_index,
               COUNT(t.tool_call_id) AS source_calls,
               MAX(t.tool_internal_latency_ms)/1000.0
                   AS max_source_runner_s,
               MAX(t.result_at) AS latest_source_result_at
        FROM base b LEFT JOIN tool_calls t ON t.round_pk=b.round_pk
        GROUP BY b.provider,b.session_id,b.round_index
    ),
    next_results AS (
        SELECT b.provider,b.session_id,b.round_index,
               COUNT(e.tool_call_id) AS all_next_round_result_events,
               COUNT(tc.tool_call_id) AS matched_next_result_events,
               -- Only result timestamps from tools attributed to this pause.
               -- Later tools issued by the NEXT inference round are excluded.
               MAX(CASE WHEN tc.tool_call_id IS NOT NULL
                        THEN e.timestamp END) AS latest_matched_result_at,
               MAX(CASE WHEN tc.tool_call_id IS NOT NULL
                             AND tc.tool_internal_latency_ms >= 0
                        THEN tc.tool_internal_latency_ms/1000.0 END)
                   AS max_next_result_runner_s
        FROM base b
        LEFT JOIN timing_events e ON e.round_pk=b.next_round_pk
                                 AND e.event_type='tool_result'
        LEFT JOIN (
            SELECT tc.*, tr.session_id
            FROM tool_calls tc JOIN rounds tr USING (round_pk)
        ) tc ON tc.tool_call_id=e.tool_call_id
              AND tc.session_id=b.session_id
              AND (
                   (b.log_format='codex_old' AND tc.round_pk=b.next_round_pk)
                   OR
                   (b.log_format <> 'codex_old' AND tc.round_pk=b.round_pk)
              )
        GROUP BY b.provider,b.session_id,b.round_index
    )
    SELECT b.*, sc.source_calls, sc.max_source_runner_s,
           sc.latest_source_result_at, nr.all_next_round_result_events, nr.matched_next_result_events,
           nr.latest_matched_result_at, nr.max_next_result_runner_s
    FROM base b
    LEFT JOIN source_calls sc USING(provider,session_id,round_index)
    LEFT JOIN next_results nr USING(provider,session_id,round_index)
    """
    con.execute("CREATE TEMP VIEW evidence AS "+query)
    summary = con.execute("""
      SELECT log_format, record_category, COUNT(*) AS rows,
             COUNT(*) FILTER (WHERE index_multiplicity>1)
                 AS ambiguous_round_index,
             COUNT(*) FILTER (
                 WHERE pause_s >= 0 AND max_next_result_runner_s > pause_s + 0.1
             ) AS next_result_runner_exceeds_pause_100ms,
             COUNT(*) FILTER (
                 WHERE pause_s >= 0 AND max_source_runner_s > pause_s + 0.1
             ) AS source_runner_exceeds_pause_100ms,
             COUNT(*) FILTER (
                 WHERE pause_end_at IS NOT NULL AND latest_matched_result_at IS NOT NULL
                   AND latest_matched_result_at > pause_end_at
             ) AS matched_result_logged_after_pause_end,
             ROUND(MEDIAN(pause_s),3) AS pause_p50_s,
             ROUND(MEDIAN(max_next_result_runner_s),3) AS next_result_runner_p50_s,
             ROUND(MEDIAN(max_source_runner_s),3) AS source_runner_p50_s
      FROM evidence
      GROUP BY log_format,record_category
      ORDER BY log_format,rows DESC
    """).df()
    print(summary.to_string(index=False))
    print("\n=== Old-format Codex examples requiring timing inspection ===")
    examples = con.execute("""
      SELECT session_id,round_index,pause_s,
             max_next_result_runner_s,max_source_runner_s,
             pause_start_at,pause_end_at,latest_matched_result_at,
             index_multiplicity
      FROM evidence
      WHERE log_format='codex_old' AND record_category='tool_wait'
        AND index_multiplicity = 1
        AND max_next_result_runner_s > pause_s + 0.1
      ORDER BY (max_next_result_runner_s-pause_s) DESC
      LIMIT 20
    """).df()
    print(examples.to_string(index=False))
    print("\nCAVEAT: A tool's runner time exceeding the recorded pause is a")
    print("measurement discrepancy to investigate, not a corrected pause duration.")
    print("Old-format call/result timestamps may be co-logged at completion.")
    print("No official dataset rows or columns have been changed.")


if __name__ == "__main__":
    main()
