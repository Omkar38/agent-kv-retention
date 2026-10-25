#!/usr/bin/env python3
"""Consolidated TraceLab Week 3 analysis. Run against the TraceLab root directory.

The SQL bodies preserve the measured diagnostic and extraction logic of the
original Week 3 experiments. See README.md for data semantics and caveats.
"""

import argparse
import os
from pathlib import Path

def inspect_codex():
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


def inspect_late_matches():
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


def inspect_negative_gap():
    import duckdb
    
    con = duckdb.connect(
        "trace/syfi_coding_trace.duckdb",
        read_only=True
    )
    
    session = (
        "claude:b2145a8e-6787-4940-9a56-"
        "ccd7a3f46934/subagents/agent-a4a42d80d5baeb377"
    )
    
    query = """
    WITH per_round AS (
        SELECT
            r.round_pk,
            r.session_id,
            r.round_index,
            MIN(e.timestamp) AS first_event_at,
            MAX(e.timestamp) FILTER (
                WHERE e.event_type IN
                ('reasoning', 'text', 'tool_call', 'usage_report')
            ) AS last_output_at,
            MAX(e.timestamp) FILTER (
                WHERE e.event_type IN ('user_message', 'tool_result')
            ) AS last_input_at
        FROM rounds r
        JOIN timing_events e USING (round_pk)
        WHERE r.session_id = ?
        GROUP BY ALL
    ),
    paired AS (
        SELECT *,
            LEAD(round_index) OVER w AS next_round,
            LEAD(first_event_at) OVER w AS next_first_event,
            LEAD(last_input_at) OVER w AS next_input
        FROM per_round
        WINDOW w AS (
            PARTITION BY session_id
            ORDER BY round_index, first_event_at, round_pk
        )
    )
    SELECT
        round_index,
        last_output_at,
        next_round,
        next_first_event,
        next_input,
        EPOCH(next_input - last_output_at) AS pause_s
    FROM paired
    WHERE round_index BETWEEN 7 AND 9
    ORDER BY round_index
    """
    
    result = con.execute(query, [session]).df()
    print(result.to_string(index=False))
    
    con.close()


CASES = [
    ("codex", inspect_codex),
    ("late_calls", inspect_late_matches),
    ("negative_gap", inspect_negative_gap),
]

def main():
    parser = argparse.ArgumentParser(description="Inspect documented TraceLab timing anomalies.")
    parser.add_argument("--trace-root", type=Path, default=Path.cwd())
    parser.add_argument("--case", default="all", choices=["all"] + [name for name, _ in CASES])
    args = parser.parse_args()
    os.chdir(args.trace_root.expanduser().resolve())
    if not Path("trace/syfi_coding_trace.duckdb").is_file():
        parser.error("Missing trace/syfi_coding_trace.duckdb in --trace-root")
    for name, fn in CASES:
        if args.case not in ("all", name):
            continue
        if name == "late_calls" and not Path("group_work/pauses.parquet").exists():
            print("Skipping late-call case audit: group_work/pauses.parquet is absent.")
            continue
        print(f"\n=== Inspecting: {name} ===", flush=True)
        fn()

if __name__ == "__main__":
    main()
