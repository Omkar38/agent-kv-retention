#!/usr/bin/env python3
"""Consolidated TraceLab Week 3 analysis. Run against the TraceLab root directory.

The SQL bodies preserve the measured diagnostic and extraction logic of the
original Week 3 experiments. See README.md for data semantics and caveats.
"""

import argparse
import os
from pathlib import Path

def check_within_round_results():
    import duckdb
    
    con = duckdb.connect(
        "trace/syfi_coding_trace.duckdb",
        read_only=True
    )
    
    query = """
    WITH generation_start AS (
        SELECT
            round_pk,
            MIN(timestamp) FILTER (
                WHERE event_type IN (
                    'reasoning',
                    'text',
                    'tool_call'
                )
            ) AS first_generation_at
        FROM timing_events
        GROUP BY round_pk
    )
    SELECT
        r.provider,
        COUNT(*) AS total_tool_results,
    
        COUNT(*) FILTER (
            WHERE e.timestamp > g.first_generation_at
        ) AS results_after_generation_started,
    
        COUNT(*) FILTER (
            WHERE e.timestamp <= g.first_generation_at
        ) AS results_before_or_at_generation,
    
        COUNT(*) FILTER (
            WHERE g.first_generation_at IS NULL
               OR e.timestamp IS NULL
        ) AS timing_unknown
    
    FROM timing_events e
    JOIN rounds r USING (round_pk)
    JOIN generation_start g USING (round_pk)
    
    WHERE e.event_type = 'tool_result'
    GROUP BY r.provider
    ORDER BY r.provider
    """
    
    result = con.execute(query).df()
    
    print("\n=== Within-round tool-result timing ===")
    print(result.to_string(index=False))
    
    con.close()


def check_interval_consistency():
    import duckdb
    
    con = duckdb.connect()
    
    DATA = "group_work/tool_intervals.parquet"
    
    print("\n=== 1. Timestamp consistency ===")
    
    result = con.execute("""
    SELECT
        provider,
        COUNT(*) AS compared_calls,
        ROUND(
            AVG(ABS(
                tool_duration_s -
                reported_tool_duration_s
            )), 6
        ) AS mean_difference_s,
        ROUND(
            QUANTILE_CONT(ABS(
                tool_duration_s -
                reported_tool_duration_s
            ), 0.99), 6
        ) AS p99_difference_s,
        COUNT(*) FILTER (
            WHERE ABS(
                tool_duration_s -
                reported_tool_duration_s
            ) > 0.1
        ) AS differences_over_100ms
    FROM read_parquet(?)
    WHERE timing_status = 'valid'
      AND reported_tool_duration_s IS NOT NULL
    GROUP BY provider
    ORDER BY provider
    """, [DATA]).df()
    
    print(result.to_string(index=False))
    
    print("\n=== 2. Overlapping tool calls ===")
    
    overlap = con.execute("""
    WITH ordered AS (
        SELECT
            provider,
            session_id,
            tool_start_ts,
            tool_end_ts,
    
            MAX(tool_end_ts) OVER (
                PARTITION BY provider, session_id
                ORDER BY
                    tool_start_ts,
                    tool_end_ts,
                    round_pk,
                    tool_index
                ROWS BETWEEN
                    UNBOUNDED PRECEDING AND
                    1 PRECEDING
            ) AS previous_max_end
    
        FROM read_parquet(?)
        WHERE timing_status = 'valid'
    )
    SELECT
        provider,
        COUNT(*) AS valid_calls,
        COUNT(*) FILTER (
            WHERE tool_start_ts < previous_max_end
        ) AS potentially_overlapping_calls
    FROM ordered
    GROUP BY provider
    ORDER BY provider
    """, [DATA]).df()
    
    print(overlap.to_string(index=False))
    
    con.close()


def check_episode_generation():
    import duckdb
    
    DB = "trace/syfi_coding_trace.duckdb"
    EPISODES = "group_work/tool_activity_episodes.parquet"
    
    con = duckdb.connect(DB, read_only=True)
    
    query = """
    WITH signals AS (
        SELECT
            r.provider,
            r.session_id,
            e.event_type,
            e.timestamp
        FROM timing_events e
        JOIN rounds r USING (round_pk)
        WHERE e.event_type IN (
            'reasoning', 'text', 'user_message'
        )
    ),
    episode_flags AS (
        SELECT
            ep.provider,
            ep.session_id,
            ep.episode_id,
    
            COUNT(s.timestamp) FILTER (
                WHERE s.event_type IN ('reasoning', 'text')
            ) AS generation_events_inside,
    
            COUNT(s.timestamp) FILTER (
                WHERE s.event_type = 'user_message'
            ) AS user_messages_inside
    
        FROM read_parquet(?) ep
        LEFT JOIN signals s
          ON s.provider = ep.provider
         AND s.session_id = ep.session_id
         AND s.timestamp > ep.start_ts
         AND s.timestamp < ep.end_ts
    
        GROUP BY
            ep.provider,
            ep.session_id,
            ep.episode_id
    )
    SELECT
        provider,
        COUNT(*) AS total_episodes,
    
        COUNT(*) FILTER (
            WHERE generation_events_inside > 0
        ) AS episodes_with_generation,
    
        COUNT(*) FILTER (
            WHERE user_messages_inside > 0
        ) AS episodes_with_user_input,
    
        COUNT(*) FILTER (
            WHERE generation_events_inside = 0
              AND user_messages_inside = 0
        ) AS episodes_without_these_signals
    
    FROM episode_flags
    GROUP BY provider
    ORDER BY provider
    """
    
    result = con.execute(query, [EPISODES]).df()
    
    print("\n=== Episode generation audit ===")
    print(result.to_string(index=False))
    
    con.close()


def check_candidate_boundaries():
    import duckdb
    
    DB = "trace/syfi_coding_trace.duckdb"
    CANDIDATES = "group_work/tool_episode_candidates.parquet"
    
    con = duckdb.connect(DB, read_only=True)
    
    query = """
    WITH round_outputs AS (
        SELECT
            round_pk,
    
            MAX(timestamp) FILTER (
                WHERE event_type IN ('reasoning', 'text')
            ) AS last_generation_event,
    
            MAX(timestamp) FILTER (
                WHERE event_type = 'tool_call'
            ) AS last_tool_call_event
    
        FROM timing_events
        GROUP BY round_pk
    )
    
    SELECT
        c.provider,
    
        COUNT(*) AS candidates,
    
        COUNT(*) FILTER (
            WHERE r.last_generation_event >
                  c.candidate_start_ts
        ) AS generation_after_first_call,
    
        COUNT(*) FILTER (
            WHERE r.last_tool_call_event >
                  c.candidate_start_ts
        ) AS later_tool_calls_same_round,
    
        COUNT(*) FILTER (
            WHERE c.first_tool_input_chars IS NULL
        ) AS missing_input_chars,
    
        COUNT(*) FILTER (
            WHERE c.context_tokens <= 0
        ) AS nonpositive_context
    
    FROM read_parquet(?) c
    LEFT JOIN round_outputs r
        ON c.first_call_round_pk = r.round_pk
    
    GROUP BY c.provider
    ORDER BY c.provider
    """
    
    result = con.execute(query, [CANDIDATES]).df()
    
    print("\n=== Candidate boundary validation ===")
    print(result.to_string(index=False))
    
    con.close()


def check_prior_pause_feature_timing():
    import duckdb
    
    con = duckdb.connect(
        "trace/syfi_coding_trace.duckdb",
        read_only=True
    )
    
    query = """
    WITH round_times AS (
        SELECT
            r.round_pk,
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
            r.round_pk, r.session_id, r.round_index
    ),
    
    paired AS (
        SELECT *,
            LEAD(round_pk) OVER (
                PARTITION BY session_id
                ORDER BY round_index, first_event_at, round_pk
            ) AS next_round_pk
    
        FROM round_times
    ),
    
    valid_pauses AS (
        SELECT p.*
        FROM paired p
    
        JOIN read_parquet('group_work/pauses.parquet') q
          ON q.session_id = p.session_id
         AND q.round_index = p.round_index
    
        WHERE q.usable IS TRUE
          AND q.cause IN ('tool', 'tool+user')
    ),
    
    matched_calls AS (
        SELECT
            p.session_id,
            p.round_index,
            p.decision_ts,
    
            tc.tool_name,
            tc.tool_call_id,
            tc.emitted_at
    
        FROM valid_pauses p
    
        JOIN timing_events e
          ON e.round_pk = p.next_round_pk
         AND e.event_type = 'tool_result'
    
        JOIN tool_calls tc
          ON tc.tool_call_id = e.tool_call_id
    
        JOIN rounds source_round
          ON source_round.round_pk = tc.round_pk
         AND source_round.session_id = p.session_id
    )
    
    SELECT
        COUNT(*) AS matched_call_records,
    
        COUNT(*) FILTER (
            WHERE emitted_at <= decision_ts
        ) AS emitted_before_decision,
    
        COUNT(*) FILTER (
            WHERE emitted_at > decision_ts
        ) AS emitted_after_decision,
    
        COUNT(*) FILTER (
            WHERE emitted_at IS NULL
               OR decision_ts IS NULL
        ) AS cannot_verify
    
    FROM matched_calls
    """
    
    result = con.execute(query).df()
    
    print("\n=== Decision-time feature validation ===")
    print(result.to_string(index=False))
    
    con.close()


STEPS = [
    ("within_round", check_within_round_results),
    ("intervals", check_interval_consistency),
    ("episode_generation", check_episode_generation),
    ("candidate_boundaries", check_candidate_boundaries),
    ("prior_pause_feature_timing", check_prior_pause_feature_timing),
]

def main():
    parser = argparse.ArgumentParser(description="Run TraceLab Week 3 validation checks.")
    parser.add_argument("--trace-root", type=Path, default=Path.cwd())
    parser.add_argument("--step", default="all", choices=["all"] + [name for name, _ in STEPS])
    args = parser.parse_args()
    os.chdir(args.trace_root.expanduser().resolve())
    if not Path("trace/syfi_coding_trace.duckdb").is_file():
        parser.error("Missing trace/syfi_coding_trace.duckdb in --trace-root")
    for name, fn in STEPS:
        if args.step not in ("all", name):
            continue
        if name == "prior_pause_feature_timing" and not Path("group_work/pauses.parquet").exists():
            print("Skipping original pause-table attribution audit: group_work/pauses.parquet is absent.")
            continue
        print(f"\n=== Validating: {name} ===", flush=True)
        fn()

if __name__ == "__main__":
    main()
