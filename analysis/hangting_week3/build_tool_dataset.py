#!/usr/bin/env python3
"""Consolidated TraceLab Week 3 analysis. Run against the TraceLab root directory.

The SQL bodies preserve the measured diagnostic and extraction logic of the
original Week 3 experiments. See README.md for data semantics and caveats.
"""

import argparse
import os
from pathlib import Path

def extract_tool_intervals():
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


def group_tool_episodes():
    import duckdb
    
    con = duckdb.connect()
    
    SOURCE = "group_work/tool_intervals.parquet"
    OUTPUT = "group_work/tool_activity_episodes.parquet"
    
    query = f"""
    COPY (
        WITH valid_calls AS (
            SELECT *
            FROM read_parquet('{SOURCE}')
            WHERE timing_status = 'valid'
        ),
    
        ordered AS (
            SELECT *,
                MAX(tool_end_ts) OVER (
                    PARTITION BY provider, session_id
                    ORDER BY
                        tool_start_ts,
                        tool_end_ts,
                        round_pk,
                        tool_index
                    ROWS BETWEEN
                        UNBOUNDED PRECEDING
                        AND 1 PRECEDING
                ) AS previous_max_end
            FROM valid_calls
        ),
    
        marked AS (
            SELECT *,
                CASE
                    WHEN previous_max_end IS NULL
                      OR tool_start_ts >= previous_max_end
                        THEN 1
                    ELSE 0
                END AS new_episode
            FROM ordered
        ),
    
        numbered AS (
            SELECT *,
                SUM(new_episode) OVER (
                    PARTITION BY provider, session_id
                    ORDER BY
                        tool_start_ts,
                        tool_end_ts,
                        round_pk,
                        tool_index
                    ROWS BETWEEN
                        UNBOUNDED PRECEDING
                        AND CURRENT ROW
                ) AS episode_id
            FROM marked
        ),
    
        episodes AS (
            SELECT
                provider,
                session_id,
                episode_id,
    
                MIN(tool_start_ts) AS start_ts,
                MAX(tool_end_ts) AS end_ts,
    
                COUNT(*) AS n_calls,
                COUNT(DISTINCT round_pk) AS n_rounds,
    
                STRING_AGG(
                    DISTINCT tool_name,
                    ',' ORDER BY tool_name
                ) AS tool_names
    
            FROM numbered
            GROUP BY provider, session_id, episode_id
        )
    
        SELECT *,
            EPOCH(end_ts - start_ts)
                AS tool_activity_duration_s,
    
            n_calls > 1 AS multiple_calls
    
        FROM episodes
    )
    TO '{OUTPUT}' (FORMAT PARQUET)
    """
    
    con.execute(query)
    
    print(f"Saved: {OUTPUT}")
    
    summary = con.execute(f"""
    SELECT
        provider,
        COUNT(*) AS episodes,
        COUNT(*) FILTER (
            WHERE multiple_calls
        ) AS multi_call_episodes,
        SUM(n_calls) AS total_calls,
        ROUND(
            MEDIAN(tool_activity_duration_s), 3
        ) AS median_s,
        ROUND(
            QUANTILE_CONT(
                tool_activity_duration_s, 0.99
            ), 3
        ) AS p99_s
    FROM read_parquet('{OUTPUT}')
    GROUP BY provider
    ORDER BY provider
    """).df()
    
    print("\n=== Tool activity episodes ===")
    print(summary.to_string(index=False))
    
    con.close()


def construct_candidates():
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


def audit_candidates():
    import duckdb
    
    DB = "trace/syfi_coding_trace.duckdb"
    INPUT = "group_work/tool_episode_candidates.parquet"
    OUTPUT = "group_work/tool_episode_candidates_audited.parquet"
    
    con = duckdb.connect(DB, read_only=True)
    
    query = f"""
    COPY (
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
        flags AS (
            SELECT
                c.provider,
                c.session_id,
                c.episode_id,
    
                COUNT(*) FILTER (
                    WHERE s.event_type IN ('reasoning', 'text')
                ) AS generation_events_inside,
    
                COUNT(*) FILTER (
                    WHERE s.event_type = 'user_message'
                ) AS user_messages_inside
    
            FROM read_parquet('{INPUT}') c
    
            LEFT JOIN signals s
              ON s.provider = c.provider
             AND s.session_id = c.session_id
             AND s.timestamp > c.candidate_start_ts
             AND s.timestamp < c.tool_activity_end_ts
    
            GROUP BY
                c.provider, c.session_id, c.episode_id
        )
        SELECT
            c.*,
    
            f.generation_events_inside,
            f.user_messages_inside,
    
            (
                c.observed_n_calls = 1
                AND f.generation_events_inside = 0
                AND f.user_messages_inside = 0
            ) AS simple_tool_wait_proxy
    
        FROM read_parquet('{INPUT}') c
        JOIN flags f
          USING (provider, session_id, episode_id)
    )
    TO '{OUTPUT}' (FORMAT PARQUET)
    """
    
    con.execute(query)
    
    summary = con.execute(f"""
    SELECT
        provider,
        COUNT(*) AS total_candidates,
    
        COUNT(*) FILTER (
            WHERE simple_tool_wait_proxy
        ) AS simple_candidates,
    
        COUNT(*) FILTER (
            WHERE NOT simple_tool_wait_proxy
        ) AS complex_candidates,
    
        COUNT(*) FILTER (
            WHERE generation_events_inside > 0
        ) AS generation_overlap,
    
        COUNT(*) FILTER (
            WHERE user_messages_inside > 0
        ) AS user_input_overlap
    
    FROM read_parquet('{OUTPUT}')
    GROUP BY provider
    ORDER BY provider
    """).df()
    
    print("\n=== Audited Week 3 candidates ===")
    print(summary.to_string(index=False))
    
    con.close()


STEPS = [
    ("intervals", extract_tool_intervals),
    ("episodes", group_tool_episodes),
    ("candidates", construct_candidates),
    ("audit", audit_candidates),
]

def main():
    parser = argparse.ArgumentParser(description="Extract audited TraceLab tool-activity candidates.")
    parser.add_argument("--trace-root", type=Path, default=Path.cwd(), help="TraceLab directory with trace/syfi_coding_trace.duckdb")
    parser.add_argument("--step", default="all", choices=["all"] + [name for name, _ in STEPS])
    args = parser.parse_args()
    os.chdir(args.trace_root.expanduser().resolve())
    if not Path("trace/syfi_coding_trace.duckdb").is_file():
        parser.error("Missing trace/syfi_coding_trace.duckdb in --trace-root")
    Path("group_work").mkdir(exist_ok=True)
    for name, fn in STEPS:
        if args.step in ("all", name):
            print(f"\n=== Building: {name} ===", flush=True)
            fn()

if __name__ == "__main__":
    main()
