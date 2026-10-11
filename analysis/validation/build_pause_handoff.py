#!/usr/bin/env python3
"""Connect Brian's original pause table with Hangting's interval/timing audit.

Run from TraceLab root after group_work/pauses.parquet (Brian) and
group_work/tool_intervals.parquet (Hangting) have been generated.

Produces quality-labeled pauses_v2.parquet and a conservative Week 4
single-tool pause-prediction dataset. No source records are overwritten.
No claim of observing actual GPU KV-cache residency is made.
"""
import argparse
import hashlib
from pathlib import Path


def quote_path(path):
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def split_for(provider, session_id):
    key = f"tracelab-v0.0.1|{provider}|{session_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path.cwd())
    parser.add_argument("--brian-pauses", type=Path, default=Path("group_work/pauses.parquet"))
    parser.add_argument("--tool-intervals", type=Path, default=Path("group_work/tool_intervals.parquet"))
    args = parser.parse_args()
    root = args.trace_root.expanduser().resolve()
    db = root / "trace/syfi_coding_trace.duckdb"
    brian = args.brian_pauses if args.brian_pauses.is_absolute() else root / args.brian_pauses
    tool_data = args.tool_intervals if args.tool_intervals.is_absolute() else root / args.tool_intervals
    for path in (db, brian, tool_data):
        if not path.is_file():
            parser.error(f"Missing {path}. Generate Brian's pause table and Hangting's intervals first.")
    import duckdb

    out = root / "group_work"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db), read_only=True)
    con.execute(f"CREATE TEMP VIEW brian_pauses AS SELECT * FROM read_parquet({quote_path(brian)})")
    con.execute(f"CREATE TEMP VIEW validated_intervals AS SELECT * FROM read_parquet({quote_path(tool_data)})")

    # Preserve original Brian rows. Duplicate session/round indices have no
    # unique linkage to an underlying round_pk; keep but mark them ambiguous.
    # The later result is retrospective eligibility evidence, not a predictor.
    query = """
    WITH anchor AS (
        SELECT r.provider, r.session_id, r.round_pk, r.round_index,
               MIN(e.timestamp) AS first_event_ts,
               MAX(e.timestamp) FILTER (WHERE e.event_type IN
                   ('reasoning','text','tool_call','usage_report')) AS stop_proxy_ts,
               MAX(e.timestamp) FILTER (WHERE e.event_type IN
                   ('tool_result','user_message')) AS last_input_ts,
               MIN(e.timestamp) FILTER (WHERE e.event_type IN
                   ('reasoning','text','tool_call')) AS first_generation_signal_ts
        FROM rounds r LEFT JOIN timing_events e USING (round_pk)
        GROUP BY r.provider, r.session_id, r.round_pk, r.round_index
    ),
    marked AS (
        SELECT *, COUNT(*) OVER (
            PARTITION BY provider, session_id, round_index
        ) AS repeated_index_count
        FROM anchor
    ),
    paired AS (
        SELECT *,
               LEAD(round_pk) OVER w AS next_round_pk,
               LEAD(repeated_index_count) OVER w AS next_repeat_count,
               LEAD(last_input_ts) OVER w AS next_last_input_ts,
               LEAD(first_generation_signal_ts) OVER w AS next_first_generation_signal_ts
        FROM marked
        WINDOW w AS (PARTITION BY provider, session_id
                     ORDER BY round_index, first_event_ts NULLS LAST, round_pk)
    ),
    pair_unique AS (SELECT * FROM paired WHERE repeated_index_count = 1),
    next_inputs AS (
        SELECT p.round_pk,
               COUNT(*) FILTER (WHERE e.event_type = 'tool_result') AS next_n_tool_results,
               COUNT(*) FILTER (WHERE e.event_type = 'user_message') AS next_n_user_messages,
               MIN(e.timestamp) FILTER (WHERE e.event_type IN
                   ('tool_result','user_message')) AS next_first_input_ts,
               MAX(e.timestamp) FILTER (WHERE e.event_type IN
                   ('tool_result','user_message')) AS next_last_input_event_ts
        FROM paired p LEFT JOIN timing_events e ON e.round_pk = p.next_round_pk
        GROUP BY p.round_pk
    ),
    source_calls AS (
        SELECT p.round_pk,
               COUNT(t.round_pk) AS all_source_calls,
               COUNT(t.round_pk) FILTER (WHERE t.timing_status = 'valid'
                   AND t.tool_start_ts <= p.stop_proxy_ts) AS known_valid_source_calls,
               COUNT(t.round_pk) FILTER (WHERE t.tool_start_ts IS NULL
                   OR t.tool_start_ts > p.stop_proxy_ts) AS unverified_call_emission,
               MIN(t.tool_name) FILTER (WHERE t.timing_status = 'valid'
                   AND t.tool_start_ts <= p.stop_proxy_ts) AS one_tool_name,
               MIN(t.input_chars) FILTER (WHERE t.timing_status = 'valid'
                   AND t.tool_start_ts <= p.stop_proxy_ts) AS one_tool_input_chars,
               MIN(t.tool_duration_s) FILTER (WHERE t.timing_status = 'valid'
                   AND t.tool_start_ts <= p.stop_proxy_ts) AS one_tool_duration_s
        FROM paired p LEFT JOIN validated_intervals t
          ON t.provider = p.provider AND t.session_id = p.session_id
         AND t.round_pk = p.round_pk
        GROUP BY p.round_pk
    ),
    matched_results AS (
        SELECT p.round_pk,
               COUNT(*) AS matched_result_events,
               COUNT(DISTINCT t.tool_call_id) AS matched_distinct_calls,
               MIN(e.timestamp) AS first_matched_result_ts,
               MAX(e.timestamp) AS last_matched_result_ts,
               COUNT(*) FILTER (WHERE e.timestamp < p.stop_proxy_ts)
                   AS results_before_stop,
               COUNT(*) FILTER (WHERE e.timestamp IS NULL) AS results_missing_ts
        FROM paired p
        JOIN timing_events e ON e.round_pk = p.next_round_pk
          AND e.event_type = 'tool_result'
        JOIN validated_intervals t ON t.provider = p.provider
          AND t.session_id = p.session_id AND t.round_pk = p.round_pk
          AND t.tool_call_id = e.tool_call_id AND t.timing_status = 'valid'
          AND t.tool_start_ts <= p.stop_proxy_ts
        GROUP BY p.round_pk
    ),
    original AS (SELECT ROW_NUMBER() OVER () AS original_row_id, * FROM brian_pauses),
    joined AS (
        SELECT b.* EXCLUDE (original_row_id), b.original_row_id,
               p.round_pk, p.next_round_pk, p.next_repeat_count,
               p.stop_proxy_ts, p.next_last_input_ts,
               p.next_first_generation_signal_ts,
               n.next_n_tool_results, n.next_n_user_messages,
               n.next_first_input_ts, n.next_last_input_event_ts,
               COALESCE(s.all_source_calls, 0) AS all_source_calls,
               COALESCE(s.known_valid_source_calls, 0) AS known_valid_source_calls,
               COALESCE(s.unverified_call_emission, 0) AS unverified_call_emission,
               s.one_tool_name, s.one_tool_input_chars, s.one_tool_duration_s,
               COALESCE(m.matched_result_events, 0) AS matched_result_events,
               COALESCE(m.matched_distinct_calls, 0) AS matched_distinct_calls,
               COALESCE(m.results_before_stop, 0) AS results_before_stop,
               COALESCE(m.results_missing_ts, 0) AS results_missing_ts,
               m.first_matched_result_ts, m.last_matched_result_ts,
               EPOCH(p.next_last_input_ts - p.stop_proxy_ts) AS corrected_gap_proxy_s,
               b.pause_s AS brian_pause_s,
               b.cause AS brian_cause,
               b.usable AS brian_usable,
               b.n_tools AS brian_n_tools,
               b.tool_names AS brian_tool_names,
               b.tool_input_chars AS brian_tool_input_chars,
               b.context_tokens AS context_tokens_proxy
        FROM original b
        LEFT JOIN pair_unique p ON p.provider = b.provider
          AND p.session_id = b.session_id AND p.round_index = b.round_index
        LEFT JOIN next_inputs n ON n.round_pk = p.round_pk
        LEFT JOIN source_calls s ON s.round_pk = p.round_pk
        LEFT JOIN matched_results m ON m.round_pk = p.round_pk
    ),
    quality AS (
        SELECT *,
            CASE
              WHEN round_pk IS NULL THEN 'ambiguous_or_missing_round_index'
              WHEN brian_usable IS NOT TRUE THEN 'original_not_usable'
              WHEN next_round_pk IS NULL THEN 'terminal_round'
              WHEN next_repeat_count > 1 THEN 'duplicate_next_round_index'
              WHEN stop_proxy_ts IS NULL OR next_last_input_ts IS NULL
                   THEN 'boundary_timestamp_missing'
              WHEN next_last_input_ts < stop_proxy_ts
                   OR next_first_input_ts < stop_proxy_ts
                   THEN 'negative_or_overlapping_boundary'
              WHEN next_first_generation_signal_ts IS NULL
                   THEN 'next_generation_signal_missing'
              WHEN next_first_input_ts IS NULL
                   OR next_last_input_event_ts > next_first_generation_signal_ts
                   THEN 'next_round_inputs_interleaved_with_generation'
              WHEN ABS(corrected_gap_proxy_s - brian_pause_s) > 0.001
                   THEN 'brian_target_mismatch'
              WHEN next_n_user_messages > 0 AND next_n_tool_results > 0
                   THEN 'mixed_tool_and_user_wait'
              WHEN next_n_user_messages > 0 THEN 'human_wait_not_tool_target'
              WHEN next_n_tool_results = 0 THEN 'no_next_tool_results'
              WHEN brian_cause <> 'tool' THEN 'brian_cause_not_tool'
              WHEN all_source_calls <> 1 OR known_valid_source_calls <> 1
                   THEN 'multi_or_unverified_source_calls'
              WHEN unverified_call_emission > 0
                   THEN 'call_emission_not_decision_time_safe'
              WHEN next_n_tool_results <> 1 OR matched_result_events <> 1
                   OR matched_distinct_calls <> 1
                   THEN 'unmatched_or_multiple_next_results'
              WHEN results_before_stop > 0 OR results_missing_ts > 0
                   THEN 'result_timestamp_invalid'
              WHEN one_tool_name IS NULL OR one_tool_input_chars IS NULL
                   THEN 'source_tool_feature_missing'
              ELSE 'strict_single_tool_pause_proxy'
            END AS quality_reason
        FROM joined
    )
    SELECT *,
           quality_reason = 'strict_single_tool_pause_proxy' AS week4_strict_eligible,
           CASE WHEN quality_reason = 'strict_single_tool_pause_proxy'
                THEN corrected_gap_proxy_s ELSE NULL END AS target_pause_s,
           CASE WHEN quality_reason = 'strict_single_tool_pause_proxy'
                THEN one_tool_name ELSE NULL END AS tool_name_at_decision,
           CASE WHEN quality_reason = 'strict_single_tool_pause_proxy'
                THEN one_tool_input_chars ELSE NULL END AS tool_input_chars_at_decision
    FROM quality
    """
    con.execute("CREATE TEMP TABLE handoff AS " + query)
    original_count = con.execute("SELECT COUNT(*) FROM brian_pauses").fetchone()[0]
    actual_count = con.execute("SELECT COUNT(*) FROM handoff").fetchone()[0]
    if original_count != actual_count:
        raise RuntimeError(f"Changed original row count: {original_count} -> {actual_count}")
    con.execute(f"COPY handoff TO {quote_path(out / 'pauses_v2.parquet')} (FORMAT PARQUET)")

    sessions = con.execute("SELECT DISTINCT provider, session_id FROM handoff").fetchall()
    con.execute("CREATE TEMP TABLE splits(provider VARCHAR, session_id VARCHAR, split VARCHAR)")
    con.executemany("INSERT INTO splits VALUES (?, ?, ?)", [
        (provider, session_id, split_for(provider, session_id))
        for provider, session_id in sessions
    ])
    # Export only verified provisional decision-time features. Keep model and
    # context_tokens_proxy in pauses_v2 for separate timing verification.
    con.execute("""
        CREATE TEMP TABLE week4 AS
        SELECT h.provider, h.session_id, h.round_index, s.split,
               h.tool_name_at_decision, h.tool_input_chars_at_decision,
               h.target_pause_s
        FROM handoff h JOIN splits s USING (provider, session_id)
        WHERE h.week4_strict_eligible
    """)
    strict_count = con.execute("SELECT COUNT(*) FROM week4").fetchone()[0]
    if not strict_count:
        raise RuntimeError("No strict observations; check trace version and audit criteria")
    invalid = con.execute("""
        SELECT COUNT(*) FROM week4
        WHERE target_pause_s IS NULL OR target_pause_s < 0
           OR tool_name_at_decision IS NULL
    """).fetchone()[0]
    duplicate_keys = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT provider, session_id, round_index
            FROM week4 GROUP BY ALL HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if invalid or duplicate_keys:
        raise RuntimeError(f"Invalid training rows={invalid}, duplicate keys={duplicate_keys}")
    con.execute(f"COPY week4 TO {quote_path(out / 'week4_pause_train.parquet')} (FORMAT PARQUET)")

    print("\n=== Week 4 handoff quality by provider ===")
    print(con.execute("""
        SELECT provider, quality_reason, COUNT(*) AS rows
        FROM handoff GROUP BY provider, quality_reason
        ORDER BY provider, rows DESC
    """).df().to_string(index=False))
    print("\n=== Strict Week 4 pause-proxy cohort and session splits ===")
    print(con.execute("""
        SELECT provider, split, COUNT(*) AS rows,
               COUNT(DISTINCT session_id) AS sessions,
               ROUND(MEDIAN(target_pause_s),3) AS median_pause_s
        FROM week4 GROUP BY provider, split ORDER BY provider, split
    """).df().to_string(index=False))
    print("\n=== Brian's original target vs reconstructed endpoints ===")
    print(con.execute("""
        SELECT provider, COUNT(*) AS compared,
               COUNT(*) FILTER (
                 WHERE ABS(brian_pause_s-corrected_gap_proxy_s)>0.001
               ) AS mismatches_over_1ms
        FROM handoff
        WHERE brian_pause_s IS NOT NULL AND corrected_gap_proxy_s IS NOT NULL
        GROUP BY provider ORDER BY provider
    """).df().to_string(index=False))
    print(f"\nPreserved {actual_count:,} Brian records; strict subset: {strict_count:,}.")
    print(f"Saved {out / 'pauses_v2.parquet'}")
    print(f"Saved {out / 'week4_pause_train.parquet'}")
    print("Warning: strict cohort selected with hindsight; not representative of all agent waits.")
    con.close()


if __name__ == "__main__":
    main()
