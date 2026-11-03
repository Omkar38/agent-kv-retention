from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Input and output files
# --------------------------------------------------

REPLAY_FILE = Path("results/week6_inputs/simulator_all_tool_pauses.csv.gz")
PREDICTION_FILE = Path("results/week6_inputs/predictions_test.csv")
OUTPUT_DIR = Path("results/week6_simulation")

# Capacities calculated from the validation split.
CAPACITIES = {
    "tight_25pct": 459541,
    "moderate_50pct": 919082,
    "roomier_100pct": 1838164,
}

# Each policy specifies how long a cache is retained.
FIXED_TTLS = {
    "immediate_eviction": 0.0,
    "fixed_ttl_1s": 1.0,
    "fixed_ttl_10s": 10.0,
}


def simulate(events, capacity, policy):
    """Simulate one cache-retention policy under one capacity limit."""

    resident = {}
    timeline = []

    for event in events:
        event_id = event["source_row_index"]
        start = event["start"]
        end = event["end"]

        if policy == "prediction_aware":
            ttl = event["prediction"]
        elif policy == "oracle":
            ttl = event["duration"]
        else:
            ttl = FIXED_TTLS[policy]

        ttl = max(0, round(float(ttl) * 1_000_000_000))

        # End events are processed before expiration events,
        # which are processed before new pause starts.
        timeline.append((end, 0, event_id, "end"))

        if ttl > 0:
            timeline.append((start + ttl, 1, event_id, "expire"))

        timeline.append((start, 2, event_id, "start"))

    timeline.sort()

    hits = 0
    misses = 0
    capacity_evictions = 0
    ttl_expirations = 0
    rejected_admissions = 0

    retained_token_seconds = 0.0
    missed_context_tokens = 0
    max_memory_used = 0
    active_tokens = 0

    event_lookup = {e["source_row_index"]: e for e in events}

    def release(event_id, current_time):
        nonlocal active_tokens, retained_token_seconds

        cached = resident.pop(event_id)
        active_tokens -= cached["tokens"]

        held_seconds = max(
            0.0, (current_time - cached["start"]) / 1_000_000_000
        )
        retained_token_seconds += cached["tokens"] * held_seconds

    for timestamp, priority, event_id, action in timeline:
        event = event_lookup[event_id]

        if action == "end":
            if event_id in resident:
                hits += 1
                release(event_id, timestamp)
            else:
                misses += 1
                missed_context_tokens += event["tokens"]

        elif action == "expire":
            if event_id in resident:
                ttl_expirations += 1
                release(event_id, timestamp)

        elif action == "start":
            tokens = event["tokens"]

            # Immediate eviction never admits the cache.
            if policy == "immediate_eviction":
                continue

            # A cache larger than the entire capacity cannot fit.
            if tokens > capacity:
                rejected_admissions += 1
                continue

            # Evict the oldest retained cache until there is room.
            while active_tokens + tokens > capacity:
                oldest_id = min(
                    resident,
                    key=lambda key: resident[key]["start"]
                )
                capacity_evictions += 1
                release(oldest_id, timestamp)

            resident[event_id] = {
                "start": timestamp,
                "tokens": tokens,
            }

            active_tokens += tokens
            max_memory_used = max(max_memory_used, active_tokens)

    return {
        "policy": policy,
        "capacity": capacity,
        "total_events": len(events),
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": round(100 * hits / len(events), 2),
        "capacity_evictions": capacity_evictions,
        "ttl_expirations": ttl_expirations,
        "rejected_admissions": rejected_admissions,
        "max_memory_used": max_memory_used,
        "retained_token_seconds": round(retained_token_seconds, 2),
        "missed_context_tokens": missed_context_tokens,
    }


def main():
    print("Loading replay data and predictions...")

    replay = pd.read_csv(REPLAY_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)

    # Use only positive-duration test events.
    test = replay[
        (replay["split"] == "test") &
        (replay["ground_truth_pause_s"] > 0)
    ].copy()

    # Match each test event to its HGB prediction.
    test = test.merge(
        predictions[
            ["source_row_index", "pred_hgb_q50_s"]
        ],
        on="source_row_index",
        how="inner",
        validate="one_to_one",
    )

    if len(test) != 47773:
        raise ValueError(
            f"Expected 47,773 matched events, found {len(test)}."
        )

    if not (
        test["ground_truth_pause_s"]
        .sub(test["target_s"] if "target_s" in test else test["ground_truth_pause_s"])
        .abs()
        .max() == 0
    ):
        raise ValueError("Actual pause durations do not match.")

    test["pause_start_at"] = pd.to_datetime(
        test["pause_start_at"], utc=True
    )
    test["pause_end_at"] = pd.to_datetime(
        test["pause_end_at"], utc=True
    )

    events = []

    for row in test.itertuples(index=False):
        events.append({
            "source_row_index": int(row.source_row_index),
            "start": row.pause_start_at.value,
            "end": row.pause_end_at.value,
            "duration": float(row.ground_truth_pause_s),
            "prediction": float(row.pred_hgb_q50_s),
            "tokens": int(row.context_tokens),
        })

    policies = [
        "immediate_eviction",
        "fixed_ttl_1s",
        "fixed_ttl_10s",
        "prediction_aware",
        "oracle",
    ]

    results = []

    print(f"Matched test events: {len(events):,}")
    print("Running policy comparisons...")

    for capacity_name, capacity in CAPACITIES.items():
        for policy in policies:
            result = simulate(events, capacity, policy)
            result["capacity_scenario"] = capacity_name
            results.append(result)

            print(
                f"{capacity_name:16s} | "
                f"{policy:20s} | "
                f"Hit rate: {result['hit_rate_pct']:6.2f}%"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "policy_comparison.csv"
    results_df.to_csv(output_file, index=False)

    print("\nSimulation complete.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()