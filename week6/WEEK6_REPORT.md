# Week 6: Prediction-Aware KV-Cache Retention Simulation

## 1. Objective

The objective of Week 6 was to develop a simplified simulator to evaluate different KV-cache retention policies for tool-using AI agents.

When an agent pauses to execute an external tool, its KV cache can either be retained in memory or evicted to free memory for other requests. Retaining the cache can reduce the need for recomputation when the agent resumes, but it also consumes memory during the pause.

This experiment compares fixed retention times with a prediction-aware policy that uses the pause-duration predictions developed in Week 4.

## 2. Dataset and Prediction Model

The experiment uses the TraceLab pause dataset and the Week 4 prediction pipeline developed by Hangting.

The selected prediction model was `hgb_q50`, a histogram gradient boosting quantile regression model predicting the median pause duration.

The test dataset contained 47,773 positive-duration events matched to their corresponding predictions using `source_row_index`. The actual pause durations in the replay data matched the prediction dataset's target durations.

The simulator uses `context_tokens` as a proxy for cache memory demand. These values are not measurements of actual GPU KV-cache memory consumption.

## 3. Simulation Methodology

Five retention policies were evaluated:

1. **Immediate eviction:** The cache is evicted as soon as the agent pauses.
2. **Fixed TTL 1 second:** The cache is retained for one second.
3. **Fixed TTL 10 seconds:** The cache is retained for ten seconds.
4. **Prediction-aware:** The cache is retained for the duration predicted by the HGB median-duration model.
5. **Oracle:** The cache is retained for the actual pause duration. This is an idealized reference because the actual duration is not known when making a real retention decision.

The simulator uses a FIFO eviction rule when memory capacity is exceeded. If a new cache cannot fit, the oldest retained cache is evicted until sufficient capacity is available. Caches larger than the entire capacity are rejected.

Three capacity scenarios were derived from the validation dataset's peak simultaneous context-token demand:

| Scenario | Capacity |
|---|---:|
| Tight (25%) | 459,541 context-token units |
| Moderate (50%) | 919,082 context-token units |
| Roomier (100%) | 1,838,164 context-token units |

These are experimental capacity settings expressed in context-token units, not physical GPU-memory capacities.

All five policies were evaluated using the same 47,773 test events.

## 4. Results

### Cache Hit Rate

| Policy | Tight (25%) | Moderate (50%) | Roomier (100%) |
|---|---:|---:|---:|
| Immediate eviction | 0.00% | 0.00% | 0.00% |
| Fixed TTL 1s | 59.13% | 61.68% | 61.83% |
| Fixed TTL 10s | 84.80% | 88.96% | 89.29% |
| Prediction-aware | 45.90% | 48.13% | 48.27% |
| Oracle | 94.88% | 99.60% | 100.00% |

Fixed 10-second TTL achieved the highest hit rate among the practical policies tested. Prediction-aware retention did not outperform this baseline.

The oracle achieved a 100% hit rate under the roomier capacity scenario because it had sufficient capacity to retain every cache until its corresponding pause ended.

### Memory-Time and Missed Context Tokens

The following results show the trade-off under the roomier capacity scenario:

| Policy | Retained token-seconds | Missed context tokens |
|---|---:|---:|
| Fixed TTL 1s | 4.11 billion | 3.33 billion |
| Fixed TTL 10s | 18.38 billion | 0.93 billion |
| Prediction-aware | 11.45 billion | 4.21 billion |
| Oracle | 215.46 billion | 0 |

Fixed 10-second TTL achieved a higher hit rate and fewer missed context tokens than prediction-aware retention, but required more memory-time.

Prediction-aware retention used less memory-time than fixed 10-second TTL, but resulted in substantially more cache misses.

The oracle required substantially more memory-time because some pauses were extremely long.

## 5. Main Findings

The experiment produced several observations:

- Fixed 10-second TTL was the strongest practical policy in terms of cache hit rate across all three capacity scenarios.
- The current prediction-aware policy did not outperform fixed TTL, despite using learned pause-duration predictions.
- Prediction-aware retention reduced memory-time compared with fixed 10-second TTL, but this came at the cost of more cache misses.
- Increasing capacity improved cache hit rates, although the improvement from moderate to roomier capacity was relatively small for the practical policies.
- Perfect knowledge of pause duration can provide a useful reference, but long pauses can result in substantial memory-time consumption.

Overall, these results show that using a learned pause-duration prediction does not automatically produce a better retention policy. The usefulness of predictions depends on how they are translated into retention decisions and how memory pressure affects other active caches.

## 6. Limitations

This is a simplified trace-driven simulation rather than a measurement on actual GPU hardware.

- TraceLab pause durations are used as proxies for agent tool-call pauses.
- Context-token counts are used as proxies for cache memory demand; they do not represent actual KV-cache sizes or GPU VRAM consumption.
- Cache misses and missed context tokens represent potential recomputation, not measured recomputation time or end-to-end latency.
- The FIFO eviction rule is a simplified replacement strategy.
- The oracle policy uses actual pause durations and is not deployable as a real-time policy.
- The capacity scenarios are normalized experimental settings derived from validation data rather than hardware-calibrated memory limits.

Consequently, the results should be interpreted as a comparison of retention behavior under simplified assumptions, not as proof of actual GPU latency improvements.

## 7. Conclusion

Week 6 produced a working simulator for comparing fixed and prediction-aware KV-cache retention policies using the Week 4 pause-duration predictions.

In the evaluated test workload, fixed 10-second TTL achieved better cache hit rates than the current prediction-aware policy across all three capacity scenarios. Prediction-aware retention used less memory-time than fixed 10-second TTL, but resulted in more cache misses.

The experiment therefore does not demonstrate that the current prediction-aware policy improves upon fixed TTL. Instead, it highlights the trade-off between cache retention, memory pressure, and the quality of pause-duration predictions. These results provide a baseline for further investigation into how imperfect predictions can be incorporated into memory-management decisions.