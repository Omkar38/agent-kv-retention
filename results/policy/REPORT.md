# Weeks 6–7 — Retention policy, simulator, and the prediction-quality sweep

**Status:** complete and reproducible. All numbers below come from
`experiments/` run against the held-out TraceLab sessions. Cost-model constants
are **not** hardware-calibrated, so absolute GPU-seconds are not meaningful;
every claim here is a comparison between policies under one shared model.

Reproduce with:

```bash
python3 -m experiments.run_single_pause   # per-pause scoring + error sweep
python3 -m experiments.run_sweeps         # budget × concurrency grid
python3 -m experiments.run_scarce         # the regime where memory binds
python3 -m experiments.make_figures
python3 -m unittest discover -s tests -p 'test_*.py'
```

---

## Headline

**The duration signal is worth about 24% of the achievable cost. Our predictor
captures none of it, because it was selected for the wrong objective.**

The policy reads the prediction through exactly one bit — whether the predicted
pause exceeds the break-even time `b = B/c`. Week 4 selected `hgb_q50` by
validation log1p-MAE, an objective dominated by the 95% of pauses that are
short. The bit the policy needs is about the 4.5% that are long, and on that
bit q50 has **22.3% recall**.

| | total cost ÷ clairvoyant optimum |
|---|---|
| Clairvoyant (per-pause oracle) | 1.000 |
| λ = 0.1 with a **perfect** prediction | 1.036 |
| Prediction-free break-even TTL | 1.360 |
| λ = 0.1 with our **actual** q50 prediction | 1.717 |

Acting on a perfect prediction would cut cost **23.8%** below the
prediction-free rule. Acting on our prediction makes it **26.2% worse**. The
gap is a predictor problem, not a policy problem.

---

## 1. Open decision #2 is settled, and it changes the answer

Whether `c` is per-request was flagged in the Week-1 literature notes and never
resolved. It is now measured. `src/costs.py` implements both.

| `c_mode` | break-even `b` across held-out pauses | spread |
|---|---|---|
| `fixed` — `c` constant per session | p1 3.5 s · p50 36.2 s · p99 209.2 s | **622×** |
| `proportional` — `c ∝ cache size` | 75.0 s for every request | **1.0×** |

Under `proportional`, `B` and `c` both scale with context length and cancel
exactly. Two consequences, both confirmed in `signal_comparison.csv`:

- **A fixed TTL stops being a weak baseline.** At `c` proportional,
  `fixed_ttl_75s` and `break_even_ttl` score *identically* (1.3115 both) —
  they are the same policy. The §5.2 argument that per-tool TTLs are blind to
  `B` only holds under `fixed`.
- **It decides whether prediction helps at all** (§3).

This is not a question the project can leave open. It is the difference between
a real gap and no gap.

## 2. Where the λ-policy gains and loses

Decomposition at `c` fixed, q50, λ = 0.5 (`decomposition.csv`), costs in
thousand GPU-seconds:

| cell | n | share | clairvoyant | break-even | λ = 0.5 | vs break-even |
|---|---:|---:|---:|---:|---:|---:|
| correct short | 45,367 | 95.0% | 45.8 | 45.8 | 45.8 | — |
| over-called (short→long) | 255 | 0.5% | 1.0 | 1.0 | 1.8 | +0.8 |
| **correct long** | 479 | 1.0% | 3.8 | 7.6 | 5.7 | **−1.9** |
| **under-called (long→short)** | 1,672 | 3.5% | 22.6 | 45.1 | 50.3 | **+5.2** |

Three things follow:

1. **95% of pauses are unaffected by any policy.** They return before the
   shortest deadline anyone would set, so every policy ties at ratio 1.
2. **The only cell where λ can gain is "correct long"** — and under-called long
   pauses outnumber it 3.5:1. The policy loses 3.2× what it gains.
3. **Over-calling is nearly free** (+0.8 across 255 pauses) because a short
   pause returns before even a shortened deadline. Under-calling is expensive.
   The cost is asymmetric, so the signal should be too.

## 3. Fixing the signal partially fixes the result

Point 3 above predicts that an upper quantile should beat the median, despite
worse log error. It does:

| signal | long-pause recall | precision | flip rate |
|---|---:|---:|---:|
| q50 (Week 4 selection) | 22.3% | 65.3% | 4.0% |
| q90 | **63.9%** | 21.5% | 12.2% |

Total cost ÷ optimum, best λ per cell (`signal_comparison.csv`):

| | break-even | λ best with q50 | λ best with q90 |
|---|---:|---:|---:|
| `c` fixed | **1.3602** | 1.3678 (λ=0.75) | 1.3702 (λ=0.75) |
| `c` proportional | 1.3115 | 1.3281 (λ=0.75) | **1.2726 (λ=0.5)** |

**Under `c` proportional, q90 at λ = 0.5 beats the prediction-free baseline by
3.0%.** That is the only configuration in the entire study where acting on a
prediction pays. Under `c` fixed, nothing beats break-even.

## 4. Against the proposal's success criterion

The proposal set the bar at: a single λ that (a) beats every prediction-free
baseline at zero prediction error, and (b) stays within a bounded margin of the
best prediction-free baseline under adversarial error.

**No λ satisfies (a) under `c` fixed, with either signal.** Under `c`
proportional, λ = 0.5 with q90 satisfies (a) by 3.0%.

(b) holds everywhere by construction, and `worst_case_probe.csv` confirms the
implementation attains `1 + 1/λ` exactly at both tight cases and never exceeds
it — 11.00, 5.00, 3.00, 2.33 for λ = 0.1, 0.25, 0.5, 0.75.

The proposal anticipated this outcome:

> If no such λ exists, that is itself a reportable result — it would say the
> duration signal is not worth acting on for these workloads, which no prior
> system has tested.

The sharper version is that the signal **is** worth acting on — the oracle row
proves 23.8% is available — but only a predictor built for the decision can
collect it. That is a stronger and more useful finding than either "it works"
or "it doesn't".

## 5. Continuum-style does poorly, and the reason is structural

Our reconstruction scores 1.661 (`c` fixed) and 2.227 (`c` proportional),
**worse than the prediction-free break-even rule** in both. Its eviction rate
is 13.7% against the oracle's 4.5% — it evicts roughly three times too often.

The mechanism is the one the gap statement names: the per-tool empirical CDF
conditions on the tool and nothing about the specific call, so it cannot see
`B`. Two sessions calling the same tool with 12k and 500k of context get the
same TTL despite break-even times that differ by 40×.

*This is our reconstruction from the published description, fitted on the
training split and evaluated on held-out sessions. It is not comparable to
figures in the original paper, which reports against its own workload and
hardware.*

## 6. Degradation under deliberate prediction failure

`single_pause_costs.csv`, 19 error conditions × 2 cost modes. `break_even_ttl`
and `continuum_style` are flat across all of them by construction — neither
reads a per-call prediction — which is a useful check that the harness is
wired correctly.

Two things worth stating carefully:

- **The `under`/`over`/`oracle` conditions are derived from the true durations.**
  They describe a *perfect predictor made biased*, not our predictor degraded.
  That is why moderate under-prediction (α = 0.8) **improves** on break-even:
  it is still a near-oracle signal. Do not read those rows as robustness
  results for the real system.
- **Adversarial flips plateau between severity 0.05 and 0.5.** Not a bug: by
  severity 0.05 every truly-long pause has already been flipped, and the
  additional victims are short pauses, which cost nothing to mis-call because
  they return before any deadline. The curve moves again only at severity 1.0.
  This is the cost asymmetry of §2 showing up directly.

## 7. Simulator: what memory pressure actually does

Two findings invalidate a naive reading of the budget sweep, and both are
instrumented rather than assumed.

**Budgets below 600k tokens are not measuring retention.** At a 150k budget,
**47.6%** of admissions have a context larger than the entire pool (median
context is 122k, p90 is 237k). No eviction can make those fit. Those rows
measure an undersized pool. `oversized_admit_fraction` is reported on every
run; it is 9.1% at 300k, 3.5% at 600k, 0% at 1.2M.

**The trace cannot saturate a realistic pool.** With 384 held-out sessions,
utilisation plateaus near 0.49 at a 600k budget no matter how hard arrival gaps
are compressed, because concurrency is bounded by session overlap rather than
arrival rate. Raising load requires replicating the session population
(`replicate_sessions`), which describes a hypothetical server at N× this
trace's concurrency — load generation, not additional evidence.

**The binding constraint is re-prefill, not capacity.** Under replication, job
completion time scales linearly with load while utilisation stays near 0.64.
The bottleneck is the serialised re-prefill path: evictions cause re-prefill,
and re-prefill saturates the GPU. This is the channel through which retention
policy reaches job completion time, and it is *not* the channel the per-pause
cost model represents.

### Under genuine load, voluntary eviction never pays

`scarce.csv` — 600k-token pool, 8× replicated load (3,072 sessions),
utilisation 0.638, oversized admissions 3.2%:

| policy | mean JCT | vs always-retain |
|---|---:|---:|
| **always retain + LRU** | **72,538 s** | **1.00×** |
| fixed TTL 36 s | 73,915 s | 1.02× |
| clairvoyant (per-pause oracle) | 74,165 s | 1.02× |
| break-even TTL | 74,941 s | 1.03× |
| λ = 0.5 (q90) | 75,281 s | 1.04× |
| Continuum-style | 92,114 s | 1.27× |
| immediate eviction | 580,300 s | 8.00× |

The ordering is identical at both budgets and all three load levels. Two
things stand out:

**Holding everything and letting LRU sort it out beats every TTL policy.**
Voluntary eviction causes re-prefill, re-prefill is the serialised bottleneck,
and LRU evicts only when genuinely forced to. A TTL that releases a cache
"early" is spending the scarce resource to conserve the abundant one.

**The per-pause clairvoyant loses to always-retain** (74,165 vs 72,538). This
is not a bug: it is optimal for the modelled per-pause cost `C(τ,T)`, which
charges occupancy continuously and assumes eviction is cheap at the margin.
The simulator charges eviction its true systemic price. The gap between those
two is precisely the thing the single-pause model cannot represent, and it
means a policy can be per-pause-optimal and system-suboptimal at once.

### The one result that dwarfs everything else

| policy | mean JCT, 600k budget, arrival scale 2000 |
|---|---:|
| immediate eviction (current vLLM / SGLang) | 67,976 s |
| every other policy | 8,525 – 10,560 s |

Across every valid configuration (budget ≥ 600k, three arrival scales),
immediate eviction is **6.4× to 14.7× worse** than the alternatives — the low
end against the weakest retention policy, the high end against the strongest.
It is also insensitive to the memory budget, because it never holds anything
for memory pressure to act on. Differences *among* retention policies are
1–2% by comparison.

That ordering is the practically important result of this project: the large
win is available to anyone who stops treating "decoding finished" as "request
complete". Choosing cleverly *among* retention policies is a second-order
refinement on this workload.

---

## Limitations

- **`B` and `c` are uncalibrated.** `prefill_tokens_per_s = 10,000` and
  `kv_budget_tokens = 300,000` are A100-class estimates for an 8B model, not
  measurements. `c0 = 0.4` is set so the two cost modes agree at the median
  context, making their difference one of spread rather than level. Week 9
  hardware calibration is still required before any absolute claim.
- **The clairvoyant is a per-pause oracle, not a system optimum.** It minimises
  the modelled per-pause cost, which is why `always_retain` sometimes beats it
  on simulated JCT. A true offline optimum over the coupled system is not
  computed; this is the Week-7 go/no-go fallback the proposal anticipated.
- **Session affinity is assumed.** A returning request always reaches the cache
  it left. SAGA's ablation puts the cost of violating this at 96% of task
  completion time, far larger than anything measured here.
- **No offload tier.** The cost model is two-way; InferCept and Choi & Joshi are
  three-way. Open decision #1 is still open, and excluding offload makes these
  results the conservative case.
- **Trace-derived pauses are proxies** for GPU cache idle time, inherited from
  the Week 3/4 caveat. They are not residency measurements.
- **Replicated-session runs are synthetic load.** They say what would happen at
  higher concurrency; they are not evidence about this trace.

## What this changes for the remaining weeks

1. **Retrain the predictor against the decision, not the distribution.** The
   target should be `P(T ≥ b)` — a calibrated classifier on the bit the policy
   reads — rather than a conditional median of duration. Week 4's model
   selection by log1p-MAE optimised a quantity the policy never consumes.
2. **Settle `c` before the hardware calibration run.** It determines whether a
   fixed TTL is a strong or weak baseline, and whether prediction helps at all.
3. **Lead with the eviction-on-decode-end result.** It is the largest, most
   robust, and most actionable finding, and it does not depend on any of the
   contested modelling choices.
