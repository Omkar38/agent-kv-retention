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

## 8. Week 8 — retraining on the decision

The obvious response to §2 is to stop predicting duration and predict the bit.
`src/classifier.py` trains on `1[T >= b]` directly, with the same feature
whitelist and the same session-level splits, so the comparison isolates the
change of target.

**The acting threshold is derived, not set to 0.5.** In normalised time
`u = T/b` the cost is dimensionless, and calling a pause long is cheaper iff
`p > p* = ΔS / (ΔS − ΔL)` where `ΔL = E[g_A − g_B | u ≥ 1]` and
`ΔS = E[g_A − g_B | u < 1]`. Measured on training data (`c` fixed, λ = 0.5):

    ΔL = −0.858    ΔS = +0.0295    →    p* = 0.033

A **29:1 asymmetry**. Calling a pause long is worth it at a 3.3% chance of
being right. That is the cost asymmetry of §2 expressed as a number.

### It fixed the bit

| Signal | Long-pause recall | Precision | Flip rate |
|---|---:|---:|---:|
| q50 (Week 4 selection) | 22.3% | 65.3% | 4.0% |
| q90 | 63.9% | 21.5% | 12.2% |
| classifier @ 0.5 | 38.0% | 42.7% | 5.1% |
| **classifier @ p\* = 0.033** | **87.0%** | 14.6% | 23.6% |

Recall nearly quadrupled, exactly as intended.

### It did not fix the result

| Signal | best λ | `c` fixed | vs baseline | `c` proportional | vs baseline |
|---|---|---:|---:|---:|---:|
| q50 | 0.75 | 1.3678 | −0.56% | 1.3281 | −1.27% |
| q90 | 0.75 / 0.5 | 1.3702 | −0.73% | **1.2726** | **+2.96%** |
| classifier @ 0.5 | 0.75 / 0.5 | **1.3627** | −0.18% | 1.2744 | +2.82% |
| classifier @ validation-selected | 0.75 / 0.5 | 1.3747 | −1.06% | 1.2910 | +1.57% |
| classifier @ p\* derived | 0.75 / 0.5 | 1.3747 | −1.06% | 1.3169 | −0.42% |

The classifier beats q50 in both cost modes, which confirms the target was
wrong. But it does not beat q90, and under `c` fixed nothing beats the
prediction-free rule. **Precision is the binding constraint, not recall**: at
87% recall the model calls ~27% of all pauses long when 4.5% are, and the
false positives cost more than the recovered true positives save.

### Two honest negatives

**The analytical threshold over-corrects.** `p* = 0.033` is derived from the
*marginal* distribution of `u`, but the decision-relevant conditional is the
one for pauses the model is unsure about, which is not the population. The
derivation gets the direction right — far below 0.5 — and the magnitude wrong.

**The threshold does not transfer between session samples.** Selected on
validation it is 0.04; the best value on test is about 0.20, and the two curves
have *opposite* shapes. With 362 validation and 384 test sessions, and the
decision driven by a few thousand long pauses, this knob is currently
unreliable. Reported as selected-on-validation rather than tuned on test.

**Calibration is good on average and bad where it matters.** Weighted error is
0.016, but that is dominated by the 85% of pauses in the lowest bin. In the
upper bins the model is badly overconfident — predicted 0.95 against observed
0.58, predicted 0.83 against observed 0.55. Isotonic calibration on 362
sessions cannot fix the tail. A single calibration number for a 4.5% base rate
is misleading and we should stop quoting one.

---

## 9. Week 8b — it was the features, not the dataset

§8 left precision as the binding constraint and asked whether the duration
signal simply is not there. It is there. The Week 4 feature set described the
*current* tool call — who is calling, which tool, how big the arguments and
context are — and never told the model what had already happened in the
session. That omission was the whole problem.

### The unused signal

| | value |
|---|---|
| lag-1 autocorrelation of log1p(pause) within session | **0.396** |
| P(long \| previous pause was long) | **37.5%** |
| P(long \| previous pause was short) | 4.1% |
| base rate | 6.1% |

A pause following a long pause is **6.15× more likely** to be long. The
information is present in the committed dataset, strictly observable at
decision time (round *k*−1 has completed before round *k* pauses), and was
never supplied. `src/features.py` builds it with an assertion that the first
pause of every session carries no history, so a mis-specified shift fails
loudly rather than silently leaking.

### What it bought

Average precision on the long-pause bit, held-out split:

| feature set | AP (c fixed) | AP (c proportional) |
|---|---:|---:|
| Week 4 base | 0.3362 | 0.1747 |
| **+ session history** | **0.5185** | **0.3419** |
| + clock | 0.5201 | 0.3281 |
| + user identity | 0.5161 | 0.3184 |

Session-bootstrapped AP gain: **+0.1822, 95% CI [+0.0584, +0.2793]**
(c fixed) and **+0.1672, [+0.0442, +0.2555]** (proportional). Precision at the
operating threshold roughly doubles (0.160 → 0.318 fixed).

Clock features and user identity add nothing once history is present — a
useful negative, since `user` carried a 14× spread in long-pause rate on its
own and would have been the riskiest feature to depend on.

### The headline negative is reversed

On the inherited 384-session split the realised-cost gain was +0.89% with a
confidence interval straddling zero. That is a power problem: the cost metric
is driven by the ~5% of pauses that are long, and **the inherited test split
has a 4.50% long-pause rate against 6.47% in train** — it is unrepresentative
on exactly the quantity the decision turns on.

`experiments/run_feature_cv.py` removes the limitation with 5-fold
cross-validation grouped by session, scoring all 3,667 sessions out-of-fold:

| c_mode | break-even | Week 4 features | + history |
|---|---:|---:|---:|
| fixed | 1.4814 | 1.4539 | **1.4219** |
| proportional | 1.4185 | 1.3918 | **1.3690** |

| comparison | gain | 95% CI | significant |
|---|---:|---|---|
| history vs break-even (fixed) | +0.0595 | [+0.0161, +0.1033] | **yes** |
| history vs break-even (proportional) | +0.0495 | [+0.0193, +0.0824] | **yes** |
| history vs Week 4 (fixed) | +0.0320 | [−0.0037, +0.0658] | no (P=0.96) |

**Prediction-aware retention now significantly beats the prediction-free rule
under both cost models.** The result survives excluding the rows used to pick
the operating threshold (fixed +0.0581, [+0.0135, +0.1048]; proportional
+0.0426, [+0.0052, +0.0721]).

This supplements rather than replaces the inherited split, which stays
canonical for every other result in the project.

### What the dataset genuinely does limit

Within identical (provider, tool, tool-count, input-size-decile) groups the
p90/p10 duration ratio is still **67×**. Some spread is irreducible given
anything knowable at decision time — the same `pytest` invocation really does
take 2 s or 200 s depending on what changed. Session history helps because it
proxies that hidden state, not because it removes it. So the honest summary is
that the dataset imposes a real ceiling, and we were nowhere near it.

---

## What this changes for the remaining weeks

1. ~~Retrain the predictor against the decision~~ — **done (§8)**. Lifted recall
   22.3% → 87.0%, bought nothing, precision fell to 14.6%.
2. ~~Attack precision~~ — **done (§9)**. Session history roughly doubles it,
   and prediction-aware retention now significantly beats the prediction-free
   rule under both cost models. This reverses the project's headline negative.
3. **Calibrate `B` and `c` on hardware — the critical path.** §1 shows the
   `c_mode` choice changes the size of the effect. The §9 result now holds in
   *both* modes, which makes the conclusion far more robust than it was, but
   absolute GPU-seconds still mean nothing until this lands.
4. **Re-run the simulator with the history-driven predictor.** Every simulator
   result above used the q50 or q90 signal. The §9 predictor is materially
   better and has not been through the capacity-constrained path, where the
   re-prefill bottleneck may change the ranking again.
5. **Replace the per-pause oracle with a system-level bound.** §7 shows it
   losing to a trivial policy on the metric we report.
6. **Fix the split, or stop relying on a single one.** The inherited test fold
   is unrepresentative on long-pause rate (4.50% vs 6.47% train). Either
   re-draw it stratified on that quantity or report cross-validated numbers
   throughout.
7. **Lead with the eviction-on-decode-end result.** Still the largest effect
   and independent of every contested modelling choice.

A full narrative version of all of the above, including Weeks 1–5, is in
[`docs/PROJECT_REPORT.md`](../../docs/PROJECT_REPORT.md) (and as a PDF).
