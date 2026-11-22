# Week 9 — Cost calibration and sensitivity

## Objective

Measure prefill throughput on a real GPU and separate hardware observations
from assumptions in the single-pause retention cost model:

C(tau, T) = c * min(T, tau) + B * 1[T > tau]; break-even b = B/c.

In the current model, B = context_tokens / prefill_tokens_per_s plus an
optional queueing penalty. c is the **memory opportunity cost**. Measuring
prefill speed alone does not measure c. Report both assumptions:

- fixed: c = c0, a constant cost per session;
- proportional: c = c0 * context_tokens / kv_budget_tokens.

The benchmark **does not** estimate c0 or establish physical KV-cache capacity.
Peak CUDA allocation includes model weights and activations, not isolated KV
bytes. Do not modify the canonical pause dataset or overwrite prior results.

## 1. Collect actual prefill measurements

On a CUDA GPU with sufficient memory and a compatible causal LM, install torch,
transformers, and accelerate in a suitable environment, then from repo root:

~~~bash
python3 -m pip install -r week4/requirements.txt
python3 -m pip install torch transformers accelerate

python3 -m experiments.benchmark_prefill \
  --model-id /path/to/local-or-accessible-causal-model \
  --context-lengths 1024,2048,4096 \
  --warmup 2 --repeats 5 \
  --output results/calibration/prefill_measurements.csv
~~~

The script records context_tokens, prefill_time_s, throughput, repeat index,
model, dtype and GPU name. It also writes a JSON manifest with environment and
methodology. It errors rather than creating fake measurements on machines
without CUDA. Increase context lengths only when supported by the model/GPU.

If using a separate harness, supply CSV with at least this header:

~~~text
context_tokens,prefill_time_s
~~~

Each row must be an actual, positive measured prefill observation, not a
guessed value. Retain repeats and hardware details. The harness uses random
token IDs to control length, measures a synchronized forward pass with cache
enabled, and does not simulate queueing or competing requests.

## 2. Analyze measured prefill and modeled break-even time

~~~bash
python3 -m experiments.calibrate_costs \
  --measurements results/calibration/prefill_measurements.csv \
  --contexts 4096,16384,65536,122065,236575 \
  --budgets 300000,600000,1200000 \
  --c0-values 0.1,0.2,0.4,0.8 \
  --out-dir results/calibration

python3 -m unittest discover -s tests -p 'test_calibration.py' -v
~~~

The analysis produces break_even_scenarios.csv and calibration_manifest.json
in results/calibration. The throughput estimate is the median of each measured
context_tokens / prefill_time_s, accompanied by P10 and P90. It is a
simplifying constant-throughput approximation; inspect size-dependent changes
before transferring it to very long contexts.

The resulting B and b are MODEL ESTIMATES CONDITIONAL ON assumptions, not
physical measurements of GPU waiting time. c0 and KV budgets in the table are
sensitivity inputs rather than independently calibrated values. To use the
measured throughput in a later policy run:

~~~python
from src.costs import CostModel

cost = CostModel(
    prefill_tokens_per_s=MEASURED_MEDIAN_TOKENS_PER_SECOND,
    kv_budget_tokens=600000,  # must be independently justified
    c_mode="fixed",           # also analyze proportional
    c0=0.4,                   # scenario assumption, NOT measured
)
~~~

The two new scripts do NOT rewrite the old cost model defaults, dataset, or
prior results. A follow-on experiment should rerun all relevant retention
policies under matched traces and the calibrated prefill assumptions.

## 3. Acceptance criteria

1. Record the real GPU, model, dtype, package versions and timing method.
2. Include positive prefill measurements for multiple context lengths.
3. Compare both fixed and proportional occupancy-cost assumptions.
4. State that c0 is uncalibrated; prefill calibration alone does not establish c.
5. Preserve the original 357,161-row, 33-column dataset and past experiment results.
6. Distinguish simulator sensitivity from real-world latency improvements.

## Remaining beyond the first Week 9 step

To calibrate c itself requires a serving experiment measuring how retained
KV occupancy displaces work under controlled concurrent loads. Measure queueing
and admission delays, actual GPU KV-byte demand and refill costs under the
target serving scheduler. The prefill harness measures a standalone model
forward pass, not an end-to-end vLLM or SGLang reload.

If serving hardware is unavailable, describe the result as prefill-throughput
calibration plus c sensitivity, NOT full hardware cost calibration.