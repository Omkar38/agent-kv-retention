# agent-kv-retention

**Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents**

How should an LLM serving system spend its GPU memory across many concurrent agent sessions that keep pausing for tool calls?

> **Project status:** Week 3 TraceLab workload characterization and validation are implemented on `Hangting_week3`. The derived tool-activity records support preliminary duration modeling, not yet validated GPU KV-cache retention intervals. Do not interpret exploratory outputs as measured latency or memory savings.

## Week 3 analysis — start here

For the complete branch-wide script guide, purpose, data provenance, execution commands, dataset feature/label rules, and the Weeks 4–10 handoff, see **[Week 3 README](analysis/validation/README.md)**. The [Week 3 validation report](analysis/validation/WEEK3_VALIDATION.md) records reproduced statistics and the Codex round-attribution issue. The branch contains `analysis/trace_facts/` latency reports and `analysis/validation/` overlap-aware candidate extraction plus QA. Brian's [original Week 3 analysis](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3) is linked rather than duplicated.

---

## The problem

LLM inference servers hold the key-value (KV) caches of many concurrent sessions in one fixed GPU memory pool. When the pool fills, arriving requests queue.

Tool-using agents break this. An agent generates a step, pauses to run tests or search a repository, then resumes with the tool output appended to a context that may span tens of thousands of tokens. During the pause the session generates nothing, but its KV cache still occupies GPU memory.

Current engines evict a cache as soon as decoding finishes, assuming a request that stops generating is complete. For an agent that assumption is wrong on every turn, and the returning request pays a full prefill recomputation plus the queueing delay incurred behind whatever was admitted in its place.

So at each pause the system must choose how long to retain the cache, without knowing how long the pause will last:

| Choice | Benefit | Cost |
|---|---|---|
| **Retain** in GPU memory | Instant resume | Holds memory others need, for unknown duration |
| **Offload** to host memory | Frees GPU memory | PCIe transfer both ways |
| **Drop** and recompute | Frees memory immediately | Full prefill on resume, plus re-queueing |

Tool-call durations make this hard. On SWE-Bench they average 925 ms with a standard deviation of 3,550 ms; the slowest 10% of `cd` calls account for 94.1% of that tool's total delay ([Continuum, arXiv:2511.02230](https://arxiv.org/abs/2511.02230)). A few very slow calls cause almost all the waiting, so predicting the average is not good enough.

## Cost model

For a paused request retained for time `τ`, with unobserved pause duration `T`:

```
C(τ, T) = c · min(T, τ) + B · 1[T > τ]
```

- `c` — per-second opportunity cost of occupying GPU memory
- `B` — eviction penalty: recompute or reload, plus queueing on return

The prediction-free break-even retention time is `B/c`, making this structurally a ski-rental decision. Two caveats we carry: `B` scales with context length so it varies per request, and `c` depends on load. We fix `c` in the core model and study the load-dependent case in simulation.

## Research question

> How should a serving system use imperfect predictions of tool-call duration to set KV-cache retention times, so as to approach the clairvoyant optimum when predictions are accurate without falling behind prediction-free policies when they are not?

## Gap

Continuum derives its TTL properly, selecting the retention time that maximizes expected net benefit against memory-occupancy cost. But its distribution is per-tool — conditioned on nothing about the specific call — and it is always fit to the same workload it is evaluated on. No published result says what happens when the recorded distribution and the live one diverge.

Meanwhile [Choi & Joshi](https://arxiv.org/abs/2608.30830) and [ConServe](https://arxiv.org/abs/2606.01839) argue per-request prediction may be unnecessary at all. The literature disagrees and nobody has tested the disagreement directly.

**We treat prediction quality as an independent variable** and characterize the regimes where prediction-aware retention beats prediction-free retention. We measure consistency and robustness empirically; we do not claim them as theorems.

## What we build

- **Discrete-event simulator** — primary results engine, trace-driven, calibrated against hardware measurements
- **Duration predictor** — quantile regression / GBM on tool identity, argument size, context length, turn index; outputs a distribution, not a point estimate
- **Retention policy** — `τ = π(T̂, c, B, λ)`, where `λ` interpolates between trusting the prediction and the prediction-free break-even rule
- **Offline optimum solver** — dynamic program over the trace with full hindsight, enabling empirical competitive ratios
- **Prototype** — vLLM / SGLang with LMCache for the offload tier (stretch goal; the simulator alone answers the research question)

## Evaluation

Seven policies as interchangeable functions over identical traces:

1. Immediate eviction (current vLLM / SGLang behavior)
2. Always-retain
3. LRU
4. Prediction-free break-even TTL
5. Continuum-style TTL *(our reconstruction from the published description; not comparable to figures in the original paper)*
6. Ours (λ-policy)
7. Offline optimum

**Swept:** memory budget (abundant → scarce), concurrency, prediction error (clean → biased → shifted → tail-heavy → adversarial).

**Measured:** job completion time, p95/p99 latency, throughput, recomputed tokens, KV memory utilization, empirical competitive ratio.

**Success criterion.** A single `λ` that stays close to the offline optimum on clean predictions and does not fall far behind the best prediction-free policy on adversarial ones. If no such `λ` exists, that is a reportable result: it would mean the duration signal is not worth acting on for these workloads, which no prior work has tested.

## Workload

[TraceLab](https://arxiv.org/abs/2606.30560) — ~4,300 coding-agent sessions, ~350,000 LLM steps, ~430,000 tool calls from Claude Code and Codex. Supplemented with SWE-Bench (mini-swe-agent) and BFCL v4 runs.

## Repository layout

```
agent-kv-retention/
├── analysis/
│   ├── trace_facts/          # tool latency and session summaries
│   └── validation/           # tool-activity dataset, audits, handoff
├── src/
│   ├── sim/                  # discrete-event simulator
│   ├── policies/             # the seven retention policies
│   ├── predictor/            # duration prediction + calibration
│   └── oracle/               # offline optimum solver
├── experiments/
│   ├── configs/              # one YAML per run
│   └── scripts/
├── data/                     # gitignored — raw traces and measurements
├── results/                  # committed — figures and summary CSVs only
├── notebooks/                # analysis only, never pipeline logic
└── requirements.txt
```

## Reproducibility

Every experiment run writes its config file and the git commit hash into its results directory. Raw data stays out of git; summary CSVs and figures are committed. Predictor train/test splits are **by session**, never by individual pause — random splits leak later turns of a session into training.

## Getting started

```bash
git clone https://github.com/Omkar38/agent-kv-retention.git
cd agent-kv-retention
pip install -r requirements.txt
```

*Run instructions will land with the simulator in Week 6.*

## Team

Group 9 — CSE 542 / AMS 560, Stony Brook University, Fall 2026

| Member | Owns |
|---|---|
| Hangting Lu | Formal cost model, retention policy design |
| Omkar Lashkare | Prediction, calibration, evaluation methodology |
| Brian Park | Trace processing, workload characterization |
| Zijie Shen | Simulator, baseline policies, offline optimum |
| David Mei | Hardware calibration, experiment harness, reproducibility |

## Key references

| | |
|---|---|
| Continuum / CacheTTL | [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) |
| InferCept | [arXiv:2402.01869](https://arxiv.org/abs/2402.01869) |
| SAGA | [arXiv:2605.00528](https://arxiv.org/abs/2605.00528) |
| TokenCake | [arXiv:2510.18586](https://arxiv.org/abs/2510.18586) |
| Adaptive KV Retention | [arXiv:2608.30830](https://arxiv.org/abs/2608.30830) |
| ConServe | [arXiv:2606.01839](https://arxiv.org/abs/2606.01839) |
| TraceLab | [arXiv:2606.30560](https://arxiv.org/abs/2606.30560) |
| Purohit, Svitkina & Kumar | NeurIPS 2018 |
| Wei & Zhang | NeurIPS 2020 |

## License

MIT — see [LICENSE](LICENSE).
