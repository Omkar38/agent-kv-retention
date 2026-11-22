# SeaWulf execution — Week 9 prefill calibration

This directory contains all files needed to run the Week 9 prefill benchmark
and cost-model sensitivity analysis on Stony Brook's SeaWulf A100 partition.

**Paths are fixed for this run:**

- Working repository: /gpfs/scratch/hanglu/agent-kv-retention
- All generated outputs: **/gpfs/scratch/hanglu/results**
- Python environment: results/venv
- Downloaded model: results/models/Qwen2.5-7B
- Run artifacts: results/week9_JOBID/
- Slurm logs: results/week9_JOBID.out and results/week9_JOBID.err

The repository and its scripts live outside results, but no measurements,
manifests, logs, model downloads, Python environments or caches are written
elsewhere intentionally.

## 1. Log in and clone the new branch

From your local computer:

~~~bash
ssh hanglu@milan.seawulf.stonybrook.edu
~~~

On SeaWulf:

~~~bash
mkdir -p /gpfs/scratch/hanglu/results
cd /gpfs/scratch/hanglu
git clone -b week9-cost-calibration \
  https://github.com/Omkar38/agent-kv-retention.git agent-kv-retention
cd agent-kv-retention
~~~

If you already have the repository instead of cloning again:

~~~bash
cd /gpfs/scratch/hanglu/agent-kv-retention
git fetch origin
git switch week9-cost-calibration
git pull --ff-only origin week9-cost-calibration
~~~

## 2. Install packages (once, on the login node)

~~~bash
cd /gpfs/scratch/hanglu/agent-kv-retention
bash seawulf/setup_week9.sh \
  2>&1 | tee /gpfs/scratch/hanglu/results/week9_setup.log
~~~

This uses SeaWulf's Python module and makes a Python venv under results/venv.
It installs CUDA-enabled PyTorch, Transformers, Accelerate, the Week 4
requirements and the Hugging Face Hub client. It also executes the Week 9
CPU-only unit tests. If module load fails, check "module avail python"
on Milan and use the same Python version in BOTH setup_week9.sh and
week9_a100.slurm.

## 3. Stage the model before submitting the GPU job

~~~bash
bash seawulf/stage_week9_model.sh \
  2>&1 | tee /gpfs/scratch/hanglu/results/week9_model_download.log
~~~

The default is Qwen/Qwen2.5-7B (an open-weight 7B causal LM, Apache-2.0).
Weights and Hugging Face caches are stored within results/. The download may
be large and requires network access; check scratch quota and SeaWulf policies.
If the login node cannot reach Hugging Face, download or copy a compatible
local Hugging Face model directory (with config.json and weights) to
results/models/Qwen2.5-7B before submitting. This script is a staging action;
the Slurm job deliberately runs with offline model loading.

**Scientific caveat:** This model is NOT Llama 3.1 and a result for Qwen must
be labeled as Qwen, not presented as a measured Llama throughput.

## 4. Submit to the A100 GPU partition

~~~bash
module load slurm
sinfo -p a100
cd /gpfs/scratch/hanglu/agent-kv-retention
sbatch seawulf/week9_a100.slurm
squeue -u "$USER"
~~~

The job requests ONE A100 GPU, four CPU threads, 50 GB system RAM and up to
two hours. It benchmarks 1,024 / 2,048 / 4,096 / 8,192-token contexts with
two warmups and five measured repetitions each. It runs CPU unit tests,
records GPU and software details, and computes break-even sensitivity under
both fixed and proportional c modes. The GPU job itself requires no network
connection.

All artifacts go to:

    /gpfs/scratch/hanglu/results/week9_<SLURM_JOB_ID>/

Expected:
- prefill_measurements.csv — observed per-run prefill durations
- prefill_measurements.json — benchmark metadata
- break_even_scenarios.csv — modeled B, c, b sensitivity
- calibration_manifest.json — measurement summary and assumptions
- python_packages.txt, nvidia_smi.txt, git_commit.txt — reproducibility info
- tmp/ — temporary job workspace

Slurm logs are separate but also inside the results directory:

    /gpfs/scratch/hanglu/results/week9_<SLURM_JOB_ID>.out
    /gpfs/scratch/hanglu/results/week9_<SLURM_JOB_ID>.err

## 5. Check progress and results

~~~bash
squeue -u "$USER"
ls -lh /gpfs/scratch/hanglu/results/week9_*
tail -n 50 /gpfs/scratch/hanglu/results/week9_<JOBID>.out
cat /gpfs/scratch/hanglu/results/week9_<JOBID>.err
~~~

Replace <JOBID> with the Slurm job ID. Once finished, inspect
break_even_scenarios.csv and calibration_manifest.json. A problem downloading
a model, denied A100 allocation, unsupported torch/CUDA combination or OOM
must be fixed before results can be interpreted.

## Limitations and next work

Prefill measurements estimate the B component under this specific GPU, model,
dtype and input lengths. They do NOT calibrate c (lost service capacity from
retaining KV memory), queueing under concurrent requests, or isolated KV
memory demand. The 8,192-token maximum is much shorter than the approximately
122K-token median context of the held-out traces, so extrapolation to those
contexts must be identified rather than reported as directly measured.

After collecting data, rerun the matched policies using justified calibrated
parameters and compare both c-mode assumptions. No official 357,161-row
33-column dataset or previous experiment results are modified by this job.
