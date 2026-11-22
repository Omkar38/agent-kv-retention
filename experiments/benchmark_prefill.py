"""Measure prefill throughput on a REAL CUDA GPU with a local/accessible HF model.

Requirements (not installed by the core project): torch, transformers.
Example:
    python3 -m experiments.benchmark_prefill \
      --model-id meta-llama/Meta-Llama-3-8B \
      --context-lengths 1024,2048,4096 --repeats 5 \
      --output results/calibration/prefill_measurements.csv

This benchmark only measures prefill forward passes, not concurrent KV memory
opportunity cost, prefill under serving load, queueing, or end-to-end latency.
Input IDs are synthetic to control sequence length. A real serving benchmark
will ultimately be needed to validate transfer to production workloads.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path


def lengths_arg(text: str) -> list[int]:
    lengths = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not lengths or any(x <= 0 for x in lengths):
        raise ValueError("All context lengths must be positive")
    return lengths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True, help="HF model name or local model folder")
    p.add_argument("--context-lengths", default="1024,2048,4096")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("results/calibration/prefill_measurements.csv"))
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = p.parse_args()
    lens = lengths_arg(args.context_lengths)
    if args.repeats < 1 or args.warmup < 0:
        p.error("repeats must be positive and warmup nonnegative")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        p.error("Install torch and transformers in a dedicated CUDA environment: " + str(exc))
    if not torch.cuda.is_available():
        p.error("A CUDA-enabled GPU is required. No measurements were produced.")

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval().to("cuda")
    device = torch.device("cuda:0")
    vocab = int(model.config.vocab_size)
    max_len = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_len, int) and max_len > 0 and max(lens) > max_len:
        p.error(f"Requested context exceeds model max_position_embeddings={max_len}")

    rows = []
    with torch.inference_mode():
        for n in lens:
            # Random token IDs avoid explicit prompt/private data; sequence lengths
            # are controlled. Forward pass includes prefill; no decoding step.
            input_ids = torch.randint(0, vocab, (1, n), device=device)
            attention_mask = torch.ones_like(input_ids)
            for i in range(args.warmup + args.repeats):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                peak = int(torch.cuda.max_memory_allocated(device))
                del result
                if i < args.warmup:
                    continue
                rows.append({
                    "context_tokens": n,
                    "prefill_time_s": elapsed,
                    "prefill_tokens_per_s": n / elapsed,
                    "cuda_peak_allocated_bytes": peak,
                    "repeat_index": i - args.warmup,
                    "model_id": args.model_id,
                    "dtype": args.dtype,
                    "gpu_name": torch.cuda.get_device_name(device),
                })
                print(f"n={n:>7} repeat={i-args.warmup} {elapsed:.4f}s ({n/elapsed:,.0f} tok/s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        p.error(f"Refusing to overwrite {args.output}; choose a new output path")
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "model_id": args.model_id,
        "gpu_name": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "sequence_lengths": lens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "method": "CUDA synchronized forward pass with use_cache=True on synthetic input",
        "caveat": "Peak allocated VRAM includes model and activations; not measured isolated KV-cache bytes. No queueing or concurrent-request calibration.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {args.output} and provenance JSON.")


if __name__ == "__main__":
    main()
