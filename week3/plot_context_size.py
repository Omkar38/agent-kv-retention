"""Plot 3: how big are the idle caches, and what does that make b?

On the formula sheet, the break-even time is b = prefill time + queueing delay.
Prefill time grows with context length, so every pause gets its own b.
We don't have GPU measurements yet, so prefill speed is a PLACEHOLDER
(tokens per second); M5's vLLM numbers should replace it later.
Queueing delay is set to 0 here (an idle server).

Reads our_work/pauses.parquet and writes our_work/fig_context_size.png.
"""
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = "our_work/pauses.parquet"
OUT = "our_work/fig_context_size.png"
PREFILL_SPEEDS = [5_000, 10_000, 20_000]   # placeholder prefill speeds, tokens/second
FIG_SPEED = 10_000                         # the one drawn on the figure's top axis

# ---- Table 1: how big is the cache that sits idle? ----
print("Context size at the start of a pause (thousands of tokens):")
duckdb.sql(f"""
    SELECT provider,
           CASE WHEN cause = 'tool' THEN 'tool' ELSE 'human' END AS kind,
           count(*) AS pauses,
           round(median(context_tokens) / 1000) AS median_k,
           round(quantile_cont(context_tokens, 0.9) / 1000) AS p90_k
    FROM '{P}' WHERE usable
    GROUP BY ALL ORDER BY provider, kind DESC
""").show()

# ---- Table 2: give every pause its own b, then ask who should keep ----
print("If b = context_tokens / prefill speed (queueing delay = 0):")
print(f"{'prefill speed':>15} | {'median b':>9} | {'keep wins: tool':>15} | {'human':>6} | {'all':>6}")
for speed in PREFILL_SPEEDS:
    median_b, tool, human, everyone = duckdb.sql(f"""
        SELECT median(context_tokens / {speed}),
               avg((pause_s < context_tokens / {speed})::INT) FILTER (WHERE cause = 'tool'),
               avg((pause_s < context_tokens / {speed})::INT) FILTER (WHERE cause <> 'tool'),
               avg((pause_s < context_tokens / {speed})::INT)
        FROM '{P}' WHERE usable
    """).fetchone()
    print(f"{speed:>9,} tok/s | {median_b:>7.1f} s | {tool:>15.1%} | {human:>6.1%} | {everyone:>6.1%}")

# ---- Figure: share of pauses whose cache is smaller than x tokens ----
rows = duckdb.sql(f"""
    SELECT CASE WHEN cause = 'tool' THEN 'tool' ELSE 'user' END AS kind, context_tokens
    FROM '{P}' WHERE usable AND context_tokens > 0
""").fetchnumpy()
groups = {
    "Tool pauses": rows["context_tokens"][rows["kind"] == "tool"],
    "Pauses involving a human": rows["context_tokens"][rows["kind"] == "user"],
}
colors = {"Tool pauses": "#2a78d6", "Pauses involving a human": "#eb6834"}  # same as plot 1

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
for name, values in groups.items():
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color=colors[name], linewidth=2, label=name)

ax.set_xscale("log")
ax.set_xlim(1_000, 1_100_000)
ax.set_ylim(0, 1.02)
ax.set_xticks([1_000, 10_000, 100_000, 1_000_000], ["1K", "10K", "100K", "1M"])
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Context size at the start of the pause (tokens, log scale)")
ax.set_ylabel("Share of pauses with a smaller cache")

# Same axis, re-labelled in seconds: how long rebuilding the cache would take
top = ax.secondary_xaxis("top", functions=(lambda t: t / FIG_SPEED, lambda s: s * FIG_SPEED))
top.set_xticks([0.1, 1, 10, 100], ["0.1 s", "1 s", "10 s", "100 s"])
top.set_xlabel(f"Break-even time b if prefill runs at {FIG_SPEED:,} tokens/s (placeholder)",
               color="#5f5e5a")
top.tick_params(colors="#5f5e5a")

ax.text(100_000, 0.62, "Tool pauses", color="#1c5cab", fontsize=10, ha="right")
ax.text(230_000, 0.42, "Pauses involving\na human", color="#c24f1f", fontsize=10)
ax.set_title("How big is the cache that sits idle?", loc="left", fontsize=13, pad=12)
ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False)
ax.spines["right"].set_visible(False)
for spine in (ax.spines["left"], ax.spines["bottom"], ax.spines["top"], top.spines["top"]):
    spine.set_color("#9a9a96")
ax.tick_params(colors="#5f5e5a")

fig.tight_layout()
fig.savefig(OUT)
print(f"\nWrote {OUT}")