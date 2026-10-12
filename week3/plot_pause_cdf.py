"""Step 4, plot 1: how long do pauses last, and how many beat the break-even time?

Reads our_work/pauses.parquet and writes our_work/fig_pause_cdf.png.
"""
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = "our_work/pauses.parquet"
OUT = "our_work/fig_pause_cdf.png"
BREAK_EVENS = [1, 5, 10, 30, 60]          # candidate break-even times b, in seconds

# Usable pauses only. "user" here means a human was involved (user or tool+user).
rows = duckdb.sql(f"""
    SELECT CASE WHEN cause = 'tool' THEN 'tool' ELSE 'user' END AS kind, pause_s
    FROM '{P}' WHERE usable
""").fetchnumpy()
kind, pause = rows["kind"], rows["pause_s"]

groups = {
    "Tool pauses": pause[kind == "tool"],
    "Pauses involving a human": pause[kind == "user"],
}
colors = {"Tool pauses": "#2a78d6", "Pauses involving a human": "#eb6834"}

# ---- Table: share of pauses shorter than each break-even time ----
print(f"{'b (seconds)':>12} | {'tool':>6} | {'human':>6} | {'all':>6}")
for b in BREAK_EVENS:
    tool_share = np.mean(groups["Tool pauses"] < b)
    human_share = np.mean(groups["Pauses involving a human"] < b)
    all_share = np.mean(pause < b)
    print(f"{b:>12} | {tool_share:6.1%} | {human_share:6.1%} | {all_share:6.1%}")
print(f"\n{len(pause):,} usable pauses "
      f"({len(groups['Tool pauses']):,} tool, {len(groups['Pauses involving a human']):,} human)")

# ---- Figure: cumulative share of pauses shorter than x (log time axis) ----
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
for name, values in groups.items():
    x = np.sort(np.clip(values, 0.01, None))   # log axis can't show 0
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color=colors[name], linewidth=2, label=name)

# Candidate break-even times, drawn quietly so the curves stay the focus
for b in (1, 10, 60):
    ax.axvline(b, color="#9a9a96", linewidth=1, linestyle="--", zorder=0)
    ax.text(b, 1.01, f"b = {b} s", color="#5f5e5a", fontsize=9, ha="center",
            va="bottom", transform=ax.get_xaxis_transform())

# Direct labels beside each curve
ax.text(20, 0.76, "Tool pauses", color="#1c5cab", fontsize=10)
ax.text(2500, 0.62, "Pauses involving\na human", color="#c24f1f", fontsize=10)

ticks = [0.01, 0.1, 1, 10, 60, 600, 3600, 86400]
labels = ["0.01 s", "0.1 s", "1 s", "10 s", "1 min", "10 min", "1 h", "1 day"]
ax.set_xscale("log")
ax.set_xticks(ticks, labels)
ax.set_xlim(0.01, 86400 * 3)
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Pause length (log scale)")
ax.set_ylabel("Share of pauses shorter than this")
ax.set_title("How long the KV cache sits idle during a pause", loc="left", fontsize=13, pad=22)
ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
ax.legend(loc="upper left", frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#9a9a96")
ax.tick_params(colors="#5f5e5a")

fig.tight_layout()
fig.savefig(OUT)
print(f"Wrote {OUT}")
