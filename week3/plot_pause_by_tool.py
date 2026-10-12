"""Step 4, plot 2: how much does the tool name alone tell you about pause length?

Reads our_work/pauses.parquet and writes our_work/fig_pause_by_tool.png.
"""
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "our_work/pauses.parquet"
OUT = "our_work/fig_pause_by_tool.png"
BREAK_EVEN = 10            # break-even time used for the keep/evict check, in seconds
MIN_PAUSES = 450  # only show tools with at least this many pauses

# ---- Per-tool summary (tool pauses only; human pauses are already separable) ----
per_tool = duckdb.sql(f"""
    SELECT provider, tool_names, count(*) AS n,
           quantile_cont(pause_s, 0.1) AS p10,
           median(pause_s)             AS p50,
           quantile_cont(pause_s, 0.9) AS p90,
           avg((pause_s > {BREAK_EVEN})::INT)   AS share_long
    FROM '{P}'
    WHERE usable AND cause = 'tool'
    GROUP BY ALL
    HAVING count(*) >= {MIN_PAUSES}
    ORDER BY p50
""").fetchall()

print(f"{'tool':<34}{'pauses':>8}{'median':>9}{'p90':>8}{f'> {BREAK_EVEN} s':>8}")
for provider, tools, n, p10, p50, p90, share in per_tool:
    label = f"{tools.replace(',', ' + ')} ({provider})"
    print(f"{label:<34}{n:>8,}{p50:>8.2f}s{p90:>7.1f}s{share:>8.0%}")

# ---- Keep/evict check: decide using ONLY the tool name ----
# Learn each tool's usual answer on 80% of sessions, test on the other 20%.
check = duckdb.sql(f"""
    WITH d AS (
        SELECT provider, tool_names, pause_s > {BREAK_EVEN} AS long,
               md5_number(session_id) % 5 = 0 AS is_test
        FROM '{P}' WHERE usable AND cause = 'tool'),
    rule AS (
        SELECT provider, tool_names, avg(long::INT) > 0.5 AS predict_long
        FROM d WHERE NOT is_test GROUP BY ALL),
    t AS (
        SELECT d.long, coalesce(r.predict_long, false) AS pred
        FROM d LEFT JOIN rule r USING (provider, tool_names) WHERE d.is_test)
    SELECT avg((NOT long)::INT), avg((long = pred)::INT),
           sum((long AND pred)::INT) / sum(long::INT)
    FROM t
""").fetchone()
print(f"\nKeep/evict at b = {BREAK_EVEN} s on held-out sessions:")
print(f"  always keep:          {check[0]:.1%} correct")
print(f"  decide by tool name:  {check[1]:.1%} correct")
print(f"  long pauses caught by tool name: {check[2]:.1%}")

# ---- Figure: median (dot) and 10th-90th percentile range (bar) per tool ----
fig, ax = plt.subplots(figsize=(9, 0.32 * len(per_tool) + 1.4), dpi=150)
for i, (provider, tools, n, p10, p50, p90, share) in enumerate(per_tool):
    lo, mid, hi = max(p10, 0.01), max(p50, 0.01), max(p90, 0.01)   # log axis can't show 0
    ax.plot([lo, hi], [i, i], color="#86b6ef", linewidth=4, solid_capstyle="round")
    ax.plot(mid, i, "o", color="#1c5cab", markersize=6)
    ax.text(2.6e4, i, f"{share:.0%}", va="center", ha="right", color="#5f5e5a", fontsize=9)

ax.axvline(BREAK_EVEN, color="#5f5e5a", linewidth=1, linestyle="--", zorder=1.5)
ax.text(BREAK_EVEN * 1.15, len(per_tool) - 0.3, f"b = {BREAK_EVEN} s", ha="left", va="bottom",
        color="#5f5e5a", fontsize=9)
ax.text(2.6e4, len(per_tool) - 0.3, f"> {BREAK_EVEN} s", ha="right", va="bottom",
        color="#5f5e5a", fontsize=9)

ax.set_yticks(range(len(per_tool)),
              [f"{t.replace(',', ' + ')} · {p.capitalize()}" for p, t, *_ in per_tool])
ticks = [0.01, 0.1, 1, 10, 60, 600, 3600]
ax.set_xscale("log")
ax.set_xticks(ticks, ["0.01 s", "0.1 s", "1 s", "10 s", "1 min", "10 min", "1 h"])
ax.set_xlim(0.01, 3e4)
ax.set_ylim(-0.7, len(per_tool) + 0.3)
ax.set_xlabel("Pause length (log scale) — dot: median, bar: middle 80% of pauses")
ax.set_title("Pause length by tool", loc="left", fontsize=13)
ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#9a9a96")
ax.tick_params(colors="#5f5e5a", length=0)

fig.tight_layout()
fig.savefig(OUT)
print(f"\nWrote {OUT}")
