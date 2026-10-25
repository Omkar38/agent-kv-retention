"""Figures for the midterm report.

Palette and mark specs follow the project's data-visualisation standard:
four categorical hues validated all-pairs for colour-vision deficiency
(blue, orange, aqua, violet), thin marks, recessive grid, direct labels on
every series. Aqua sits at 2.74:1 against the surface, below the 3:1 bar, so
direct labels are mandatory rather than decorative -- they are the relief that
makes the series identifiable without relying on colour.

Run from the repository root:

    python3 -m experiments.make_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results/policy")
FIGURES = Path("results/figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b8b7b1"
GRID = "#e7e6e2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
REFERENCE = "#8a8984"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "grid.color": GRID,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def fig_policy_comparison() -> None:
    """Every policy against the clairvoyant optimum, both cost models."""
    d = pd.read_csv(RESULTS / "single_pause_costs.csv")
    d = d[d.error_mode == "clean"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=False)
    for ax, mode in zip(axes, ["fixed", "proportional"]):
        x = d[d.c_mode == mode].sort_values("total_ratio")
        x = x[x.policy != "clairvoyant"]
        labels = [p.replace("_", " ") for p in x.policy]
        values = x.total_ratio.to_numpy()

        # One measure across categories: one hue. Break-even is the baseline
        # the contribution must clear, so it is called out by annotation and a
        # reference line -- not by a second colour, which would imply identity.
        colors = [SERIES[1] if p == "break_even_ttl" else SERIES[0] for p in x.policy]
        bars = ax.barh(labels, values, color=colors, height=0.68)
        for bar, v in zip(bars, values):
            ax.text(
                v + 0.12, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", ha="left", fontsize=9, color=INK_2,
            )

        ax.axvline(1.0, color=REFERENCE, lw=1.2, ls=":", zorder=0)
        # Sit the reference label below the shortest bar so it never crosses a mark.
        ax.text(
            1.0, -0.95, " clairvoyant", fontsize=8.5, color=INK_2,
            va="center", ha="left",
        )
        ax.set_xlim(0, max(values) * 1.22)
        ax.set_ylim(-1.4, len(labels) - 0.4)
        ax.set_xlabel("total cost ÷ clairvoyant optimum")
        ax.set_title(f"c {mode}", loc="left")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

    fig.suptitle(
        "No prediction-aware policy beats the break-even rule",
        x=0.012, ha="left", fontsize=12.5, fontweight="semibold",
    )
    fig.text(
        0.012, 0.005,
        "Held-out TraceLab pauses (47,773 over 384 sessions), q50 signal. "
        "Orange marks the prediction-free break-even baseline.",
        fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.93])
    fig.savefig(FIGURES / "fig1_policy_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_lambda_by_signal() -> None:
    """The trust dial, driven by two different quantiles of the same model."""
    d = pd.read_csv(RESULTS / "signal_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=False)
    for ax, mode in zip(axes, ["fixed", "proportional"]):
        x = d[d.c_mode == mode]
        baseline = float(x[x.policy == "break_even_ttl"].total_ratio.iloc[0])

        for i, (signal, label) in enumerate([("hgb_q50", "q50  (Week 4 choice)"), ("hgb_q90", "q90")]):
            s = x[(x.signal == signal) & x.policy.str.startswith("lambda_")].copy()
            s["lam"] = s.policy.str.replace("lambda_", "").astype(float)
            s = s.sort_values("lam")
            ax.plot(s.lam, s.total_ratio, color=SERIES[i], lw=2, marker="o", ms=5, label=label)
            # Direct label: identity must not depend on colour alone.
            ax.annotate(
                label.split()[0], (s.lam.iloc[0], s.total_ratio.iloc[0]),
                textcoords="offset points", xytext=(-6, 6), ha="right",
                fontsize=9, color=SERIES[i], fontweight="semibold",
            )

        ax.axhline(baseline, color=REFERENCE, lw=1.2, ls="--", zorder=0)
        # Park the reference label mid-axis, clear of both series.
        ax.text(
            0.62, baseline, "break-even", fontsize=8.5, color=INK_2,
            va="bottom", ha="center",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
        )
        ax.set_xlabel("λ        (smaller = more trust in the prediction)")
        ax.set_ylabel("total cost ÷ optimum")
        ax.set_title(f"c {mode}", loc="left")
        ax.invert_xaxis()

    fig.suptitle(
        "Trusting the prediction pays only under proportional c, and only with the recall-oriented quantile",
        x=0.012, ha="left", fontsize=12, fontweight="semibold",
    )
    fig.text(
        0.012, 0.005,
        "Anything above the dashed line is worse than ignoring the prediction entirely.",
        fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.92])
    fig.savefig(FIGURES / "fig2_lambda_by_signal.png", bbox_inches="tight")
    plt.close(fig)


def fig_decomposition() -> None:
    """Where the lambda-policy gains and where it loses."""
    d = pd.read_csv(RESULTS / "decomposition.csv")
    d = d[(d.c_mode == "fixed") & (d.signal == "hgb_q50")]
    order = ["correct_short", "over_called", "correct_long", "under_called"]
    d = d.set_index("cell").loc[order].reset_index()

    pretty = {
        "correct_short": "correct\nshort",
        "over_called": "over-called\n(short→long)",
        "correct_long": "correct\nlong",
        "under_called": "under-called\n(long→short)",
    }
    series = [
        ("opt_cost", "clairvoyant", SERIES[0]),
        ("break_even_cost", "break-even", SERIES[1]),
        ("lambda_0.5_cost", "λ = 0.5", SERIES[2]),
    ]

    # The headline ratio is computed, not asserted: how much the policy loses
    # on mis-called pauses for every unit it gains on correctly-called ones.
    by_cell = d.set_index("cell")
    gain = float(by_cell.loc["correct_long", "break_even_cost"] - by_cell.loc["correct_long", "lambda_0.5_cost"])
    loss = float(
        sum(
            by_cell.loc[c, "lambda_0.5_cost"] - by_cell.loc[c, "break_even_cost"]
            for c in ("under_called", "over_called")
        )
    )
    ratio = loss / gain if gain > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(d))
    width = 0.25
    gap = 0.03  # surface gap between adjacent bars
    peak = max(d[c].max() for c, _, _ in series) / 1000.0
    for i, (col, label, color) in enumerate(series):
        offset = (i - 1) * (width + gap)
        vals = d[col].to_numpy() / 1000.0
        bars = ax.bar(x + offset, vals, width, color=color, label=label)
        # Aqua is below 3:1 on this surface, so every bar carries its value.
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + peak * 0.018, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8, color=INK_2,
            )

    ax.set_xticks(x)
    # Count and share live in the tick label so nothing can collide with it.
    ax.set_xticklabels(
        [
            f"{pretty[row.cell]}\nn = {int(row.n):,}  ·  {row.share * 100:.1f}%"
            for _, row in d.iterrows()
        ]
    )
    ax.set_ylabel("total cost  (thousand GPU-seconds)")
    # Headroom so the legend never lands on a mark.
    ax.set_ylim(0, peak * 1.30)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    ax.set_title(
        f"λ gains on correctly-called long pauses and loses {ratio:.1f}× that elsewhere",
        loc="left",
    )
    fig.text(
        0.012, 0.005,
        "c fixed, q50 signal. Under-called long pauses outnumber correct ones 3.5:1, so no λ can win.",
        fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    fig.savefig(FIGURES / "fig3_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def fig_simulator() -> None:
    """Job completion time against offered load, once sessions compete.

    Budgets below 600k tokens are excluded: there, roughly half of admissions
    have a context larger than the whole pool, so those runs measure an
    undersized pool rather than a retention policy.
    """
    path = RESULTS / "scarce.csv"
    if not path.exists():
        print("  (skipping fig4: scarce.csv not present)")
        return
    d = pd.read_csv(path)

    # Two measures of very different scale -- an 8x gap and a 1-2% ordering --
    # so they get two panels rather than one log axis that hides the smaller.
    d = d[d.budget_tokens == 600_000]
    loads = sorted(d.replicas.unique())

    def series(policy: str, signal: str = "none") -> pd.DataFrame:
        return d[(d.policy == policy) & (d.signal == signal)].sort_values("replicas")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.2, 4.6))

    # -- Panel A: the gap that matters
    for policy, label, color in [
        ("immediate_evict", "immediate evict", SERIES[1]),
        ("always_retain_lru", "any retention policy", SERIES[0]),
    ]:
        s = series(policy)
        ax_a.plot(s.replicas, s.mean_jct_s / 60.0, color=color, lw=2, marker="o", ms=6)
        ax_a.annotate(
            label, (s.replicas.iloc[-1], s.mean_jct_s.iloc[-1] / 60.0),
            textcoords="offset points", xytext=(-8, 14 if policy == "immediate_evict" else -18),
            ha="right", fontsize=9.5, color=color, fontweight="semibold",
        )
    ratio = float(series("immediate_evict").mean_jct_s.iloc[-1]) / float(
        series("always_retain_lru").mean_jct_s.iloc[-1]
    )
    ax_a.set_xlabel("offered load  (× trace concurrency)")
    ax_a.set_ylabel("mean job completion time  (minutes)")
    ax_a.set_title(f"Evicting on decode-end costs {ratio:.1f}×", loc="left")
    ax_a.set_xticks(loads)
    ax_a.margins(x=0.18, y=0.22)

    # -- Panel B: the ordering among retention policies, as a ratio so it is visible
    baseline = series("always_retain_lru").set_index("replicas").mean_jct_s
    contenders = [
        ("break_even_ttl", "none", "break-even TTL", SERIES[0]),
        ("lambda_0.5", "hgb_q90", "λ = 0.5 (q90)", SERIES[2]),
        ("clairvoyant_per_pause", "none", "clairvoyant (per-pause)", SERIES[3]),
        ("continuum_style", "none", "Continuum-style", SERIES[1]),
    ]
    # Several policies land within ~2% of each other, so end labels would
    # overlap. Draw first, then lay the labels out with a minimum vertical
    # separation, keeping their original order.
    ends = []
    for policy, signal, label, color in contenders:
        s = series(policy, signal)
        rel = (s.set_index("replicas").mean_jct_s / baseline) * 100.0
        ax_b.plot(rel.index, rel.to_numpy(), color=color, lw=2, marker="o", ms=5)
        ends.append([float(rel.to_numpy()[-1]), label, color])

    ends.sort(key=lambda e: e[0])
    span = ax_b.get_ylim()[1] - ax_b.get_ylim()[0]
    min_gap = span * 0.062
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < min_gap:
            ends[i][0] = ends[i - 1][0] + min_gap
    for y, label, color in ends:
        ax_b.annotate(
            label, (loads[-1], y),
            textcoords="offset points", xytext=(9, 0), ha="left", va="center",
            fontsize=8.5, color=color, fontweight="semibold",
        )

    ax_b.axhline(100.0, color=REFERENCE, lw=1.2, ls="--", zorder=0)
    ax_b.text(
        loads[0], 100.0, "always retain + LRU", fontsize=8.5, color=INK_2,
        va="top", ha="left",
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
    )
    ax_b.set_xlabel("offered load  (× trace concurrency)")
    ax_b.set_ylabel("JCT relative to always-retain  (%)")
    ax_b.set_title("Among retention policies, no one beats just holding on", loc="left")
    ax_b.set_xticks(loads)
    ax_b.margins(x=0.42, y=0.18)

    fig.suptitle(
        "The large win is not evicting on decode-end; choosing among retention policies is second-order",
        x=0.012, ha="left", fontsize=12.5, fontweight="semibold",
    )
    fig.text(
        0.012, 0.005,
        "600k-token pool, c fixed. Load above 1× replicates the session population, so it describes a "
        "hypothetical server at that concurrency rather than more evidence about the trace.",
        fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    fig.savefig(FIGURES / "fig4_simulator_load.png", bbox_inches="tight")
    plt.close(fig)


def fig_degradation() -> None:
    """What happens as the prediction is deliberately made worse."""
    d = pd.read_csv(RESULTS / "single_pause_costs.csv")
    d = d[d.c_mode == "fixed"]
    baseline = float(
        d[(d.policy == "break_even_ttl") & (d.error_mode == "clean")].total_ratio.iloc[0]
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    panels = [
        ("noise", "multiplicative noise  (dex)", [0.25, 0.5, 1.0, 2.0]),
        ("under", "systematic under-prediction  (α)", [0.8, 0.5, 0.25, 0.1]),
        ("adversarial", "adversarial flips  (fraction)", [0.05, 0.1, 0.25, 0.5, 1.0]),
    ]
    lams = [0.25, 0.5, 0.75]

    for ax, (mode, xlabel, severities) in zip(axes, panels):
        for i, lam in enumerate(lams):
            s = d[(d.error_mode == mode) & (d.policy == f"lambda_{lam:g}")]
            s = s[s.severity.isin(severities)].sort_values("severity")
            ax.plot(
                s.severity, s.total_ratio, color=SERIES[i], lw=2,
                marker="o", ms=5, label=f"λ = {lam}",
            )
            ax.annotate(
                f"λ={lam}", (s.severity.iloc[-1], s.total_ratio.iloc[-1]),
                textcoords="offset points", xytext=(5, 0), ha="left",
                fontsize=8.5, color=SERIES[i], fontweight="semibold",
            )
        ax.axhline(baseline, color=REFERENCE, lw=1.2, ls="--", zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_title(mode, loc="left")
        if mode == "under":
            ax.invert_xaxis()
    axes[0].set_ylabel("total cost ÷ optimum")
    # Below the line and boxed, so it cannot be mistaken for a series label.
    axes[0].text(
        1.1, baseline, "break-even", fontsize=8.5, color=INK_2,
        va="top", ha="center",
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
    )

    fig.suptitle(
        "Degradation under deliberate prediction failure, against the prediction-free baseline",
        x=0.012, ha="left", fontsize=12.5, fontweight="semibold",
    )
    fig.text(
        0.012, 0.005,
        "c fixed. 'under' and 'adversarial' are derived from the true durations, so they are a perfect "
        "predictor made wrong, not our predictor degraded.",
        fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    fig.savefig(FIGURES / "fig5_degradation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    for fn in (
        fig_policy_comparison,
        fig_lambda_by_signal,
        fig_decomposition,
        fig_simulator,
        fig_degradation,
    ):
        print(f"  {fn.__name__}")
        fn()
    print(f"\nwrote figures to {FIGURES}/")


if __name__ == "__main__":
    main()
