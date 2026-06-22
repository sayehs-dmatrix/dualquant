"""Plot DualQuant PPL vs. iteration count for different methods/models.

Edit the DATA block below to fill in your numbers, then run:
    python plot_dq_iterations.py

One panel per model; one curve per method.
Output: results_dq_iterations/dq_iterations_plot.pdf  (+.png)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "results_dq_iterations")

# ═══════════════════════════════════════════════════════════════════════════════
#  FILL IN YOUR NUMBERS HERE
# ═══════════════════════════════════════════════════════════════════════════════

# x-axis: iteration numbers you ran
ITERATIONS = [1, 5, 10, 15]

# For each model, list your PPL values in the same order as ITERATIONS above.
# _baselines: optional flat reference lines (single scalar per label).
# Comment out / delete any row you don't have yet.

DATA = {
    "Qwen-2.5-7B": {
        # "DualQuant INT4 (W4A4)":  [6.03, 6.01, 6.01, 0.0],  # ← PPL at iter 1, 5, 10, 15
        # "DualQuant MXFP4 (W4A4)": [6.26, 6.23, 6.21, 0.0],  # ← PPL at iter 1, 5, 10, 15
        "DualQuant INT4 (W4A4)":  [6.47, 6.46, 6.45, 6.46],  # ← PPL at iter 1, 5, 10, 15
        "DualQuant MXFP4 (W4A4)": [6.64, 6.60, 6.60, 6.62],  # ← PPL at iter 1, 5, 10, 15
        "_baselines": {
            "RTN INT4 (no DQ)":  0.0,  # ← single scalar, drawn as horizontal dashed line
            "RTN MXFP4 (no DQ)": 0.0,  # ← single scalar, drawn as horizontal dashed line
        },
    },
    # ── copy this block for each additional model ──────────────────────────────
    # "Qwen2.5-7B": {
    #     "DualQuant INT4 (W4A4)":  [0.0, 0.0, 0.0, 0.0],
    #     "DualQuant MXFP4 (W4A4)": [0.0, 0.0, 0.0, 0.0],
    #     "_baselines": {
    #         "RTN INT4 (no DQ)":  0.0,
    #         "RTN MXFP4 (no DQ)": 0.0,
    #     },
    # },
}

# ═══════════════════════════════════════════════════════════════════════════════

# ── STYLE ─────────────────────────────────────────────────────────────────────
_CURVE_STYLE = {
    "DualQuant INT4 (W4A4)":  dict(color="#d7191c", marker="s", linestyle="-"),
    "DualQuant MXFP4 (W4A4)": dict(color="#f97c0a", marker="D", linestyle="-"),
}
_BASELINE_COLORS = ["#2c7bb6", "#1a9641", "#d7191c", "#f97c0a", "#7b2d8b"]
_DEFAULT_STYLE   = dict(color="gray", marker="x", linestyle="-")


def _plot_panel(ax, model_name, model_data):
    baselines = model_data.get("_baselines", {})
    curves    = {k: v for k, v in model_data.items() if k != "_baselines"}

    # flat baselines
    for i, (label, val) in enumerate(baselines.items()):
        if val == 0.0:
            continue
        color = _BASELINE_COLORS[i % len(_BASELINE_COLORS)]
        ax.axhline(val, linestyle="--", linewidth=1.2, color=color,
                   alpha=0.7, label=label, zorder=1)

    # iteration curves
    for label, ppls in curves.items():
        if not ppls or all(v == 0.0 for v in ppls):
            continue
        iters = ITERATIONS[:len(ppls)]
        style = _CURVE_STYLE.get(label, _DEFAULT_STYLE)
        ax.plot(iters, ppls,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=1.8, marker=style["marker"], markersize=5.5,
                markerfacecolor="white", markeredgecolor=style["color"],
                markeredgewidth=1.3, label=label, zorder=2)

        # star at minimum
        best_idx = int(np.argmin(ppls))
        ax.plot(iters[best_idx], ppls[best_idx], marker="*", markersize=9,
                color=style["color"], zorder=4, linestyle="none")

    ax.set_title(model_name, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("DualQuant iterations", fontsize=8.5)
    ax.set_ylabel("WikiText-2 PPL", fontsize=8.5)
    ax.set_xticks(ITERATIONS)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=7, framealpha=0.85, loc="upper right",
              handlelength=1.6, labelspacing=0.3)


def main():
    models = list(DATA.keys())
    ncols  = min(3, len(models))
    nrows  = (len(models) + ncols - 1) // ncols
    cell   = 4.2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell * ncols, cell * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(models):
        _plot_panel(axes_flat[idx], model_name, DATA[model_name])

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"dq_iterations_plot.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
