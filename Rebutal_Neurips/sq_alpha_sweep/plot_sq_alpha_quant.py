"""Plot SmoothQuant alpha sweep with quantization — one panel per model.

Reads results_sq_alpha_sweep/sq_alpha_quant_sweep.csv and produces a 2×3 grid.
Each panel shows 4 lines (one per quantization scenario). The minimum-PPL point
per line is marked with a ★ and annotated with its alpha value.

Output: results_sq_alpha_sweep/sq_alpha_quant_plot.pdf  (+.png)

Usage:
    python plot_sq_alpha_quant.py
    python plot_sq_alpha_quant.py --csv path/to/other.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results_sq_alpha_sweep")
OUT_DIR      = RESULTS_DIR

_DISPLAY = {
    "meta-llama/Llama-3.1-8B":  "Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B":  "Llama-3.2-1B",
    "meta-llama/Llama-2-7b-hf": "Llama-2-7B",
    "mistralai/Mistral-7B-v0.3":"Mistral-7B",
    "Qwen/Qwen3-0.6B":          "Qwen3-0.6B",
    "Qwen/Qwen2.5-7B":          "Qwen2.5-7B",
    "Qwen/Qwen3-14B":           "Qwen3-14B",
}

# One colour + marker per scenario — ordered to match sweep script
_SCENARIO_STYLE = {
    "SQ+INT4 (W4A16)":  dict(color="#2c7bb6", marker="o",  linestyle="-",  label="SQ + INT4  (W4A16)"),
    "SQ+INT4 (W4A4)":   dict(color="#d7191c", marker="s",  linestyle="--", label="SQ + INT4  (W4A4)"),
    "SQ+MXFP4 (W4A16)": dict(color="#1a9641", marker="^",  linestyle="-",  label="SQ + MXFP4 (W4A16)"),
    "SQ+MXFP4 (W4A4)":  dict(color="#f97c0a", marker="D",  linestyle="--", label="SQ + MXFP4 (W4A4)"),
}


_OPTIMAL_THRESHOLD = 0.01   # shade between curve and PPL_min × (1 + this)


def _plot_panel(ax, sub_model, title, min_alpha=0.0):
    """Draw all scenario curves for one model onto ax."""
    scenarios = sub_model["scenario"].unique()

    for scen_label in scenarios:
        style = _SCENARIO_STYLE.get(scen_label, dict(color="gray", marker="o", linestyle="-", label=scen_label))
        sub = sub_model[
            (sub_model["scenario"] == scen_label) & (sub_model["alpha"] >= min_alpha)
        ].sort_values("alpha")
        alphas = sub["alpha"].values
        ppls   = sub["ppl_wikitext"].values

        best_ppl      = ppls.min()
        threshold_ppl = best_ppl * (1 + _OPTIMAL_THRESHOLD)
        in_band       = ppls <= threshold_ppl

        # shaded fill between curve and threshold line, clipped to the discrete
        # alpha range [first qualifying, last qualifying] — no interpolated edges
        idx = np.where(in_band)[0]
        if len(idx):
            sl = slice(idx[0], idx[-1] + 1)
            ax.fill_between(alphas[sl], ppls[sl], threshold_ppl,
                            color=style["color"], alpha=0.18,
                            linewidth=0, zorder=1)

        # line (drawn on top of the fill)
        ax.plot(alphas, ppls,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=1.8, marker=style["marker"], markersize=4.5,
                markerfacecolor="white", markeredgecolor=style["color"],
                markeredgewidth=1.3, label=style["label"], zorder=2)

        # star at minimum PPL
        best_idx = ppls.argmin()
        ax.plot(alphas[best_idx], ppls[best_idx], marker="*", markersize=8,
                color=style["color"], zorder=4, linestyle="none")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Migration strength α", fontsize=8.5)
    ax.set_ylabel("WikiText-2 PPL", fontsize=8.5)
    ax.set_xticks([round(a * 0.1, 1) for a in range(11)])  # always 0.0 … 1.0
    ax.tick_params(labelsize=8)

    visible = sub_model[sub_model["alpha"] >= min_alpha]["ppl_wikitext"].values
    y_min = visible.min()
    y_max = visible.max()
    pad = max((y_max - y_min) * 0.15, 0.05)
    ax.set_ylim(y_min - pad, y_max + pad * 2)
    ax.set_xlim(-0.03, 1.03)  # always show 0 on the axis even if no data there

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)

    # Legend inside the panel (upper right), no shared legend below
    ax.legend(fontsize=7, framealpha=0.85, loc="upper right",
              handlelength=1.6, labelspacing=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=RESULTS_DIR,
                   help="Folder containing per-model CSVs (default: results_sq_alpha_sweep/)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model ids to plot (default: all CSVs found in results-dir)")
    p.add_argument("--min-alpha", type=float, default=0.1,
                   help="Smallest alpha to show on x-axis (default: 0.1; use 0.0 to include α=0)")
    args = p.parse_args()

    # Collect all per-model CSVs
    csv_files = sorted(f for f in os.listdir(args.results_dir) if f.endswith(".csv"))
    frames = []
    for fname in csv_files:
        frames.append(pd.read_csv(os.path.join(args.results_dir, fname)))
    if not frames:
        raise SystemExit(f"No CSV files found in {args.results_dir}")
    df = pd.concat(frames, ignore_index=True)

    models = df["model"].unique().tolist()
    if args.models:
        models = [m for m in models if m in args.models]
    ncols, nrows = 3, (len(models) + 2) // 3
    cell = 4.2  # square-ish cell size in inches
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows))
    axes = np.array(axes).flatten()

    for idx, model_id in enumerate(models):
        title = _DISPLAY.get(model_id, model_id.split("/")[-1])
        _plot_panel(axes[idx], df[df["model"] == model_id], title, min_alpha=args.min_alpha)

    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"sq_alpha_quant_plot.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
