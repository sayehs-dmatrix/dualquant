"""Plot DualQuant PPL vs. iteration for weight-only quantization.

Reads all CSVs from results_dualQuant_with_Iter_weight_only/.
Filename pattern: {model_id}_iter{N}.csv
Each CSV has columns: model, weight_fmt, ppl_wikitext (and others).

One panel per model (2×3 grid), one curve per weight format.
Output: results_dualQuant_with_Iter_weight_only/dq_iter_weight_only.pdf (+.png)
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "results_dualQuant_with_Iter_weight_only")
OUT_DIR     = DATA_DIR

_DISPLAY = {
    "meta-llama/Llama-3.1-8B":  "Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B":  "Llama-3.2-1B",
    "meta-llama/Llama-2-7b-hf": "Llama-2-7B",
    "Qwen/Qwen3-0.6B":          "Qwen3-0.6B",
    "Qwen/Qwen2.5-7B":          "Qwen2.5-7B",
    "Qwen/Qwen3-14B":           "Qwen3-14B",
}

_FMT_STYLE = {
    "rtn_int4": dict(color="#2c7bb6", marker="o", linestyle="-",  label="RTN INT4"),
    "mxfp4":    dict(color="#1a9641", marker="^", linestyle="-",  label="MXFP4"),
    "mxint4":   dict(color="#f97c0a", marker="D", linestyle="-",  label="MXINT4"),
    "nvfp4":    dict(color="#d7191c", marker="s", linestyle="--", label="NVFP4"),
}
_DEFAULT_STYLE = dict(color="gray", marker="x", linestyle="-")


def load_data(data_dir):
    """Return DataFrame with columns: model, weight_fmt, iter, ppl_wikitext."""
    pattern = re.compile(r"^(.+)_iter(\d+)\.csv$")
    rows = []
    for fname in os.listdir(data_dir):
        m = pattern.match(fname)
        if not m:
            continue
        iteration = int(m.group(2))
        df = pd.read_csv(os.path.join(data_dir, fname))
        df["iter"] = iteration
        rows.append(df)
    if not rows:
        raise SystemExit(f"No *_iter*.csv files found in {data_dir}")
    return pd.concat(rows, ignore_index=True)


def _plot_panel(ax, sub_model, title):
    fmts = sorted(sub_model["weight_fmt"].unique())
    for fmt in fmts:
        style = _FMT_STYLE.get(fmt, {**_DEFAULT_STYLE, "label": fmt})
        sub = sub_model[sub_model["weight_fmt"] == fmt].sort_values("iter")
        iters = sub["iter"].values
        ppls  = sub["ppl_wikitext"].values

        ax.plot(iters, ppls,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=1.8, marker=style["marker"], markersize=5.5,
                markerfacecolor="white", markeredgecolor=style["color"],
                markeredgewidth=1.3, label=style["label"], zorder=2)

        best_idx = int(np.argmin(ppls))
        ax.plot(iters[best_idx], ppls[best_idx], marker="*", markersize=8,
                color=style["color"], zorder=4, linestyle="none")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("DualQuant iterations", fontsize=8.5)
    ax.set_ylabel("WikiText-2 PPL", fontsize=8.5)
    all_iters = sorted(sub_model["iter"].unique())
    ax.set_xticks(all_iters)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=7, framealpha=0.85, loc="upper right",
              handlelength=1.6, labelspacing=0.3)


def main():
    df = load_data(DATA_DIR)

    # normalise model id (CSV stores "meta-llama/Llama-3.1-8B", filename has "_")
    MODELS_TO_PLOT = [
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen3-14B",
    ]
    models = [m for m in MODELS_TO_PLOT if m in df["model"].unique()]

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    cell  = 4.2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell * ncols, cell * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, model_id in enumerate(models):
        title = _DISPLAY.get(model_id, model_id.split("/")[-1])
        _plot_panel(axes_flat[idx], df[df["model"] == model_id], title)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"dq_iter_weight_only.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
