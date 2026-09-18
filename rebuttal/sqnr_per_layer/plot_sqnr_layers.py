"""Plot per-layer block SQNR for DualQuant vs SmoothQuant.

Generates one figure per model (Llama-3.2-1B, Qwen3-0.6B), each with:
  - Top row: SQNR per layer for A16W4 (actoff) and A4W4 (acton), with ±std bands
  - Bottom row: DQ − SQ difference per layer, with ±std bands and zero line

Output: results/sqnr/sqnr_layer_plot_<model>.pdf
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SQNR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "results", "sqnr")

MODELS = {
    "Llama-3.2-1B": "meta-llama_Llama-3.2-1B",
    "Llama-3.1-8B": "meta-llama_Llama-3.1-8B",
    "Qwen3-0.6B":   "Qwen_Qwen3-0.6B",
}

SETTINGS = [
    ("A16W4  (weight-only)", "actoff"),
    ("A4W4   (weight + activation)", "acton"),
]

DQ_COLOR  = "steelblue"
SQ_COLOR  = "tomato"
DIFF_COLOR = "darkgreen"


def load(model_tag, act):
    path = os.path.join(SQNR_DIR, f"{model_tag}_rtn_int4_{act}.csv")
    df = pd.read_csv(path)
    # Keep only integer layer rows (drop Mean/Med if present)
    df = df[pd.to_numeric(df["layer"], errors="coerce").notna()].copy()
    df["layer"] = df["layer"].astype(int)
    return df.sort_values("layer").reset_index(drop=True)


for model_label, model_tag in MODELS.items():
    out_pdf = os.path.join(SQNR_DIR, f"sqnr_layer_plot_{model_tag}.pdf")

    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                                 gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(f"{model_label}  —  Per-layer block SQNR: DualQuant vs SmoothQuant\n"
                     f"(rtn_int4, mean ± std across samples)",
                     fontsize=13, y=1.01)

        for col, (setting_label, act) in enumerate(SETTINGS):
            df = load(model_tag, act)
            layers = df["layer"].values
            dq_mean = df["dq_mean"].values
            dq_std  = df["dq_std"].values
            sq_mean = df["sq_mean"].values
            sq_std  = df["sq_std"].values
            diff    = dq_mean - sq_mean
            # Error propagation for difference: sqrt(σ_dq² + σ_sq²)
            diff_std = np.sqrt(dq_std**2 + sq_std**2)

            # ── Top: absolute SQNR ──────────────────────────────────────────
            ax = axes[0, col]
            ax.fill_between(layers, dq_mean - dq_std, dq_mean + dq_std,
                            alpha=0.15, color=DQ_COLOR)
            ax.fill_between(layers, sq_mean - sq_std, sq_mean + sq_std,
                            alpha=0.15, color=SQ_COLOR)
            ax.plot(layers, dq_mean, "o-", color=DQ_COLOR, lw=2, ms=4,
                    label=f"DualQuant  (mean={dq_mean.mean():.1f} dB)")
            ax.plot(layers, sq_mean, "s-", color=SQ_COLOR, lw=2, ms=4,
                    label=f"SmoothQuant (mean={sq_mean.mean():.1f} dB)")
            ax.set_title(setting_label, fontsize=12)
            ax.set_ylabel("SQNR (dB)", fontsize=11)
            ax.set_xlabel("Layer", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(layers[::2])

            # ── Bottom: DQ − SQ difference ─────────────────────────────────
            ax2 = axes[1, col]
            ax2.fill_between(layers, diff - diff_std, diff + diff_std,
                             alpha=0.15, color=DIFF_COLOR)
            ax2.plot(layers, diff, "^-", color=DIFF_COLOR, lw=2, ms=4,
                     label="DQ − SQ")
            ax2.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
            # Shade positive (DQ better) vs negative (SQ better)
            ax2.fill_between(layers, diff, 0,
                             where=diff >= 0, alpha=0.10, color=DQ_COLOR,
                             label="DQ better")
            ax2.fill_between(layers, diff, 0,
                             where=diff < 0,  alpha=0.10, color=SQ_COLOR,
                             label="SQ better")
            ax2.set_ylabel("DQ − SQ (dB)", fontsize=11)
            ax2.set_xlabel("Layer", fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(layers[::2])

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote: {out_pdf}")
