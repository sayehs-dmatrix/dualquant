import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

import os as _os
_DQ_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))  # repo root, resolved from this file

# ── Pick a model ────────────────────────────────────────────────────────────
# MODEL = "meta-llama_Llama-3.1-8B"
MODEL = "meta-llama_Llama-3.2-1B"
# MODEL = "Qwen_Qwen3-0.6B"
# MODEL = "Qwen_Qwen2.5-7B"

SQ_DIR = _os.path.join(_DQ_ROOT, "scales_smoothquant_rtn_int4")
DQ_DIR = _os.path.join(_DQ_ROOT, "scales_dualquant_rtn_int4")
OUT_DIR = f"./plots_{MODEL}"
os.makedirs(OUT_DIR, exist_ok=True)

sq05 = torch.load(f"{SQ_DIR}/SQ_scales_{MODEL}_alpha_0.5.pt")["saved_scales"]
sq00 = torch.load(f"{SQ_DIR}/SQ_scales_{MODEL}_alpha_0.0.pt")["saved_scales"]   # weight-only trivial baseline
dq   = torch.load(f"{DQ_DIR}/DQ_{MODEL}_iter15scales_rtn_int4_init_l1Norm.pt")

# Quick range check on layer-0 q_proj
sq05_q = sq05["model.layers.0.q_proj"].float().numpy()
sq00_q = sq00["model.layers.0.q_proj"].float().numpy()
dq_q   = dq["layer0.self_attn.q_proj"].float().numpy()
print(f"SQ α=0.5 | min: {sq05_q.min():.4f}  max: {sq05_q.max():.4f}  ratio: {sq05_q.max()/max(sq05_q.min(), 1e-12):.1f}x")
print(f"SQ α=0   | min: {sq00_q.min():.4f}  max: {sq00_q.max():.4f}  ratio: {sq00_q.max()/max(sq00_q.min(), 1e-12):.1f}x")
print(f"DQ       | min: {dq_q.min():.4f}  max: {dq_q.max():.4f}  ratio: {dq_q.max()/max(dq_q.min(), 1e-12):.1f}x")

NUM_LAYERS = max(
    int(k.split('.', 1)[0].replace('layer', '')) for k in dq.keys()
) + 1
print(f"\nMODEL = {MODEL} | NUM_LAYERS = {NUM_LAYERS}")

proj_map = {
    "q_proj":    ("attn", "q_proj",    "self_attn.q_proj"),
    "k_proj":    ("attn", "k_proj",    "self_attn.k_proj"),
    "v_proj":    ("attn", "v_proj",    "self_attn.v_proj"),
    "gate_proj": ("ffn",  "gate_proj", "mlp.gate_proj"),
    "up_proj":   ("ffn",  "up_proj",   "mlp.up_proj"),
}
PROJ_COLORS = {
    "q_proj":    "#378ADD",
    "k_proj":    "#1D9E75",
    "v_proj":    "#D85A30",
    "gate_proj": "#7F77DD",
    "up_proj":   "#BA7517",
}

def zscore(arr, eps=1e-8):
    return (arr - arr.mean()) / (arr.std() + eps)

def cosine_sim(x, y, eps=1e-8):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + eps)

def subsample(x, y, n=400):
    idx = np.random.choice(len(x), min(n, len(x)), replace=False)
    return x[idx], y[idx]

def add_regression_line(ax, x, y, color):
    m, b = np.polyfit(x, y, 1)
    xr = np.array([x.min(), x.max()])
    ax.plot(xr, m*xr+b, color=color, linewidth=1.5,
            linestyle='--', alpha=0.9)

for l in range(NUM_LAYERS):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"SmoothQuant vs DualQuant — {MODEL}  Layer {l:02d}  "
        f"(z-score normalized, each method independently)",
        fontsize=12, fontweight='normal', y=1.005
    )
    gs = gridspec.GridSpec(3, 5, figure=fig,
                           hspace=0.65, wspace=0.40)

    sq_prefix = f"model.layers.{l}"
    dq_prefix = f"layer{l}"

    for col, (proj, (block, sq_perproj_suffix, dq_proj_suffix)) \
            in enumerate(proj_map.items()):

        sq05_share = sq05[f"{sq_prefix}.{block}"].float().numpy()
        sq05_pp    = sq05[f"{sq_prefix}.{sq_perproj_suffix}"].float().numpy()
        sq00_pp    = sq00[f"{sq_prefix}.{sq_perproj_suffix}"].float().numpy()
        dq_pp      = dq[f"{dq_prefix}.{dq_proj_suffix}"].float().numpy()

        # z-score each vector independently
        sq05_share_z = zscore(sq05_share)
        sq05_pp_z    = zscore(sq05_pp)
        sq00_pp_z    = zscore(sq00_pp)
        dq_z         = zscore(dq_pp)

        color = PROJ_COLORS[proj]

        # Row 0: [C]  SQ α=0.5 shared    vs DQ
        # Row 1: [B]  SQ α=0.5 per-proj  vs DQ
        # Row 2: [D]  SQ α=0   per-proj  vs DQ      (weight-only baseline)
        for row, (sq_z, sq_raw, xlabel) in enumerate([
            (sq05_share_z, sq05_share, "SQ α=0.5 shared (z-score)"),
            (sq05_pp_z,    sq05_pp,    "SQ α=0.5 per-proj (z-score)"),
            (sq00_pp_z,    sq00_pp,    "SQ α=0 per-proj (z-score)"),
        ]):
            ax = fig.add_subplot(gs[row, col])

            x_sub, y_sub = subsample(sq_z, dq_z)
            ax.scatter(x_sub, y_sub, s=8, alpha=0.4,
                       color=color, linewidths=0)
            add_regression_line(ax, sq_z, dq_z, color)

            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
            ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)

            lim_min = min(sq_z.min(), dq_z.min()) - 0.2
            lim_max = max(sq_z.max(), dq_z.max()) + 0.2
            ax.plot([lim_min, lim_max], [lim_min, lim_max],
                    color='gray', linewidth=0.7,
                    linestyle=':', alpha=0.5)
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)

            # Pearson (on z-scored vectors == Pearson on raw, since z-score is linear)
            r_z,  _ = pearsonr(sq_z, dq_z)
            cos_z   = cosine_sim(sq_z, dq_z)
            cos_raw = cosine_sim(sq_raw, dq_pp)

            ax.set_title(
                f"{proj}\n"
                f"P={r_z:+.3f}  "
                f"cos(z)={cos_z:+.3f}  "
                f"cos(raw)={cos_raw:+.3f}",
                fontsize=8
            )
            if col == 0:
                ax.set_ylabel("DualQuant (z-score)", fontsize=8)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)

    # Row labels (y-positions are approximate centres of each row in figure coords)
    fig.text(0.005, 0.83, "[C]  SQ α=0.5 shared vs DQ",
             va='center', rotation='vertical',
             fontsize=9, color='gray')
    fig.text(0.005, 0.53, "[B]  SQ α=0.5 per-proj vs DQ",
             va='center', rotation='vertical',
             fontsize=9, color='gray')
    fig.text(0.005, 0.23, "[D]  SQ α=0   per-proj vs DQ\n(weight-only baseline)",
             va='center', rotation='vertical',
             fontsize=9, color='gray')

    out_path = os.path.join(OUT_DIR, f"scatter_zscore_layer{l:02d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")

print("done.")
