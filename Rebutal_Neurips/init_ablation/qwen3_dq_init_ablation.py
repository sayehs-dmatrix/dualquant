"""Init-sensitivity ablation for Dualquant on Qwen3-0.6B.

For each Dualquant init variant we have, compare against the corresponding
SmoothQuant scales (α=0.5 and α=0.0) and against each other. The point is
to check whether choosing col_init / row_init materially changes either:
  - the saved DQ `1/β` itself                 (DQ-vs-DQ pairwise)
  - the agreement between DQ and SmoothQuant  (SQ-vs-DQ, per variant)

Standalone — no imports from the other Rebutal_Neurips scripts. Disposable.
"""

import os
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ── Config ──────────────────────────────────────────────────────────────────
MODEL  = "meta-llama_Llama-3.1-8B"
SQ_DIR = "/home/coder/numrd/Quantization_Repo_July2025/Dualquant_codebase_20260508/scales_smoothquant_rtn_int4"
DQ_DIR = "/home/coder/numrd/Quantization_Repo_July2025/Dualquant_codebase_20260508/scales_dualquant_rtn_int4"

# DQ init tags — filename is `DQ_{MODEL}_iter15scales_rtn_int4_init_{tag}.pt`
DQ_TAGS = [
    "l1Norm",               # col=l1_norm, row=max_abs (original)
    "l1Norm_maxabs",        # col=l1_norm, row=max_abs (re-run)
    "l1Norm_row_all_one",   # col=l1_norm, row=all_one
    "all_one_maxabs",       # col=all_one, row=max_abs
    "all_one_all_one",      # col=all_one, row=all_one
]

proj_map = {
    "q_proj":    ("attn", "q_proj",    "self_attn.q_proj"),
    "k_proj":    ("attn", "k_proj",    "self_attn.k_proj"),
    "v_proj":    ("attn", "v_proj",    "self_attn.v_proj"),
    "gate_proj": ("ffn",  "gate_proj", "mlp.gate_proj"),
    "up_proj":   ("ffn",  "up_proj",   "mlp.up_proj"),
}


def to_np(t):
    return t.detach().float().cpu().flatten().numpy()


def correl(s1, s2, eps=1e-8):
    s1, s2 = to_np(s1), to_np(s2)
    # Drop channels where either side is non-finite (DQ all_one init can produce inf 1/β).
    finite = np.isfinite(s1) & np.isfinite(s2)
    if finite.sum() < 3:
        return None
    s1, s2 = s1[finite], s2[finite]
    if np.std(s1) < eps or np.std(s2) < eps:
        return None
    pr,  _ = pearsonr(s1, s2)
    sr,  _ = spearmanr(s1, s2)
    plr, _ = pearsonr(np.log(np.clip(s1, eps, None)),
                      np.log(np.clip(s2, eps, None)))
    return {"P": pr, "logP": plr, "S": sr, "n_finite": int(finite.sum()), "n_total": int(finite.size)}


# ── Load ────────────────────────────────────────────────────────────────────
sq05 = torch.load(f"{SQ_DIR}/SQ_scales_{MODEL}_alpha_0.5.pt")["saved_scales"]
sq00 = torch.load(f"{SQ_DIR}/SQ_scales_{MODEL}_alpha_0.0.pt")["saved_scales"]

dq_variants = {}
for tag in DQ_TAGS:
    path = f"{DQ_DIR}/DQ_{MODEL}_iter15scales_rtn_int4_init_{tag}.pt"
    if os.path.exists(path):
        dq_variants[tag] = torch.load(path)
    else:
        print(f"  [skip] {tag} — not found at {path}")

if not dq_variants:
    raise SystemExit("No DQ variants loaded.")

ref_dq = next(iter(dq_variants.values()))
NUM_LAYERS = max(
    int(k.split('.', 1)[0].replace('layer', '')) for k in ref_dq.keys()
) + 1
print(f"\nMODEL={MODEL}  NUM_LAYERS={NUM_LAYERS}  DQ variants loaded: {list(dq_variants.keys())}\n")

# Sanity: how many non-finite entries per DQ variant?
print(f"{'DQ init':<22} | {'total channels':>14} | {'non-finite':>11}")
print("-" * 54)
for tag, dq in dq_variants.items():
    total, bad = 0, 0
    for v in dq.values():
        a = to_np(v)
        total += a.size
        bad += int((~np.isfinite(a)).sum())
    print(f"{tag:<22} | {total:>14d} | {bad:>11d}")
print()


# ── Computation: mean correlations over layers ─────────────────────────────
def mean_sq_vs_dq(sq_dict, dq_dict, sq_use_perproj=True):
    """For each projection, return mean(P, logP, S) over all layers
       between `sq_dict[<sq_key>]` and `dq_dict[<dq_key>]`.
       sq_use_perproj=True picks the per-projection SQ key; False picks shared (block)."""
    out = {}
    for proj, (block, sq_pp, dq_suffix) in proj_map.items():
        sq_suffix = sq_pp if sq_use_perproj else block
        accum = {"P": [], "logP": [], "S": []}
        for l in range(NUM_LAYERS):
            r = correl(sq_dict[f"model.layers.{l}.{sq_suffix}"],
                       dq_dict[f"layer{l}.{dq_suffix}"])
            if r:
                for k in accum:
                    accum[k].append(r[k])
        out[proj] = {k: float(np.mean(v)) if v else float('nan') for k, v in accum.items()}
    return out


def mean_dq_vs_dq(dq_a, dq_b):
    """Mean Pearson across all (layer × projection) entries."""
    vals = []
    for proj, (_, _, dq_suffix) in proj_map.items():
        for l in range(NUM_LAYERS):
            key = f"layer{l}.{dq_suffix}"
            r = correl(dq_a[key], dq_b[key])
            if r:
                vals.append(r["P"])
    return float(np.mean(vals)) if vals else float('nan')


# ── Print helpers ──────────────────────────────────────────────────────────
def print_sq_vs_dq(title, sq_dict, sq_use_perproj, metric):
    """metric = 'P' | 'logP' | 'S' """
    label = {"P": "Pearson", "logP": "Log-Pearson", "S": "Spearman"}[metric]
    print("=" * 92)
    print(f"{title}  —  {label} (mean over {NUM_LAYERS} layers)")
    print("=" * 92)
    header = f"{'DQ init':<22} | " + " ".join(f"{p:>10}" for p in proj_map) + f" | {'mean':>8}"
    print(header)
    print("-" * len(header))
    for tag, dq in dq_variants.items():
        row = mean_sq_vs_dq(sq_dict, dq, sq_use_perproj)
        cells = [row[p][metric] for p in proj_map]
        mean = float(np.mean(cells))
        print(f"{tag:<22} | " + " ".join(f"{v:>+10.4f}" for v in cells) + f" | {mean:>+8.4f}")
    print()


# ── Tables ─────────────────────────────────────────────────────────────────
print_sq_vs_dq("[B]  SQ α=0.5 per-proj vs DQ", sq05, sq_use_perproj=True,  metric="P")
print_sq_vs_dq("[B]  SQ α=0.5 per-proj vs DQ", sq05, sq_use_perproj=True,  metric="logP")

print_sq_vs_dq("[D]  SQ α=0   per-proj vs DQ  (weight-only baseline)",
               sq00, sq_use_perproj=True,  metric="P")

print_sq_vs_dq("[C]  SQ α=0.5 shared   vs DQ", sq05, sq_use_perproj=False, metric="P")

# ── DQ ↔ DQ init-sensitivity matrix ───────────────────────────────────────
tags = list(dq_variants.keys())
print("=" * 92)
print(f"[F]  DQ ↔ DQ pairwise mean Pearson  (mean over layers × projections)")
print("=" * 92)
print(f"{'':<22} | " + " ".join(f"{t[:18]:>18}" for t in tags))
print("-" * (22 + 3 + (18 + 1) * len(tags)))
for ta in tags:
    cells = []
    for tb in tags:
        v = 1.0 if ta == tb else mean_dq_vs_dq(dq_variants[ta], dq_variants[tb])
        cells.append(f"{v:>18.4f}")
    print(f"{ta:<22} | " + " ".join(cells))
print()


# ── PDF output: color-coded heatmaps for the init-ablation ───────────────────
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _build_sq_vs_dq_matrix(sq_dict, sq_use_perproj, metric):
    """Return (n_variants × n_projs) np.ndarray of mean correlations."""
    M = np.zeros((len(tags), len(proj_map)))
    for i, tag in enumerate(tags):
        row = mean_sq_vs_dq(sq_dict, dq_variants[tag], sq_use_perproj=sq_use_perproj)
        for j, p in enumerate(proj_map):
            M[i, j] = row[p][metric]
    return M


def _build_dq_dq_matrix():
    """(n_variants × n_variants) DQ↔DQ pairwise mean Pearson."""
    n = len(tags)
    M = np.zeros((n, n))
    for i, ta in enumerate(tags):
        for j, tb in enumerate(tags):
            M[i, j] = 1.0 if ta == tb else mean_dq_vs_dq(dq_variants[ta], dq_variants[tb])
    return M


def _heatmap_page(pdf, mat, row_labels, col_labels, title, vmin=-1.0, vmax=1.0,
                  cmap="RdBu_r", fmt="{:+.3f}"):
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(col_labels) + 4),
                                    max(3.5, 0.55 * len(row_labels) + 2)))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("correlation", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


_OUT_PDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"dq_init_ablation_{MODEL}.pdf",
)
projs = list(proj_map.keys())

with PdfPages(_OUT_PDF) as pdf:
    _heatmap_page(pdf,
        _build_sq_vs_dq_matrix(sq05, sq_use_perproj=True,  metric="P"),
        row_labels=tags, col_labels=projs,
        title=f"{MODEL}  —  [B] SQ α=0.5 per-proj vs DQ  (Pearson, mean over {NUM_LAYERS} layers)")
    _heatmap_page(pdf,
        _build_sq_vs_dq_matrix(sq05, sq_use_perproj=True,  metric="logP"),
        row_labels=tags, col_labels=projs,
        title=f"{MODEL}  —  [B] SQ α=0.5 per-proj vs DQ  (Log-Pearson, mean over {NUM_LAYERS} layers)")
    _heatmap_page(pdf,
        _build_sq_vs_dq_matrix(sq00, sq_use_perproj=True,  metric="P"),
        row_labels=tags, col_labels=projs,
        title=f"{MODEL}  —  [D] SQ α=0 per-proj vs DQ  (weight-only baseline, Pearson)")
    _heatmap_page(pdf,
        _build_sq_vs_dq_matrix(sq05, sq_use_perproj=False, metric="P"),
        row_labels=tags, col_labels=projs,
        title=f"{MODEL}  —  [C] SQ α=0.5 shared vs DQ  (Pearson, mean over {NUM_LAYERS} layers)")
    _heatmap_page(pdf,
        _build_dq_dq_matrix(),
        row_labels=tags, col_labels=tags,
        title=f"{MODEL}  —  [F] DQ ↔ DQ pairwise mean Pearson")

print(f"Wrote heatmap PDF: {_OUT_PDF}")

print("done.")
