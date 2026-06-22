"""
Compare SmoothQuant and DualQuant column scales across all available configs.

SQ files  → scales_smoothquant_rtn_int4/  (varies: alpha × weight_stat)
DQ files  → scales_dualquant_rtn_int4/    (varies: scale_option × row_init × col_init)

Compared projections: q_proj, k_proj, v_proj, gate_proj, up_proj
  (o_proj and down_proj are absent from SQ.)

Two SQ scale types compared against each DQ projection:
  per-proj : model.layers.{l}.{proj}   — scale computed from that projection alone
  shared   : model.layers.{l}.attn/ffn — shared scale actually applied to the LayerNorm
                                         (max over q+k+v for attn; max over gate+up for ffn)

PDF layout: one page per projection, 2 rows × 4 cols.
  Row 0 = per-projection SQ vs DQ    Row 1 = shared SQ vs DQ
  Cols  = Pearson / Log-Pearson / Spearman / Top-20 overlap
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr, spearmanr

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(os.path.dirname(_HERE))  # scale_comparison/ → Rebutal_Neurips/ → Dualquant_codebase_20260508/
SQ_DIR   = os.path.join(_ROOT, "scales_smoothquant_rtn_int4")
DQ_DIR   = os.path.join(_ROOT, "scales_dualquant_rtn_int4")

# ── Model ────────────────────────────────────────────────────────────────────
# Use the HuggingFace model ID (e.g. "Qwen/Qwen3-0.6B" or "meta-llama/Llama-3.2-1B").
# The slash is replaced with _ automatically when building filenames.
# MODEL = "Qwen/Qwen3-0.6B"
MODEL = "meta-llama/Llama-3.2-1B"
_M = MODEL.replace("/", "_")   # filesystem-safe tag used in all filenames

# ── Available SQ configs  (label → filename) ─────────────────────────────────
SQ_CONFIGS = {
    "α=0.5 max_abs": f"SQ_scales_{_M}_alpha_0.5_wstat_max_abs.pt",
    "α=0.5 l1_norm": f"SQ_scales_{_M}_alpha_0.5_wstat_l1_norm.pt",
    "α=0.0 max_abs": f"SQ_scales_{_M}_alpha_0.0_wstat_max_abs.pt",
    "α=0.0 l1_norm": f"SQ_scales_{_M}_alpha_0.0_wstat_l1_norm.pt",
}

# ── Available DQ configs  (label → filename) ─────────────────────────────────
DQ_CONFIGS = {
    "rowcol row=all1 col=l1":  f"DQ_{_M}_iter15_scale_row_column_row_all_one_col_l1_norm_rtn_int4.pt",
    "rowcol row=all1 col=max": f"DQ_{_M}_iter15_scale_row_column_row_all_one_col_max_abs_rtn_int4.pt",
    "rowcol row=max  col=l1":  f"DQ_{_M}_iter15_scale_row_column_row_max_abs_col_l1_norm_rtn_int4.pt",
    "colonly row=all1 col=l1": f"DQ_{_M}_iter15_scale_only_column_row_all_one_col_l1_norm_rtn_int4.pt",
    "colonly row=all1 col=max":f"DQ_{_M}_iter15_scale_only_column_row_all_one_col_max_abs_rtn_int4.pt",
}

# ── Projection map: human name → (SQ per-proj suffix, SQ shared suffix, DQ suffix)
PROJ_MAP = {
    "q_proj":    ("q_proj",    "attn", "self_attn.q_proj"),
    "k_proj":    ("k_proj",    "attn", "self_attn.k_proj"),
    "v_proj":    ("v_proj",    "attn", "self_attn.v_proj"),
    "gate_proj": ("gate_proj", "ffn",  "mlp.gate_proj"),
    "up_proj":   ("up_proj",   "ffn",  "mlp.up_proj"),
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _topk_overlap(s1, s2, k):
    top1 = set(np.argsort(s1)[-k:])
    top2 = set(np.argsort(s2)[-k:])
    return len(top1 & top2) / k


def compare_scales(s1, s2, eps=1e-8):
    s1 = s1.detach().float().cpu().flatten().numpy()
    s2 = s2.detach().float().cpu().flatten().numpy()
    if len(s1) != len(s2):
        return None
    if np.std(s1) < eps or np.std(s2) < eps:
        return None

    pearson_r,  _ = pearsonr(s1, s2)
    spearman_r, _ = spearmanr(s1, s2)
    s1_log = np.log(np.clip(s1, eps, None))
    s2_log = np.log(np.clip(s2, eps, None))
    pearson_log, _ = pearsonr(s1_log, s2_log)

    return {
        "pearson_raw": pearson_r,
        "spearman_raw": spearman_r,
        "pearson_log": pearson_log,
        "overlap_top20": _topk_overlap(s1, s2, 20),
    }


# ── Load files ────────────────────────────────────────────────────────────────

def _load_sq(filename):
    path = os.path.join(SQ_DIR, filename)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    return torch.load(path)["saved_scales"]

def _load_dq(filename):
    path = os.path.join(DQ_DIR, filename)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    return torch.load(path)

sq_data = {label: _load_sq(fname) for label, fname in SQ_CONFIGS.items()}
dq_data = {label: _load_dq(fname) for label, fname in DQ_CONFIGS.items()}

# Drop any that failed to load
sq_data = {k: v for k, v in sq_data.items() if v is not None}
dq_data = {k: v for k, v in dq_data.items() if v is not None}

if not dq_data:
    raise FileNotFoundError(
        f"No DQ scale files found for model '{MODEL}' (tag '{_M}') in {DQ_DIR}.\n"
        f"Run the DQ sweep for this model first, or set MODEL to one that has files."
    )
if not sq_data:
    raise FileNotFoundError(
        f"No SQ scale files found for model '{MODEL}' (tag '{_M}') in {SQ_DIR}."
    )

# Infer number of layers from DQ keys
_any_dq = next(iter(dq_data.values()))
NUM_LAYERS = max(int(k.split('.')[0].replace('layer', '')) for k in _any_dq) + 1
print(f"\nMODEL = {MODEL}  |  NUM_LAYERS = {NUM_LAYERS}")
print(f"SQ configs loaded: {list(sq_data)}")
print(f"DQ configs loaded: {list(dq_data)}\n")


# ── Run comparisons ───────────────────────────────────────────────────────────
# results_pp[sq][dq][layer][proj]     — per-projection SQ key vs DQ
# results_sh[sq][dq][layer][proj]     — shared SQ key (attn/ffn) vs DQ

def _empty_result_store():
    return {
        sq_lbl: {dq_lbl: {l: {} for l in range(NUM_LAYERS)} for dq_lbl in dq_data}
        for sq_lbl in sq_data
    }

results_pp = _empty_result_store()
results_sh = _empty_result_store()

for sq_lbl, sq_scales in sq_data.items():
    for dq_lbl, dq_scales in dq_data.items():
        print(f"\n{'='*60}")
        print(f"SQ: {sq_lbl}   vs   DQ: {dq_lbl}")
        print(f"{'='*60}")
        for l in range(NUM_LAYERS):
            sq_prefix = f"model.layers.{l}"
            dq_prefix = f"layer{l}"
            for proj, (sq_pp_sfx, sq_sh_sfx, dq_sfx) in PROJ_MAP.items():
                dq_key    = f"{dq_prefix}.{dq_sfx}"
                sq_pp_key = f"{sq_prefix}.{sq_pp_sfx}"
                sq_sh_key = f"{sq_prefix}.{sq_sh_sfx}"

                if dq_key not in dq_scales:
                    results_pp[sq_lbl][dq_lbl][l][proj] = None
                    results_sh[sq_lbl][dq_lbl][l][proj] = None
                    continue

                results_pp[sq_lbl][dq_lbl][l][proj] = (
                    compare_scales(sq_scales[sq_pp_key], dq_scales[dq_key])
                    if sq_pp_key in sq_scales else None
                )
                results_sh[sq_lbl][dq_lbl][l][proj] = (
                    compare_scales(sq_scales[sq_sh_key], dq_scales[dq_key])
                    if sq_sh_key in sq_scales else None
                )

                m = results_pp[sq_lbl][dq_lbl][l][proj]
                if m is not None:
                    print(f"  L{l:02d} {proj:<10} [per-proj] | "
                          f"P:{m['pearson_raw']:+.4f}  "
                          f"logP:{m['pearson_log']:+.4f}  "
                          f"S:{m['spearman_raw']:+.4f}  "
                          f"top20:{m['overlap_top20']:.2%}")


# ── Summary tables ────────────────────────────────────────────────────────────

def _mean_metric(store, sq_lbl, dq_lbl, proj, key):
    vals = [
        store[sq_lbl][dq_lbl][l][proj][key]
        for l in range(NUM_LAYERS)
        if store[sq_lbl][dq_lbl][l].get(proj) is not None
    ]
    return float(np.mean(vals)) if vals else float('nan')


METRICS = [
    ("Pearson",      "pearson_raw"),
    ("Log-Pearson",  "pearson_log"),
    ("Spearman",     "spearman_raw"),
    ("Top-20 Ovlp",  "overlap_top20"),
]

# Short display names for the table axes.
# DQ: "rowcol row=all1 col=l1" → "RC/a1/l1"  etc.
_DQ_SHORT = {
    "rowcol row=all1 col=l1":  "RC/a1/l1",
    "rowcol row=all1 col=max": "RC/a1/mx",
    "rowcol row=max  col=l1":  "RC/mx/l1",
    "colonly row=all1 col=l1": "CO/a1/l1",
    "colonly row=all1 col=max":"CO/a1/mx",
}
_SQ_SHORT = {
    "α=0.5 max_abs": "α0.5/max",
    "α=0.5 l1_norm": "α0.5/l1 ",
    "α=0.0 max_abs": "α0.0/max",
    "α=0.0 l1_norm": "α0.0/l1 ",
}

sq_lbls  = list(sq_data)
dq_lbls  = list(dq_data)
dq_short = [_DQ_SHORT.get(d, d[:9]) for d in dq_lbls]
sq_short = [_SQ_SHORT.get(s, s[:9]) for s in sq_lbls]

ROW_W = 10   # SQ label column
COL_W = 10   # per-DQ-config column

for metric_label, metric_key in METRICS:
    print(f"\n\n{'='*70}")
    print(f"SUMMARY — {metric_label}  (mean over {NUM_LAYERS} layers)")
    print(f"  Rows = SmoothQuant config   Cols = DualQuant config")
    print(f"  RC=row_column  CO=col_only  a1=row_init=all_one  mx=col_init=max_abs  l1=col_init=l1_norm")
    print(f"{'='*70}")

    col_header = f"{'':>{ROW_W}}" + "".join(f"{h:>{COL_W}}" for h in dq_short)
    sep = "-" * len(col_header)

    for proj in PROJ_MAP:
        print(f"\n  {proj}  [per-projection SQ]")
        print(f"  {col_header}")
        print(f"  {sep}")
        for sq_lbl, sq_sh in zip(sq_lbls, sq_short):
            row = f"  {sq_sh:>{ROW_W}}"
            for dq_lbl in dq_lbls:
                v = _mean_metric(results_pp, sq_lbl, dq_lbl, proj, metric_key)
                row += f"{v:>{COL_W}.4f}"
            print(row)

        print(f"\n  {proj}  [shared SQ — scale actually applied to LayerNorm]")
        print(f"  {col_header}")
        print(f"  {sep}")
        for sq_lbl, sq_sh in zip(sq_lbls, sq_short):
            row = f"  {sq_sh:>{ROW_W}}"
            for dq_lbl in dq_lbls:
                v = _mean_metric(results_sh, sq_lbl, dq_lbl, proj, metric_key)
                row += f"{v:>{COL_W}.4f}"
            print(row)


# ── PDF heatmaps ──────────────────────────────────────────────────────────────
# Layout: 10 pages — one per (projection × SQ type).
# Each page: 4 rows × 3 cols.
#   Rows = mean-over-all-layers / Layer 0 / Layer 10 / Layer 15
#   Cols = Pearson / Log-Pearson / Spearman   (overlap removed)
# Each cell in a heatmap: rows=SQ configs, cols=DQ configs.

_OUT_PDF = os.path.join(_HERE, f"sq_vs_dq_configs_{_M}.pdf")

# PDF uses only the 3 correlation metrics — overlap dropped.
PDF_METRICS = [
    ("Pearson",     "pearson_raw"),
    ("Log-Pearson", "pearson_log"),
    ("Spearman",    "spearman_raw"),
]

# Which SQ config to show in the PDF. Must match a key in SQ_CONFIGS.
FIXED_SQ = "α=0.5 max_abs"
# FIXED_SQ = "α=0.5 l1_norm"
# FIXED_SQ = "α=0.0 max_abs"
# FIXED_SQ = "α=0.0 l1_norm"

# Example layer shown alongside the mean.
EXAMPLE_LAYER = 10
# EXAMPLE_LAYER = 0
# EXAMPLE_LAYER = 15

PAGES = [
    ("Mean  (all layers)", None),
    (f"Layer {EXAMPLE_LAYER}",  EXAMPLE_LAYER),
]

SQ_PANELS = [
    ("SQ per-projection",            results_pp),
    ("SQ shared  (applied to LN)",   results_sh),
]

def _cell(store, sq_lbl, dq_lbl, proj, key, layer):
    if layer is None:
        return _mean_metric(store, sq_lbl, dq_lbl, proj, key)
    m = store[sq_lbl][dq_lbl][layer].get(proj)
    return float('nan') if m is None else m.get(key, float('nan'))

def _sq_dq_mat(store, proj, key, layer):
    """Build (n_sq × n_dq) matrix — rows=SQ configs, cols=DQ configs."""
    return np.array([
        [_cell(store, sq, dq, proj, key, layer) for dq in dq_lbls]
        for sq in sq_lbls
    ])

# Layout: 10 pages — one per (projection × layer view).
# Each page: 3 rows (metrics) × 2 cols (per-proj | shared).
# Each subplot: heatmap  rows=SQ configs (4)  cols=DQ configs (5).
proj_list = list(PROJ_MAP)

with PdfPages(_OUT_PDF) as pdf:
    for proj in proj_list:
        for page_label, layer in PAGES:
            fig, axes = plt.subplots(
                len(PDF_METRICS), len(SQ_PANELS),
                figsize=(8 * len(SQ_PANELS), 6 * len(PDF_METRICS)),
            )
            fig.suptitle(
                f"{MODEL}  —  {proj}  |  {page_label}\n"
                f"Rows = SQ config    Cols = DQ config\n"
                f"RC=row_column  CO=col_only  "
                f"a1=row_init=all_one  mx=col_init=max_abs  l1=col_init=l1_norm",
                fontsize=13, y=1.02,
            )

            for row_idx, (metric_label, metric_key) in enumerate(PDF_METRICS):
                for col_idx, (panel_title, store) in enumerate(SQ_PANELS):
                    ax = axes[row_idx][col_idx]
                    mat = _sq_dq_mat(store, proj, metric_key, layer)

                    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            v = mat[i, j]
                            text_color = "white" if abs(v) > 0.6 else "black"
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                    color=text_color, fontsize=14, fontweight="bold")

                    ax.set_xticks(range(len(dq_lbls)))
                    ax.set_xticklabels(dq_short, rotation=30, ha="right", fontsize=12)
                    ax.set_yticks(range(len(sq_lbls)))
                    ax.set_yticklabels(sq_short if col_idx == 0 else [], fontsize=12)

                    if row_idx == 0:
                        ax.set_title(panel_title, fontsize=13, pad=8)
                    if col_idx == 0:
                        ax.set_ylabel(metric_label, fontsize=13, labelpad=10)

                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=11)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

print(f"\nWrote heatmap PDF: {_OUT_PDF}")
