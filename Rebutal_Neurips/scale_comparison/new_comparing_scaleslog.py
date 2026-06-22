import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

def topk_overlap(s1, s2, k=20):
    """
    What fraction of the top-k channels in s1 are also in top-k of s2?
    """
    top1 = set(np.argsort(s1)[-k:])
    top2 = set(np.argsort(s2)[-k:])
    overlap = len(top1 & top2)
    return overlap / k

# # Example: top-10, top-20, top-50
# for k in [10, 20, 50, 100]:
#     ov = topk_overlap(sq, dq, k=k)
#     print(f"Top-{k:3d} overlap: {ov:.2%}")

def weighted_pearson(s1, s2, power=2.0):
    """
    Weight each channel by its scale magnitude raised to `power`.
    Large channels contribute more to the correlation.
    """
    s1, s2 = np.array(s1), np.array(s2)
    weights = (s1 ** power + s2 ** power) / 2.0
    weights = weights / weights.sum()

    mx = np.sum(weights * s1)
    my = np.sum(weights * s2)

    num = np.sum(weights * (s1 - mx) * (s2 - my))
    dx  = np.sqrt(np.sum(weights * (s1 - mx) ** 2))
    dy  = np.sqrt(np.sum(weights * (s2 - my) ** 2))

    return num / (dx * dy + 1e-8)

def topk_pearson(s1, s2, k=50):
    """
    Compute Pearson only on the union of top-k channels from either method.
    """
    s1, s2 = np.array(s1), np.array(s2)
    top1 = set(np.argsort(s1)[-k:])
    top2 = set(np.argsort(s2)[-k:])
    idx  = np.array(list(top1 | top2))  # union of important channels

    if len(idx) < 3:
        return float('nan')

    r, _ = pearsonr(s1[idx], s2[idx])
    return r

def topk_spearman(s1, s2, k=50):
    from scipy.stats import spearmanr
    s1, s2 = np.array(s1), np.array(s2)
    top1 = set(np.argsort(s1)[-k:])
    top2 = set(np.argsort(s2)[-k:])
    idx  = np.array(list(top1 | top2))
    r, _ = spearmanr(s1[idx], s2[idx])
    return r

###### New Code ######
def compare_scales_normalized(sq_scales, dq_scales, name="", eps=1e-8):
    sq_temp = sq_scales/(torch.norm(sq_scales))
    dq_temp = dq_scales/(torch.norm(dq_scales))

    corr_new_metric = sq_temp@dq_temp
    
    print('new corr is ', corr_new_metric)
    return corr_new_metric.item()



# def compare_scales(s1, s2, name="", eps=1e-8):
#     s1 = s1.detach().float().cpu().flatten()
#     s2 = s2.detach().float().cpu().flatten()
    

#     if s1.numel() != s2.numel():
#         print(f"  [{name}] SKIPPED — shape mismatch: {s1.shape} vs {s2.shape}")
#         return None

#     x, y = s1.numpy(), s2.numpy()

#     if np.std(x) < eps or np.std(y) < eps:
#         print(f"  [{name}] SKIPPED — constant vector")
#         return None

#     pearson_r,  _ = pearsonr(x, y)
#     spearman_r, _ = spearmanr(x, y)

#     # Log-space correlation (more appropriate for multiplicative scales)
#     x_log = np.log(np.clip(x, eps, None))
#     y_log = np.log(np.clip(y, eps, None))
#     pearson_log,  _ = pearsonr(x_log, y_log)
#     spearman_log, _ = spearmanr(x_log, y_log)

#     # Reciprocal check: does s1 * s2 ≈ constant?
#     product    = x * y
#     recip_cv   = np.std(product) / (np.mean(np.abs(product)) + eps)
#     recip_corr, _ = pearsonr(x, 1.0 / np.clip(y, eps, None))

#     print(f"  {name:<25} | "
#           f"Raw  P:{pearson_r:+.4f} S:{spearman_r:+.4f} | "
#           f"Log  P:{pearson_log:+.4f} S:{spearman_log:+.4f} | "
#           f"RecipCorr:{recip_corr:+.4f} CV:{recip_cv:.3f}")

#     return {
#         "pearson_raw": pearson_r,   "spearman_raw": spearman_r,
#         "pearson_log": pearson_log, "spearman_log": spearman_log,
#         "recip_corr":  recip_corr,  "recip_cv":     recip_cv,
#     }


######## New Code #########
def compare_scales(s1, s2, name="", eps=1e-8, topk_vals=(10, 20, 50)):
    s1 = s1.detach().float().cpu().flatten().numpy()
    s2 = s2.detach().float().cpu().flatten().numpy()

    if len(s1) != len(s2):
        print(f"  [{name}] SKIPPED — shape mismatch")
        return None
    if np.std(s1) < eps or np.std(s2) < eps:
        print(f"  [{name}] SKIPPED — constant vector")
        return None

    # ── Standard metrics ──
    pearson_r,  _ = pearsonr(s1, s2)
    spearman_r, _ = spearmanr(s1, s2)

    # Log-space Pearson — scales are multiplicative and heavy-tailed.
    s1_log = np.log(np.clip(s1, eps, None))
    s2_log = np.log(np.clip(s2, eps, None))
    pearson_log, _ = pearsonr(s1_log, s2_log)

    # ── Outlier-focused metrics ──
    overlaps        = {k: topk_overlap(s1, s2, k) for k in topk_vals}
    wpearson        = weighted_pearson(s1, s2, power=2.0)
    topk_p          = {k: topk_pearson(s1, s2, k)  for k in topk_vals}

    if name:
        print(f"\n  [{name}]")
    print(f"    Global  | Pearson: {pearson_r:+.4f}  Spearman: {spearman_r:+.4f}  Log-Pearson: {pearson_log:+.4f}  Weighted-P: {wpearson:+.4f}")
    for k in topk_vals:
        print(f"    Top-{k:<4d}| Overlap: {overlaps[k]:.2%}  Pearson: {topk_p[k]:+.4f}")

    return {
        "pearson_raw":   pearson_r,
        "spearman_raw":  spearman_r,
        "pearson_log":   pearson_log,
        "weighted_pearson": wpearson,
        **{f"overlap_top{k}":  overlaps[k]  for k in topk_vals},
        **{f"pearson_top{k}":  topk_p[k]    for k in topk_vals},
    }

############################

# ── Pick a model ────────────────────────────────────────────────────────────
# MODEL = "meta-llama_Llama-3.1-8B"
# MODEL = "meta-llama_Llama-3.2-1B"
# MODEL = "Qwen_Qwen3-0.6B"
MODEL = "Qwen_Qwen2.5-7B"

SQ_DIR = "/home/coder/numrd/Quantization_Repo_July2025/Dualquant_codebase_20260508/scales_smoothquant_rtn_int4"
DQ_DIR = "/home/coder/numrd/Quantization_Repo_July2025/Dualquant_codebase_20260508/scales_dualquant_rtn_int4"

sq05 = torch.load(f"{SQ_DIR}/SQ_scales_{MODEL}_alpha_0.5.pt")["saved_scales"]
sq00 = torch.load(f"{SQ_DIR}/SQ_scales_{MODEL}_alpha_0.0.pt")["saved_scales"]   # weight-only trivial baseline
dq   = torch.load(f"{DQ_DIR}/DQ_{MODEL}_iter15scales_rtn_int4_init_l1Norm.pt")


###### Comapring each projection layer ###########
# ── Inspect keys first ──
# print("=== SmoothQuant keys (first 20) ===")
# for k in list(sq05.keys())[:20]:
#     print(f"  {k}: {sq05[k].shape}")
# print("\n=== DualQuant keys (first 20) ===")
# for k in list(dq.keys())[:20]:
#     print(f"  {k}: {dq[k].shape}")
import numpy as np

# Auto-detect number of transformer layers from the DQ key set.
NUM_LAYERS = max(
    int(k.split('.', 1)[0].replace('layer', '')) for k in dq.keys()
) + 1
print(f"\nMODEL = {MODEL} | NUM_LAYERS = {NUM_LAYERS}\n")

# sq_prefix.{block}              → shared scale (one per attn/ffn block)
# sq_prefix.{sq_perproj_suffix}  → SQ per-projection scale
# dq_prefix.{dq_proj_suffix}     → dualquant per-projection scale (1/beta)
proj_map = {
    "q_proj":    ("attn", "q_proj",    "self_attn.q_proj"),
    "k_proj":    ("attn", "k_proj",    "self_attn.k_proj"),
    "v_proj":    ("attn", "v_proj",    "self_attn.v_proj"),
    "gate_proj": ("ffn",  "gate_proj", "mlp.gate_proj"),
    "up_proj":   ("ffn",  "up_proj",   "mlp.up_proj"),
}

# Comparisons (per layer × per projection):
#   [A]   SQ α=0.5 shared    vs SQ α=0.5 per-proj   — does sharing distort SQ vs per-proj?
#   [B]   SQ α=0.5 per-proj  vs DQ                   — main comparison
#   [D]   SQ α=0   per-proj  vs DQ                   — weight-only baseline ⇄ DQ
#   [C]   SQ α=0.5 shared    vs DQ
#   [C0]  SQ α=0   shared    vs DQ                   — weight-only shared baseline
#   [E]   SQ α=0.5 per-proj  vs SQ α=0 per-proj      — magnitude of the activation tilt
results = {
    "A_sq05share_vs_sq05pp": {l: {} for l in range(NUM_LAYERS)},
    "B_sq05pp_vs_dq":        {l: {} for l in range(NUM_LAYERS)},
    "D_sq00pp_vs_dq":        {l: {} for l in range(NUM_LAYERS)},
    "C_sq05share_vs_dq":     {l: {} for l in range(NUM_LAYERS)},
    "C0_sq00share_vs_dq":    {l: {} for l in range(NUM_LAYERS)},
    "E_sq05pp_vs_sq00pp":    {l: {} for l in range(NUM_LAYERS)},
}

for l in range(NUM_LAYERS):
    print(f"\n{'='*20} Layer {l:02d} {'='*20}")

    sq_prefix = f"model.layers.{l}"
    dq_prefix = f"layer{l}"

    for proj, (block, sq_perproj_suffix, dq_proj_suffix) in proj_map.items():
        sq05_share = sq05[f"{sq_prefix}.{block}"].float().cpu()
        sq05_pp    = sq05[f"{sq_prefix}.{sq_perproj_suffix}"].float().cpu()
        sq00_share = sq00[f"{sq_prefix}.{block}"].float().cpu()
        sq00_pp    = sq00[f"{sq_prefix}.{sq_perproj_suffix}"].float().cpu()
        dq_pp      = dq[f"{dq_prefix}.{dq_proj_suffix}"].float().cpu()

        print(f"\n  [A]  SQ α=0.5 shared   vs SQ α=0.5 per-proj | {proj}")
        results["A_sq05share_vs_sq05pp"][l][proj] = compare_scales(
            sq05_share, sq05_pp, name=f"SQ05_shared vs SQ05_{proj}"
        )

        print(f"  [B]  SQ α=0.5 per-proj vs DQ                | {proj}")
        results["B_sq05pp_vs_dq"][l][proj] = compare_scales(
            sq05_pp, dq_pp, name=f"SQ05_{proj} vs DQ_{proj}"
        )

        print(f"  [D]  SQ α=0   per-proj vs DQ                | {proj}  (weight-only baseline)")
        results["D_sq00pp_vs_dq"][l][proj] = compare_scales(
            sq00_pp, dq_pp, name=f"SQ00_{proj} vs DQ_{proj}"
        )

        print(f"  [C]  SQ α=0.5 shared   vs DQ                | {proj}")
        results["C_sq05share_vs_dq"][l][proj] = compare_scales(
            sq05_share, dq_pp, name=f"SQ05_shared vs DQ_{proj}"
        )

        print(f"  [C0] SQ α=0   shared   vs DQ                | {proj}  (weight-only shared baseline)")
        results["C0_sq00share_vs_dq"][l][proj] = compare_scales(
            sq00_share, dq_pp, name=f"SQ00_shared vs DQ_{proj}"
        )

        print(f"  [E]  SQ α=0.5 per-proj vs SQ α=0 per-proj   | {proj}  (activation-signal contribution)")
        results["E_sq05pp_vs_sq00pp"][l][proj] = compare_scales(
            sq05_pp, sq00_pp, name=f"SQ05_{proj} vs SQ00_{proj}"
        )


# ── Summary: mean Pearson / Log-Pearson / Spearman across layers ────────────
def safe_mean(table, proj, key):
    vals = [table[l][proj][key] for l in table
            if proj in table[l] and table[l][proj] is not None]
    return float(np.mean(vals)) if vals else float('nan')

print("\n\n" + "=" * 100)
print(f"SUMMARY (mean over {NUM_LAYERS} layers) — Pearson / Log-Pearson / Spearman")
print("=" * 100)
print(f"{'Projection':<10} | "
      f"{'[B] SQ.5 pp vs DQ':^28} | "
      f"{'[D] SQ.0 pp vs DQ':^28} | "
      f"{'[E] SQ.5 vs SQ.0 pp':^28}")
print(f"{'':10} | {'P':>8} {'logP':>8} {'S':>8}   | "
      f"{'P':>8} {'logP':>8} {'S':>8}   | "
      f"{'P':>8} {'logP':>8} {'S':>8}")
print("-" * 100)
for proj in proj_map:
    row = []
    for tbl_key in ("B_sq05pp_vs_dq", "D_sq00pp_vs_dq", "E_sq05pp_vs_sq00pp"):
        tbl = results[tbl_key]
        row += [safe_mean(tbl, proj, "pearson_raw"),
                safe_mean(tbl, proj, "pearson_log"),
                safe_mean(tbl, proj, "spearman_raw")]
    print(f"{proj:<10} | "
          f"{row[0]:>+8.4f} {row[1]:>+8.4f} {row[2]:>+8.4f}   | "
          f"{row[3]:>+8.4f} {row[4]:>+8.4f} {row[5]:>+8.4f}   | "
          f"{row[6]:>+8.4f} {row[7]:>+8.4f} {row[8]:>+8.4f}")

print("\nShared-vs-DQ controls:")
print(f"{'Projection':<10} | "
      f"{'[C] SQ.5 share vs DQ':^28} | "
      f"{'[C0] SQ.0 share vs DQ':^28}")
print(f"{'':10} | {'P':>8} {'logP':>8} {'S':>8}   | "
      f"{'P':>8} {'logP':>8} {'S':>8}")
print("-" * 70)
for proj in proj_map:
    row = []
    for tbl_key in ("C_sq05share_vs_dq", "C0_sq00share_vs_dq"):
        tbl = results[tbl_key]
        row += [safe_mean(tbl, proj, "pearson_raw"),
                safe_mean(tbl, proj, "pearson_log"),
                safe_mean(tbl, proj, "spearman_raw")]
    print(f"{proj:<10} | "
          f"{row[0]:>+8.4f} {row[1]:>+8.4f} {row[2]:>+8.4f}   | "
          f"{row[3]:>+8.4f} {row[4]:>+8.4f} {row[5]:>+8.4f}")


# ── PDF output: color-coded heatmaps, one page per metric ────────────────────
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_COMPARISONS = [
    ("[B] SQ.5pp vs DQ",       "B_sq05pp_vs_dq"),
    ("[D] SQ.0pp vs DQ",       "D_sq00pp_vs_dq"),
    ("[E] SQ.5pp vs SQ.0pp",   "E_sq05pp_vs_sq00pp"),
    ("[C] SQ.5share vs DQ",    "C_sq05share_vs_dq"),
    ("[C0] SQ.0share vs DQ",   "C0_sq00share_vs_dq"),
]
_METRICS = [
    ("Pearson",     "pearson_raw"),
    ("Log-Pearson", "pearson_log"),
    ("Spearman",    "spearman_raw"),
]


def _build_matrix(metric_key):
    """Return (5 projections × 5 comparisons) np.ndarray of mean correlations."""
    M = np.zeros((len(proj_map), len(_COMPARISONS)))
    for i, proj in enumerate(proj_map):
        for j, (_, tbl_key) in enumerate(_COMPARISONS):
            M[i, j] = safe_mean(results[tbl_key], proj, metric_key)
    return M


_OUT_PDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"summary_tables_{MODEL}.pdf",
)

with PdfPages(_OUT_PDF) as pdf:
    for metric_label, metric_key in _METRICS:
        mat = _build_matrix(metric_key)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        # RdBu_r diverges around 0; correlation lives in [-1, 1].
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        # Annotate each cell with its value.
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                # White text on dark cells, black on light — based on |value|.
                color = "white" if abs(v) > 0.55 else "black"
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        color=color, fontsize=10)
        ax.set_xticks(range(len(_COMPARISONS)))
        ax.set_xticklabels([c[0] for c in _COMPARISONS],
                           rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(len(proj_map)))
        ax.set_yticklabels(list(proj_map), fontsize=10)
        ax.set_title(f"{MODEL}  —  {metric_label}  (mean over {NUM_LAYERS} layers)",
                     fontsize=11, pad=12)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("correlation", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"\nWrote heatmap PDF: {_OUT_PDF}")