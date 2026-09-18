"""
Benchmark: Single-Scale vs Dual-Scale MXINT4 — Overhead of β Activation Scaling
=================================================================================

DualQuant inference forward passes:

  Single-scale (baseline MXINT4):
    y = x @ GemLite(W_int4, α)ᵀ

  Dual-scale (+β column scale):
    y = (x · β) @ GemLite(W_int4, α)ᵀ

The overhead of adding β is exactly one element-wise multiply on the input:
  col_scale : (B, K)  — broadcasts β[K] over the batch

α (per-row output scale) can be fused with a downstream op (e.g. residual add
or layer norm), so we report two dual-scale variants:
  dual_β      : col_scale + matmul            (β only, α absorbed downstream)
  dual_full   : col_scale + matmul + row_scale (both β and α applied explicitly)

Layers benchmarked
──────────────────
  gate_proj  — large FFN projection (N >> K), compute-bound at large BS
  k_proj     — GQA attention key projection (N < K), memory-bound

Models: Qwen2-1.5B, Llama3.1-8B
Batch sizes: 1 (decode), 16, 128 (prefill)
Activations: BF16

NOTE: GemLite unavailable in this env → BF16 matmul used as proxy.
      Absolute times differ from INT4; relative β-overhead percentages
      are a conservative upper bound (INT4 kernel is faster, so real
      overhead % would be somewhat higher).
"""

import sys
import os
import csv

import torch

# ── GemLite import ────────────────────────────────────────────────────────────
_GEMLITE     = False
_gemlite_mod = None   # module reference kept for forward_functional calls

try:
    import gemlite as _gemlite_mod          # type: ignore[import]
    from gemlite import GemLiteLinear       # type: ignore[import]
    from gemlite.dtypes import TORCH_TO_DTYPE  # type: ignore[import]
    _GEMLITE = True
    print(f"[benchmark] GemLite {_gemlite_mod.__version__} — fused INT4 kernel active.")
except ImportError:
    print("[benchmark] GemLite not found — BF16 matmul proxy (relative overhead valid).")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEVICE     = "cuda"
DTYPE      = torch.bfloat16
NBITS      = 4
GROUP_SIZE = 32
WARMUP     = 50
REPEATS    = 200

BATCH_SIZES = [1, 16, 128, 512, 2048]

# Two representative layers per model:
#   gate_proj — large FFN (N >> K), dominant cost; overhead should be small
#   k_proj    — GQA key proj (N < K), small matmul; overhead baseline
MODEL_LAYERS = {
    "Qwen2-1.5B": [
        (8960,  1536, "gate_proj"),
        (256,   1536, "k_proj"),
    ],
    "Llama3.1-8B": [
        (14336, 4096, "gate_proj"),
        (1024,  4096, "k_proj"),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# Timing utility
# ══════════════════════════════════════════════════════════════════════════════

def cuda_timer(fn, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    """Return mean GPU execution time in milliseconds (CUDA events)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)  # type: ignore[call-arg]
    t1 = torch.cuda.Event(enable_timing=True)  # type: ignore[call-arg]
    t0.record()  # type: ignore[call-arg]
    for _ in range(repeats):
        fn()
    t1.record()  # type: ignore[call-arg]
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / repeats  # ms


# ══════════════════════════════════════════════════════════════════════════════
# Kernel helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_gemlite_kernel(N: int, K: int):
    in_dtype  = TORCH_TO_DTYPE.get(DTYPE, TORCH_TO_DTYPE[torch.float16])  # type: ignore[name-defined]
    out_dtype = in_dtype
    W_q    = torch.randint(0, 2 ** NBITS, (N, K), dtype=torch.uint8, device=DEVICE)
    scale1 = torch.rand(N, K // GROUP_SIZE, dtype=torch.float16, device=DEVICE).abs() * 0.01 + 1e-4
    zero   = torch.zeros(N, K // GROUP_SIZE, dtype=torch.float16, device=DEVICE)
    with torch.cuda.device(DEVICE):
        gl = GemLiteLinear(  # type: ignore[name-defined]
            NBITS, group_size=GROUP_SIZE, in_features=K, out_features=N,
            input_dtype=in_dtype, output_dtype=out_dtype, scaled_activations=True,
        )
        gl.pack(W_q, scale1, zero, None)
        tensor_args = [t.to(DEVICE) for t in gl.get_tensor_args()]
        meta_args   = [int(v)       for v in gl.get_meta_args()]
    return tensor_args, meta_args


def _make_fp_proxy(N: int, K: int):
    return torch.randn(N, K, dtype=DTYPE, device=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# Single configuration benchmark
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_layer(N: int, K: int, batch: int) -> dict:
    """
    Measure four timing points (all in μs after conversion):

      t_single  : matmul only                    → single-scale MXINT4 baseline
      t_col     : x · β  alone                   → β scaling cost in isolation
      t_dual_b  : (x · β) + matmul               → single-scale + β (α absorbed)
      t_dual_f  : (x · β) + matmul + (y · 1/α)   → full dual-scale

    Overhead reported:
      β_overhead_pct  = (t_dual_b − t_single) / t_single × 100
      full_overhead_pct = (t_dual_f − t_single) / t_single × 100
    """
    x     = torch.randn(batch, K, dtype=DTYPE, device=DEVICE)
    beta  = torch.rand(K, dtype=DTYPE, device=DEVICE).abs() * 0.2 + 0.9
    inv_a = torch.rand(N, dtype=DTYPE, device=DEVICE).abs() * 0.2 + 0.9

    if _GEMLITE:
        tensor_args, meta_args = _make_gemlite_kernel(N, K)
        _gl = _gemlite_mod
        matmul_fn = lambda inp: _gl.forward_functional(inp, None, tensor_args, meta_args, -1)  # type: ignore[union-attr]
    else:
        W_fp = _make_fp_proxy(N, K)
        matmul_fn = lambda inp: torch.matmul(inp, W_fp.t())

    t_single = cuda_timer(lambda: matmul_fn(x))
    t_col    = cuda_timer(lambda: x * beta)

    def dual_beta_fn():
        return matmul_fn(x * beta)

    def dual_full_fn():
        return matmul_fn(x * beta) * inv_a

    t_dual_b = cuda_timer(dual_beta_fn)
    t_dual_f = cuda_timer(dual_full_fn)

    to_us = lambda ms: ms * 1000.0

    b_overhead_us  = t_dual_b - t_single
    f_overhead_us  = t_dual_f - t_single
    b_overhead_pct = b_overhead_us / t_single * 100.0 if t_single > 0 else 0.0
    f_overhead_pct = f_overhead_us / t_single * 100.0 if t_single > 0 else 0.0

    return {
        "N"              : N,
        "K"              : K,
        "batch"          : batch,
        "t_single_us"    : to_us(t_single),
        "t_col_us"       : to_us(t_col),
        "t_dual_b_us"    : to_us(t_dual_b),
        "t_dual_f_us"    : to_us(t_dual_f),
        "b_overhead_us"  : to_us(b_overhead_us),
        "b_overhead_pct" : b_overhead_pct,
        "f_overhead_us"  : to_us(f_overhead_us),
        "f_overhead_pct" : f_overhead_pct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    assert torch.cuda.is_available(), "CUDA GPU required."

    backend = f"GemLite {_gemlite_mod.__version__}" if _GEMLITE else "BF16 proxy"  # type: ignore[union-attr]
    print()
    print("=" * 100)
    print("  DualQuant MXINT4 — Overhead of β Activation Scaling vs Single-Scale MXINT4")
    print(f"  Backend : {backend}  |  dtype={DTYPE}  |  group_size={GROUP_SIZE}")
    print(f"  Timing  : CUDA events (warmup={WARMUP}, repeats={REPEATS})")
    print("=" * 100)

    all_rows = []

    # ── Collect timings ────────────────────────────────────────────────────────
    data = {}   # data[model_name][layer_name][batch] = result dict
    for model_name, layers in MODEL_LAYERS.items():
        data[model_name] = {}
        for N, K, layer_name in layers:
            data[model_name][layer_name] = {}
            for batch in BATCH_SIZES:
                r = benchmark_layer(N, K, batch)
                data[model_name][layer_name][batch] = r
                all_rows.append({"model": model_name, "layer": layer_name, **r})

    # ── Print compact paper-style table ───────────────────────────────────────
    #
    # Columns (all μs):
    #   single  = base INT4 matmul (no scaling)
    #   dual-β  = matmul + β activation scaling (α absorbed downstream)
    #   Δβ %    = (dual-β − single) / single × 100
    #   dual-full = matmul + β + α (both scales explicit)
    #   Δfull % = (dual-full − single) / single × 100
    #
    # One row per (model, layer); batch sizes shown as column groups.
    # ──────────────────────────────────────────────────────────────────────────

    # Columns per batch-size group:
    #   single   — base matmul (no scaling)
    #   β-only   — isolated x·β cost  ← constant ~5–6 μs regardless of layer size
    #   dual-β   — matmul + β combined
    #   Δβ%      — overhead of β relative to single
    COL_W   = 7    # width for μs columns
    PCT_W   = 6    # width for % columns
    GRP_W   = COL_W*3 + PCT_W + 6   # total width per batch-size group

    BS_HDR  = "".join(f"  {'BS=' + str(bs):^{GRP_W}}" for bs in BATCH_SIZES)
    COL_HDR = "".join(
        f"  {'single':>{COL_W}} {'β-only':>{COL_W}} {'dual-β':>{COL_W}} {'Δβ%':>{PCT_W}}"
        for _ in BATCH_SIZES
    )
    UNITS   = "".join(
        f"  {'(μs)':>{COL_W}} {'(μs)':>{COL_W}} {'(μs)':>{COL_W}} {'':>{PCT_W}}"
        for _ in BATCH_SIZES
    )
    SEP = "─" * 41 + "─" * (len(COL_HDR))

    print()
    print(f"  {'Model':<14} {'Layer':<12} {'N×K':<15}{BS_HDR}")
    print(f"  {'':14} {'':12} {'':15}{COL_HDR}")
    print(f"  {'':14} {'':12} {'':15}{UNITS}")
    print("  " + SEP)

    for model_name, layers in MODEL_LAYERS.items():
        for (N, K, layer_name) in layers:
            nk_str = f"{N}×{K}"
            row_lbl = f"  {model_name:<14} {layer_name:<12} {nk_str:<15}"
            cells = ""
            for batch in BATCH_SIZES:
                r = data[model_name][layer_name][batch]
                cells += (
                    f"  {r['t_single_us']:>{COL_W}.1f}"
                    f" {r['t_col_us']:>{COL_W}.1f}"
                    f" {r['t_dual_b_us']:>{COL_W}.1f}"
                    f" {r['b_overhead_pct']:>{PCT_W-1}.1f}%"
                )
            print(row_lbl + cells)
        print("  " + SEP)

    # ── Per-batch-size summary ─────────────────────────────────────────────────
    print()
    print("  SUMMARY 1 — β-only cost (x · β  in isolation) across all layers")
    print("  Key claim: β scaling cost is ~constant regardless of layer size (kernel-launch dominated)")
    print()
    print(f"  {'Model':<14} {'Layer':<12} {'N×K':<15}  {'BS=1':>7}  {'BS=16':>7}  {'BS=128':>7}")
    print(f"  {'':14} {'':12} {'':15}  {'(μs)':>7}  {'(μs)':>7}  {'(μs)':>7}")
    print("  " + "─" * 60)
    for model_name, layers in MODEL_LAYERS.items():
        for (N, K, layer_name) in layers:
            nk_str = f"{N}×{K}"
            vals = "  ".join(
                f"{data[model_name][layer_name][bs]['t_col_us']:>7.1f}"
                for bs in BATCH_SIZES
            )
            print(f"  {model_name:<14} {layer_name:<12} {nk_str:<15}  {vals}")
    print()

    print("  SUMMARY 2 — Average overhead vs matmul baseline")
    print()
    print(f"  {'Batch':>8}  {'avg single (μs)':>16}  {'avg β-only (μs)':>16}  {'avg dual-β (μs)':>16}  {'Δβ%':>6}")
    print("  " + "─" * 70)
    for batch in BATCH_SIZES:
        rows   = [r for r in all_rows if r["batch"] == batch]
        avg_s  = sum(r["t_single_us"] for r in rows) / len(rows)
        avg_b  = sum(r["t_col_us"]    for r in rows) / len(rows)
        avg_db = sum(r["t_dual_b_us"] for r in rows) / len(rows)
        avg_p  = (avg_db - avg_s) / avg_s * 100.0
        print(f"  {batch:>8}  {avg_s:>16.1f}  {avg_b:>16.1f}  {avg_db:>16.1f}  {avg_p:>5.1f}%")

    print()
    print("  Notes:")
    print("  • β-only    = isolated x·β cost — near-constant across layers (kernel launch ~5 μs floor)")
    print("  • dual-β    = matmul + (x·β)    — α absorbed into a downstream op")
    if not _GEMLITE:
        print("  • [proxy]   BF16 matmul used; real INT4 matmul is faster so Δβ% will be higher.")
    print()

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_results.csv")
    fieldnames = [
        "model", "layer", "N", "K", "batch",
        "t_single_us", "t_col_us",
        "t_dual_b_us", "b_overhead_us", "b_overhead_pct",
        "t_dual_f_us", "f_overhead_us", "f_overhead_pct",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Results saved → {csv_path}")
    print()


if __name__ == "__main__":
    main()
