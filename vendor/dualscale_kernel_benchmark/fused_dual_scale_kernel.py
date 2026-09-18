"""
fused_dual_scale_kernel.py
===========================
Triton kernel that fuses the per-column β activation scale into the
INT4 dequant + GEMM in a single kernel pass.

Problem
-------
DualQuant inference forward pass:
    y = (x · β)  @  dequant(W_int4, scale1)ᵀ

Naive (unfused) execution:
    x_scaled = x * β          ← separate kernel launch  (~5 μs overhead)
    y        = int4_gemm(x_scaled, W)

Fused execution (this file):
    β is loaded as a [K] vector and multiplied in-register while the
    activation tile already sits in registers — zero extra global memory
    traffic, zero extra kernel launch.

Quantization scheme
-------------------
Symmetric INT4, no zero-point:
    W_float[n, k] = (W_q[n, k] - 8) * scale1[n, k // GROUP_SIZE]
    W_q ∈ [0, 15], center = 8  →  W_float ∈ [-8, 7] × scale1

Packing (2 INT4s per uint8 byte, same convention as GemLite)
------------------------------------------------------------
    W_packed[n, k//2] lower nibble = W_q[n, even k]
    W_packed[n, k//2] upper nibble = W_q[n, odd  k]

    Unpack (following GemLite gemm_kernels.py):
        q_shift = (k % 2) * 4       →  0 for even k, 4 for odd k
        W_q[n, k] = (W_packed[n, k//2] >> q_shift) & 0xF

Constraints (simple kernel — extend for production)
----------------------------------------------------
    • BLOCK_K must equal GROUP_SIZE  (one group scale per K-tile)
    • K must be divisible by GROUP_SIZE
    • BF16 activations, BF16 output
    • Symmetric quantization only

Requirements:  pip install triton torch
"""

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════════════
# Triton kernel
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _kernel_fused_beta_int4_gemm(
    # ── Tensor pointers ───────────────────────────────────────────────────────
    X_ptr,      # [M, K]               BF16 activations
    Beta_ptr,   # [K]                  BF16 per-column β scale
    W_ptr,      # [N, K//2]            uint8 packed INT4 weights
    S_ptr,      # [N, K//GROUP_SIZE]   BF16 group dequant scales
    Y_ptr,      # [M, N]               BF16 output
    # ── Problem dimensions ────────────────────────────────────────────────────
    M, N, K,
    # ── Strides ───────────────────────────────────────────────────────────────
    stride_xm, stride_xk,
    stride_wn, stride_wk,   # W strides in packed-byte units
    stride_sn, stride_sg,   # S strides over [N, num_groups]
    stride_ym, stride_yn,
    # ── Compile-time constants ────────────────────────────────────────────────
    GROUP_SIZE: tl.constexpr,   # weight group size; must equal BLOCK_K
    BLOCK_M:    tl.constexpr,
    BLOCK_N:    tl.constexpr,
    BLOCK_K:    tl.constexpr,   # must equal GROUP_SIZE
):
    """
    Each program computes one [BLOCK_M, BLOCK_N] tile of Y.

    K-loop: BLOCK_K = GROUP_SIZE → one group scale per iteration.

    β fusion: after loading the X tile [BLOCK_M, BLOCK_K] into registers,
    multiply by β[BLOCK_K] in-register before the dot product.
    No extra global memory reads — β is a [K] vector that fits in cache
    after the first tile access.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                      # [BLOCK_K] local K

    # FP32 accumulator for numerical accuracy across tiles
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── K-tile loop ───────────────────────────────────────────────────────────
    for k_tile in range(tl.cdiv(K, BLOCK_K)):

        k_abs    = k_tile * BLOCK_K + offs_k   # [BLOCK_K] absolute K indices
        k_packed = k_abs // 2                  # [BLOCK_K] packed byte index (0,0,1,1,...)
        q_shift  = (k_abs % 2 * 4).to(tl.int32)  # [BLOCK_K] nibble shift: 0 or 4

        # ── Step 1: Load X tile ───────────────────────────────────────────────
        a = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_abs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_abs[None, :] < K),
            other=0.0,
        ).to(tl.float32)   # [BLOCK_M, BLOCK_K]

        # ── Step 2: Load β and apply in-register  ←  the fusion ──────────────
        # β is [K]; we load the BLOCK_K slice for this tile.
        # The multiply happens in registers while 'a' is already loaded —
        # no extra global memory round-trip, no separate kernel launch.
        beta = tl.load(Beta_ptr + k_abs, mask=k_abs < K, other=1.0).to(tl.float32)
        a *= beta[None, :]   # [BLOCK_M, BLOCK_K]

        # ── Step 3: Load packed INT4 weights and unpack ───────────────────────
        # k_packed maps each K index to its packed-byte position.
        # The same packed byte is loaded for two adjacent K values;
        # q_shift selects the correct nibble (lo=0 for even k, hi=4 for odd k).
        # This follows GemLite's nibble extraction pattern exactly.
        w_packed = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_packed[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_packed[None, :] < K // 2),
            other=0,
        )   # uint8 [BLOCK_N, BLOCK_K] — duplicate bytes for even/odd pairs

        w_int4 = (w_packed >> q_shift[None, :]) & 0xF   # [BLOCK_N, BLOCK_K]

        # ── Step 4: Dequantize ────────────────────────────────────────────────
        # One scale per output channel per K-tile (BLOCK_K == GROUP_SIZE).
        scale = tl.load(
            S_ptr + offs_n * stride_sn + k_tile * stride_sg,
            mask=offs_n < N,
            other=1.0,
        ).to(tl.float32)   # [BLOCK_N]

        w_fp = (w_int4.to(tl.float32) - 8.0) * scale[:, None]   # [BLOCK_N, BLOCK_K]

        # ── Step 5: Accumulate ────────────────────────────────────────────────
        # a [BLOCK_M, BLOCK_K] @ w_fp.T [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, tl.trans(w_fp))

    # ── Store output ──────────────────────────────────────────────────────────
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pack / unpack utilities
# ══════════════════════════════════════════════════════════════════════════════

def pack_int4_weights(W: torch.Tensor, group_size: int = 32):
    """
    Quantize a float weight matrix to symmetric INT4 and pack 2 values/byte.

    Args:
        W          : [N, K] float32 or bfloat16 weight matrix
        group_size : elements per quantization group

    Returns:
        w_packed : [N, K//2] uint8  — lo nibble = even K, hi nibble = odd K
        scale1   : [N, K//group_size] bfloat16 — per-group dequant scales
    """
    N, K = W.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 2 == 0, "K must be even for INT4 packing"

    W_f32 = W.float()

    # Per-group max-abs scale: maps ±max → ±7
    W_groups = W_f32.view(N, K // group_size, group_size)
    max_abs  = W_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale    = max_abs / 7.0   # [N, K//group_size, 1]

    # Quantize: W_float → INT4 ∈ [-8, 7], stored as unsigned [0, 15]
    W_q = (W_groups / scale).round().clamp(-8, 7).to(torch.int8) + 8
    W_q = W_q.to(torch.uint8).view(N, K)   # [N, K] uint8 values in [0, 15]

    # Pack: lo nibble = even K column, hi nibble = odd K column
    w_lo = W_q[:, 0::2] & 0xF                   # [N, K//2]
    w_hi = (W_q[:, 1::2] & 0xF) << 4            # [N, K//2]
    w_packed = (w_lo | w_hi).contiguous()        # [N, K//2] uint8

    scale1 = scale.squeeze(-1).to(torch.bfloat16).contiguous()  # [N, K//group_size]
    return w_packed, scale1


def dequant_int4_reference(w_packed: torch.Tensor, scale1: torch.Tensor,
                           group_size: int = 32) -> torch.Tensor:
    """Dequantize packed INT4 weights back to float32 for correctness checks."""
    N, K_half = w_packed.shape
    K = K_half * 2
    w_lo = (w_packed & 0xF).float()
    w_hi = ((w_packed >> 4) & 0xF).float()
    W_q = torch.zeros(N, K, dtype=torch.float32, device=w_packed.device)
    W_q[:, 0::2] = w_lo
    W_q[:, 1::2] = w_hi
    # Dequantize: (W_q - 8) * scale
    num_groups = K // group_size
    W_q = W_q.view(N, num_groups, group_size)
    W_dq = (W_q - 8.0) * scale1.float().unsqueeze(-1)
    return W_dq.view(N, K)


# ══════════════════════════════════════════════════════════════════════════════
# Python wrapper
# ══════════════════════════════════════════════════════════════════════════════

# Fixed block sizes — simplest valid config for tl.dot:
#   BLOCK_M=16, BLOCK_N=32 (≥ 16), BLOCK_K=32 = GROUP_SIZE
_BLOCK_M = 16
_BLOCK_N = 32
_BLOCK_K = 32   # must equal GROUP_SIZE


def fused_beta_int4_gemm(
    x:        torch.Tensor,   # [M, K] BF16
    beta:     torch.Tensor,   # [K]    BF16
    w_packed: torch.Tensor,   # [N, K//2] uint8
    scale1:   torch.Tensor,   # [N, K//GROUP_SIZE] BF16
    group_size: int = 32,
) -> torch.Tensor:
    """
    Compute Y = (X · β) @ dequant(W_int4, scale1)ᵀ  with β fused in-kernel.

    β is applied to X in-register during the activation tile load,
    eliminating the separate x * β kernel launch (~5 μs overhead).
    """
    assert x.dtype == torch.bfloat16 and beta.dtype == torch.bfloat16
    assert w_packed.dtype == torch.uint8
    assert group_size == _BLOCK_K, \
        f"This kernel requires group_size == BLOCK_K == {_BLOCK_K}, got {group_size}"

    M, K = x.shape
    N    = w_packed.shape[0]
    assert K % group_size == 0

    x        = x.contiguous()
    beta     = beta.contiguous()
    w_packed = w_packed.contiguous()
    scale1   = scale1.contiguous()

    y    = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))

    _kernel_fused_beta_int4_gemm[grid](
        x, beta, w_packed, scale1, y,
        M, N, K,
        x.stride(0),        x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scale1.stride(0),   scale1.stride(1),
        y.stride(0),        y.stride(1),
        GROUP_SIZE = group_size,
        BLOCK_M    = _BLOCK_M,
        BLOCK_N    = _BLOCK_N,
        BLOCK_K    = _BLOCK_K,
    )
    return y


# ══════════════════════════════════════════════════════════════════════════════
# Correctness check
# ══════════════════════════════════════════════════════════════════════════════

def check_correctness():
    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)

    torch.manual_seed(42)
    device     = "cuda"
    GROUP_SIZE = 32

    configs = [
        (4,   64,  128, "tiny"),
        (16,  256, 256, "small"),
        (1,  4096, 4096, "BS=1  decode"),
        (128, 4096, 4096, "BS=128 prefill"),
    ]

    for M, N, K, label in configs:
        W      = torch.randn(N, K,  device=device)
        x      = torch.randn(M, K,  device=device, dtype=torch.bfloat16)
        beta   = torch.rand( K,     device=device, dtype=torch.bfloat16) * 0.4 + 0.8

        w_packed, scale1 = pack_int4_weights(W, GROUP_SIZE)
        w_packed = w_packed.to(device)
        scale1   = scale1.to(device)

        # Reference: unfused
        W_dq  = dequant_int4_reference(w_packed, scale1, GROUP_SIZE)
        y_ref = (x.float() * beta.float()) @ W_dq.T

        # Fused kernel
        y_fused = fused_beta_int4_gemm(x, beta, w_packed, scale1, GROUP_SIZE)

        max_err = (y_fused.float() - y_ref).abs().max().item()
        rel_err = ((y_fused.float() - y_ref).abs() /
                   (y_ref.abs() + 1e-6)).mean().item()
        status  = "PASS" if rel_err < 0.01 else "FAIL"   # 1% relative tolerance
        print(f"  [{status}] {label:<18}  M={M:4d} N={N:5d} K={K:5d}"
              f"  max_abs={max_err:.3e}  mean_rel={rel_err:.3e}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark: fused vs unfused β overhead
# ══════════════════════════════════════════════════════════════════════════════

def cuda_timer(fn, warmup=50, repeats=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(repeats):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / repeats * 1000   # ms → μs


def benchmark():
    """
    Compare:
      unfused : x_scaled = x * β  (separate kernel)  +  fused_kernel(x_scaled, β=1)
      fused   :                                           fused_kernel(x, β)

    Both paths use our INT4 kernel — the only difference is whether β is
    applied externally (extra kernel launch) or inside the kernel (free).
    """
    device     = "cuda"
    GROUP_SIZE = 32
    BATCH_SIZES = [1, 16, 128, 512, 2048]

    LAYERS = [
        (8960,  1536, "Qwen2-1.5B",  "gate_proj"),
        (256,   1536, "Qwen2-1.5B",  "k_proj   "),
        (14336, 4096, "Llama3.1-8B", "gate_proj"),
        (1024,  4096, "Llama3.1-8B", "k_proj   "),
    ]

    # ── Collect all timings first ─────────────────────────────────────────────
    results = []
    ones_cache = {}

    print("  Running benchmark ... ", flush=True)
    for N, K, model, layer in LAYERS:
        W = torch.randn(N, K, device=device)
        w_packed, scale1 = pack_int4_weights(W, GROUP_SIZE)
        if K not in ones_cache:
            ones_cache[K] = torch.ones(K, device=device, dtype=torch.bfloat16)
        beta_ones = ones_cache[K]

        for bs in BATCH_SIZES:
            x    = torch.randn(bs, K, device=device, dtype=torch.bfloat16)
            beta = torch.rand( K,    device=device, dtype=torch.bfloat16) * 0.4 + 0.8

            def unfused():
                return fused_beta_int4_gemm(x * beta, beta_ones, w_packed, scale1, GROUP_SIZE)

            def fused():
                return fused_beta_int4_gemm(x, beta, w_packed, scale1, GROUP_SIZE)

            t_fused = cuda_timer(fused)
            t_unf   = cuda_timer(unfused)

            results.append(dict(
                model=model, layer=layer, N=N, K=K, bs=bs,
                t_fused=t_fused, t_unf=t_unf,
                delta_us=t_unf - t_fused,
                overhead_pct=(t_unf - t_fused) / t_fused * 100.0,
            ))

    # ── Print table ───────────────────────────────────────────────────────────
    W  = 80
    print()
    print("═" * W)
    print("  Fused β  vs  Unfused β  —  INT4 GEMM, BF16 activations")
    print("  Fused  : β applied IN-REGISTER inside the INT4 kernel (no extra launch)")
    print("  Unfused: x * β as a separate CUDA kernel, then INT4 matmul")
    print("═" * W)

    # Column layout: Model | Layer | N×K | BS | Fused μs | Unfused μs | Δ μs | Δ%
    hdr = (f"  {'Model':<13} {'Layer':<11} {'N×K':<14}"
           f" {'BS':>5}  {'Fused':>9}  {'Unfused':>9}  {'β cost':>7}  {'β cost':>7}")
    sub = (f"  {'':13} {'':11} {'':14}"
           f" {'':>5}  {'(μs)':>9}  {'(μs)':>9}  {'(μs)':>7}  {'(%)':>7}")
    sep = "  " + "─" * (len(hdr) - 2)

    prev_model = None
    for r in results:
        # Print header at start and between models
        if r["model"] != prev_model:
            print()
            print(hdr)
            print(sub)
            print(sep)
            prev_model = r["model"]

        # Indent layer/shape only on first BS row for this layer
        is_first_bs = (r["bs"] == BATCH_SIZES[0])
        model_str = r["model"]  if is_first_bs else ""
        layer_str = r["layer"]  if is_first_bs else ""
        nk_str    = f"{r['N']}×{r['K']}" if is_first_bs else ""

        # Colour-code the overhead: < 2% looks clean, 2-5% caution, > 5% notable
        pct = r["overhead_pct"]
        if   pct < 2.0:  marker = " ✓"
        elif pct < 5.0:  marker = " ~"
        else:            marker = " !"

        print(
            f"  {model_str:<13} {layer_str:<11} {nk_str:<14}"
            f" {r['bs']:>5}  {r['t_fused']:>9.1f}  {r['t_unf']:>9.1f}"
            f"  {r['delta_us']:>6.1f}μs  {pct:>6.1f}%{marker}"
        )

        # Blank line after last BS of each layer
        if r["bs"] == BATCH_SIZES[-1]:
            print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("═" * W)
    print("  SUMMARY  —  average β overhead when unfused vs fused")
    print()
    print(f"  {'Batch':>5}  {'avg Fused (μs)':>15}  {'avg Unfused (μs)':>17}"
          f"  {'avg β cost (μs)':>16}  {'avg β cost (%)':>14}")
    print("  " + "─" * 76)
    for bs in BATCH_SIZES:
        rows = [r for r in results if r["bs"] == bs]
        avg_f   = sum(r["t_fused"]      for r in rows) / len(rows)
        avg_u   = sum(r["t_unf"]        for r in rows) / len(rows)
        avg_d   = sum(r["delta_us"]     for r in rows) / len(rows)
        avg_pct = (avg_u - avg_f) / avg_f * 100.0
        print(f"  {bs:>5}  {avg_f:>15.1f}  {avg_u:>17.1f}"
              f"  {avg_d:>16.2f}  {avg_pct:>13.1f}%")

    print()
    print("  Legend:  ✓ < 2%  (negligible)   ~ 2–5%  (small)   ! > 5%  (notable)")
    print("  β cost = time added by NOT fusing β  (= unfused − fused)")
    print("═" * W)
    print()

    # ── Export PDF table and LaTeX numbers ────────────────────────────────────
    save_pdf_table(results, BATCH_SIZES, LAYERS)


# ══════════════════════════════════════════════════════════════════════════════
# PDF table + LaTeX numbers export
# ══════════════════════════════════════════════════════════════════════════════

def save_pdf_table(results: list, batch_sizes: list, layers: list) -> None:
    """
    Write two files into the same directory as this script:

      overhead_table.pdf         — publication-ready table figure
                                   \\includegraphics[width=\\linewidth]{overhead_table}
      overhead_table_numbers.tex — tabular body rows, paste/\\input into LaTeX table

    The PDF table mirrors Table 1 in inference_overhead.tex:
      rows    = (Model, Layer, N×K) + Average row
      columns = Δβ (%) for each batch size
    """
    import os

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except ImportError:
        print("  [save_pdf_table] matplotlib not found — skipping PDF export.")
        return

    # ── Organise data ──────────────────────────────────────────────────────────
    layer_meta = []   # (model_str, layer_str, "N×K")
    pct_matrix = []   # pct_matrix[layer_idx][bs_idx]

    for N, K, model, layer in layers:
        m = model.strip()
        l = layer.strip()
        layer_meta.append((m, l, f"{N}×{K}"))
        row = []
        for bs in batch_sizes:
            r = next(
                x for x in results
                if x["model"].strip() == m and x["layer"].strip() == l and x["bs"] == bs
            )
            row.append(r["overhead_pct"])
        pct_matrix.append(row)

    avg_row = [
        sum(pct_matrix[i][j] for i in range(len(layer_meta))) / len(layer_meta)
        for j in range(len(batch_sizes))
    ]

    # ── Build cell text ────────────────────────────────────────────────────────
    col_labels = ["Model", "Layer", "N×K"] + [f"B={bs}" for bs in batch_sizes]
    n_cols = len(col_labels)

    cell_text = []
    for i, (model, layer, nk) in enumerate(layer_meta):
        cell_text.append([model, layer, nk] + [f"{v:.1f}%" for v in pct_matrix[i]])
    cell_text.append(["", "Average", ""] + [f"{v:.1f}%" for v in avg_row])

    n_rows = len(cell_text)

    # ── Colours and markers ────────────────────────────────────────────────────
    HDR_BG   = "#2c3e50"   # dark navy
    HDR_FG   = "white"
    ROW_A    = "#f8f9fa"
    ROW_B    = "#ffffff"
    AVG_BG   = "#eaf4fb"
    CLR_GOOD = "#27ae60"   # green  < 2 %
    CLR_WARN = "#d68910"   # amber  2–5 %
    CLR_BAD  = "#c0392b"   # red    > 5 %

    def _pct_color(val: float) -> str:
        if val < 2.0:  return CLR_GOOD
        if val < 5.0:  return CLR_WARN
        return CLR_BAD

    # ── Draw figure ────────────────────────────────────────────────────────────
    rcParams["font.family"] = "DejaVu Sans"
    fig_w = 3.2 + len(batch_sizes) * 1.05
    fig_h = 0.48 * (n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.55)

    # header row
    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor(HDR_BG)
        cell.set_text_props(color=HDR_FG, fontweight="bold")
        cell.set_edgecolor("#444444")

    # data rows
    for i in range(n_rows):
        is_avg = (i == n_rows - 1)
        bg = AVG_BG if is_avg else (ROW_A if i % 2 == 0 else ROW_B)
        for j in range(n_cols):
            cell = tbl[(i + 1, j)]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#cccccc")
            if is_avg and j < 3:
                cell.set_text_props(fontstyle="italic", color="#555555")
            if j >= 3:                          # Δβ% columns
                try:
                    val = float(cell_text[i][j].replace("%", ""))
                    cell.set_text_props(color=_pct_color(val), fontweight="bold")
                except ValueError:
                    pass

    # top/bottom heavy rule (booktabs style)
    for j in range(n_cols):
        tbl[(0,      j)].visible_edges = "BT"
        tbl[(n_rows, j)].visible_edges = "B"

    # caption
    ax.set_title(
        r"$\Delta_\beta$ (%) overhead: unfused vs. fused $\beta$ scaling  "
        "| green < 2%  amber 2–5%  red > 5%",
        fontsize=8.5, pad=6, color="#333333",
    )

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(out_dir, "overhead_table.pdf")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  PDF table  → {pdf_path}")
    print(r"  LaTeX use  → \includegraphics[width=\linewidth]{overhead_table}")

    # ── LaTeX tabular body ─────────────────────────────────────────────────────
    tex_path = os.path.join(out_dir, "overhead_table_numbers.tex")
    with open(tex_path, "w") as fh:
        for i, (model, layer, nk) in enumerate(layer_meta):
            vals = " & ".join(f"{v:.1f}" for v in pct_matrix[i])
            # group separator before second model
            if i > 0 and layer_meta[i][0] != layer_meta[i - 1][0]:
                fh.write("\\addlinespace\n")
            nk_tex = nk.replace('×', r' \times ')
            fh.write(
                f"{model} & \\texttt{{{layer}}} & ${nk_tex}$ "
                f"& {vals} \\\\\n"
            )
        avg_vals = " & ".join(f"{v:.1f}" for v in avg_row)
        fh.write("\\midrule\n")
        fh.write(
            f"\\multicolumn{{3}}{{l}}{{\\emph{{Average across all layers}}}}"
            f" & {avg_vals} \\\\\n"
        )
    print(f"  LaTeX body → {tex_path}")
    print(r"  LaTeX use  → \input{overhead_table_numbers} inside your tabular")
    print()


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required."
    check_correctness()
    benchmark()
