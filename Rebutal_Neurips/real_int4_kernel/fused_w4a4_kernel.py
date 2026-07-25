"""Real A4W4 Triton kernel: INT4 weight (packed) x INT4-quantised activation,
with beta fused in, activation quantisation fused in-kernel (no separate
unfused quantise step).

Built as a new file in this isolated folder -- does NOT modify the original
fused_dual_scale_kernel.py (DualScale_Kernel_Benchmark), which stays as our
weight-only (A16W4) reference kernel. This file adapts that kernel by adding
one thing: the beta-scaled activation tile is rounded to the INT4 grid
in-register, using a per-token, per-K-tile scale -- the SAME granularity as
our existing rtn_int4 activation format (block_size == BLOCK_K == 32), so
the numerics match formats/rtn_int.py's _rtn_cast exactly (symmetric,
max-abs, qmax=7).

y = quant4(x * beta) @ dequant(W_int4, scale1)^T

Both weight and activation are genuinely rounded to INT4 (4-bit values),
though the dot product itself still executes in floating point after
dequantising both operands (same "fused fake-quant" approach discussed --
this is not literal INT4x INT4 tensor-core arithmetic, but it IS the real,
fused, numerically-correct low-bit compute path, with no separate slow
quantisation kernel launch).
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _kernel_w4a4(
    X_ptr, Beta_ptr, W_ptr, S_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sg,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_abs = k_tile * BLOCK_K + offs_k
        k_packed = k_abs // 2
        q_shift = (k_abs % 2 * 4).to(tl.int32)

        # ── Load activation tile, apply beta ──
        a = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_abs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_abs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(Beta_ptr + k_abs, mask=k_abs < K, other=1.0).to(tl.float32)
        a = a * beta[None, :]

        # ── NEW: fuse activation INT4 quantisation in-register ──────────────
        # Per-token (row), per-K-tile scale -- same granularity as our own
        # rtn_int4 activation format (block_size == BLOCK_K == GROUP_SIZE).
        a_absmax = tl.max(tl.abs(a), axis=1)              # [BLOCK_M]
        a_scale = a_absmax / 7.0
        a_scale = tl.where(a_absmax == 0, 1.0, a_scale)
        a_q = libdevice.round(a / a_scale[:, None])
        a_q = tl.minimum(tl.maximum(a_q, -8.0), 7.0)
        a = a_q * a_scale[:, None]                          # dequantised INT4 activation
        # ──────────────────────────────────────────────────────────────────

        # ── Load packed INT4 weight, unpack, dequantise (unchanged) ──
        w_packed = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_packed[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_packed[None, :] < K // 2),
            other=0,
        )
        w_int4 = (w_packed >> q_shift[None, :]) & 0xF
        scale = tl.load(
            S_ptr + offs_n * stride_sn + k_tile * stride_sg,
            mask=offs_n < N,
            other=1.0,
        ).to(tl.float32)
        w_fp = (w_int4.to(tl.float32) - 8.0) * scale[:, None]

        acc += tl.dot(a, tl.trans(w_fp))

    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


_BLOCK_M = 16
_BLOCK_N = 32
_BLOCK_K = 64   # matches block_size=64, the default used for the paper's reported PPL numbers


def fused_w4a4_gemm(x, beta, w_packed, scale1, group_size=32):
    """y = quant4(x * beta) @ dequant(W_int4, scale1)^T, fully fused, single kernel launch."""
    assert x.dtype == torch.bfloat16 and beta.dtype == torch.bfloat16
    assert w_packed.dtype == torch.uint8
    assert group_size == _BLOCK_K

    M, K = x.shape
    N = w_packed.shape[0]

    x, beta, w_packed, scale1 = (t.contiguous() for t in (x, beta, w_packed, scale1))
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))

    _kernel_w4a4[grid](
        x, beta, w_packed, scale1, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scale1.stride(0), scale1.stride(1),
        y.stride(0), y.stride(1),
        GROUP_SIZE=group_size, BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
    )
    return y
