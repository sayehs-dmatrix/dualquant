"""Real A4W4 Triton kernel: INT4 weight (packed) x INT4-quantised activation,
with beta fused in, activation quantisation fused in-kernel.

History of this file (kept for the rebuttal record):
  v1: unpacked/dequantised BOTH operands back to float and ran `tl.dot` in
      floating point -- never touched an integer tensor core, and paid pure
      unpack/dequant overhead on top. Measured SLOWER than plain bf16 e2e.
  v2: the core dot product became a real INT8 x INT8 -> INT32 tensor-core
      matmul (int4 values are a strict subset of int8; Triton 3.2 has no
      public int4 MMA path -- `tl.dot` only supports
      {int8, fp8_e5m2, fp16, bf16, fp32} operands), scales applied as a
      post-matmul epilogue. Real speedup on large (MLP) shapes, but small-N
      shapes (k/v_proj, N=1024) stayed ~3x SLOWER than bf16 even after
      widening the BLOCK_N autotune search -- profiling explained why:
      decode is M=1, i.e. literally a GEMV. For N=1024 the (grid_m, grid_n)
      tiling launches only a handful of thread-blocks total, so almost all
      128 SMs on the GPU sit idle for the whole K=4096 reduction. cuBLAS
      wins here because M=1 GEMV has a dedicated, highly-parallel code path
      that cuBLAS/cuBLASLt selects automatically; our generic tiled GEMM
      kernel has no equivalent.
  v3 (this file): adds SPLIT_K. The K dimension is additionally split
      across a 3rd grid axis, so many thread-blocks cooperate on the same
      (M,N) output tile via atomic_add into an fp32 accumulator buffer
      (cast to bf16 once at the end) -- the standard "split-K GEMM" fix for
      exactly this GPU-under-utilization regime. SPLIT_K joins BLOCK_M/
      BLOCK_N in the per-shape autotune search, so it only activates where
      it actually helps (tiny grids) and autotune falls back to SPLIT_K=1
      (plain single-writer store, no atomics) where the grid is already
      large enough (e.g. gate_proj/down_proj).

y = quant4(x * beta) @ dequant(W_int4, scale1)^T
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


_BLOCK_K = 64   # == GROUP_SIZE (one BCD scale group per K-tile iteration; not autotunable)

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "SPLIT_K": sk}, num_warps=4, num_stages=2)
    for bm in (16, 32)
    for bn in (64, 128, 256)
    for sk in (1, 4, 8, 16, 32)
    if bm * bn <= 65536
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"], reset_to_zero=["Y_ptr"])
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
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_split = tl.cdiv(num_k_tiles, SPLIT_K)
    k_tile_lo = pid_k * tiles_per_split
    k_tile_hi = tl.minimum(k_tile_lo + tiles_per_split, num_k_tiles)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(k_tile_lo, k_tile_hi):
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

        # ── Quantise activation to INT4 range, stored as int8 (per-row, per-K-tile scale) ──
        a_absmax = tl.max(tl.abs(a), axis=1)              # [BLOCK_M]
        a_scale = a_absmax / 7.0
        a_scale = tl.where(a_absmax == 0, 1.0, a_scale)
        a_q = libdevice.round(a / a_scale[:, None])
        a_q = tl.minimum(tl.maximum(a_q, -8.0), 7.0).to(tl.int8)

        # ── Load packed INT4 weight, unpack to int8 (NOT dequantised) ──
        w_packed = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_packed[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_packed[None, :] < K // 2),
            other=0,
        )
        w_i8 = (((w_packed >> q_shift[None, :]) & 0xF).to(tl.int32) - 8).to(tl.int8)
        w_scale = tl.load(
            S_ptr + offs_n * stride_sn + k_tile * stride_sg,
            mask=offs_n < N,
            other=1.0,
        ).to(tl.float32)

        # ── REAL int8 tensor-core matmul (int32 accumulate) for this K-group ──
        partial = tl.dot(a_q, tl.trans(w_i8), out_dtype=tl.int32)

        # ── Epilogue: de-scale this group's exact int32 partial sum, sum in fp32 ──
        acc += partial.to(tl.float32) * (a_scale[:, None] * w_scale[None, :])

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=out_mask)


def fused_w4a4_gemm(x, beta, w_packed, scale1, group_size=32):
    """y = quant4(x * beta) @ dequant(W_int4, scale1)^T, fully fused.
    BLOCK_M/BLOCK_N/SPLIT_K are autotuned per (M, N, K) shape; BLOCK_K is
    fixed == group_size (one BCD scale group per K-tile iteration).
    SPLIT_K > 1 parallelises the K-reduction across extra thread-blocks
    (atomic_add into an fp32 buffer) -- needed when (M, N) alone doesn't
    launch enough thread-blocks to use the GPU (e.g. decode, M=1)."""
    assert x.dtype == torch.bfloat16 and beta.dtype == torch.bfloat16
    assert w_packed.dtype == torch.uint8
    assert group_size == _BLOCK_K

    M, K = x.shape
    N = w_packed.shape[0]

    x, beta, w_packed, scale1 = (t.contiguous() for t in (x, beta, w_packed, scale1))
    y_f32 = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]), META["SPLIT_K"],
    )

    _kernel_w4a4[grid](
        x, beta, w_packed, scale1, y_f32,
        M, N, K,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scale1.stride(0), scale1.stride(1),
        y_f32.stride(0), y_f32.stride(1),
        GROUP_SIZE=group_size, BLOCK_K=_BLOCK_K,
    )
    # NOTE: returns fp32, not bf16 -- callers (e.g. RealW4A4Linear.forward)
    # already cast the result to their target dtype, so casting here too
    # would just be a wasted bf16->fp32->bf16 round trip on every call.
    return y_f32
