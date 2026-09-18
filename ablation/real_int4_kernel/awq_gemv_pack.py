"""Weight repacking for nunchaku's bundled AWQ/TensorRT-LLM-derived GEMV kernel
(src/kernels/awq/gemv_awq.cu), used as a fast M<=4 decode path alongside the
main W4A4 CUTLASS+CUDA-graph path (which stays the path for M>4, i.e. batched
decode/prefill).

Why this kernel, and why only for tiny M:
  - It's weight-only INT4 (activations stay bf16) -- no activation-quantize
    kernel, unlike the W4A4 path. Measured on real Llama-3.1-8B projection
    shapes: up to 4.5x faster than BF16 on the MLP shapes (gate/up/down_proj),
    vs W4A4's 1.1-1.7x there -- avoiding the activation-quantize kernel matters
    most exactly where that kernel's cost (proportional to M*K) is largest.
  - It only supports M in [1,4] in practice (M in [5,7] crashes with an
    illegal-memory-access in this build, despite the kernel's own `assert(m>0
    && m<8)` -- that assert is compiled out under -DNDEBUG anyway; the safe
    range was found empirically, not from the assert).

Packing format reference: this kernel is derived from the same TensorRT-LLM
"weightOnlyBatchedGemv" lineage as AutoAWQ's WQLinear_GEMVFast (interleave=4,
kstride=64 match exactly). AutoAWQ's own `pack_intweight` (awq/modules/linear/
gemv_fast.py) packs 4 nibbles per int16; this kernel reads PACK_FACTOR=8
nibbles per int32 -- same total bytes, so the int16-packed result is
bit-reinterpreted (NOT numerically cast) into int32, preserving byte order.
Validated against a plain-matmul reference (SQNR ~50dB, i.e. bf16-rounding-only
noise) at real Llama-3.1-8B projection shapes before use here.

Beta folding: gemv_awq computes y = x_bf16 @ dequant(W).T with NO activation
rescale of its own. DualQuant's beta rescales activations column-wise
(X~ = X * beta), and by linearity (X*beta) @ W.T = X @ (beta*W).T -- so instead
of multiplying the activation by beta at every call (an extra kernel launch,
exactly the overhead this path exists to avoid), beta is folded into the
weight ONCE at conversion time: quantize (mat_q * beta[None, :]) instead of
mat_q alone. Runtime calls need no activation-side beta multiply at all.
"""
import torch

GROUP_SIZE = 64
ZERO_POINT = 8.0  # symmetric int4 stored as unsigned nibbles biased by 8


def pack_intweight(unpacked_qweight: torch.Tensor, interleave: int = 4, kstride: int = 64) -> torch.Tensor:
    """unpacked_qweight: [N, K] int, unsigned nibble values in [0, 15].
    Returns int32 [N // interleave, K // 2], nunchaku gemv_awq's expected layout."""
    N, K = unpacked_qweight.shape
    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, interleave, K // kstride, kstride)
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K // kstride, kstride, interleave)
    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight_i16 = torch.tensor(Packed_Kernel.astype("int16")).contiguous()
    return qweight_i16.view(torch.int32)


def quantize_symmetric_int4(W: torch.Tensor, group_size: int = GROUP_SIZE):
    """W: [N, K] float (cpu or cuda). Symmetric per-group int4 quantization,
    matching DualQuant's own convention (no zero-point in the real weight
    values -- the AWQ kernel's zero-point is used purely as the standard
    signed-to-unsigned nibble bias of 8, not an actual asymmetric offset).
    Returns (unsigned_nibbles [N,K] int32, scale [N, K//group_size] float32)."""
    N, K = W.shape
    Wg = W.reshape(N, K // group_size, group_size)
    scale = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.round(Wg / scale).clamp(-8, 7)
    unsigned_q = (q + ZERO_POINT).to(torch.int32).reshape(N, K)
    return unsigned_q, scale.squeeze(-1)


def build_awq_gemv_weights(mat_q: torch.Tensor, beta: torch.Tensor, device, group_size: int = GROUP_SIZE):
    """mat_q: [N, K] bf16/float (DualQuant's quantized-domain weight, same
    tensor fed to nunchaku's load_weight for the W4A4 path). beta: [K].
    Returns (qweight_int32, scales_bf16, scaled_zeros_bf16) ready for
    nunchaku_min.gemv_awq(x, qweight, scales, scaled_zeros, m, n, k, group_size)
    with beta ALREADY folded in -- call gemv_awq directly on raw x, no
    activation-side multiply needed."""
    N, K = mat_q.shape
    assert K % group_size == 0
    W = (mat_q.detach().to(torch.float32).cpu() * beta.detach().to(torch.float32).cpu()[None, :])
    unsigned_q, scale = quantize_symmetric_int4(W, group_size=group_size)
    qweight = pack_intweight(unsigned_q, interleave=4, kstride=64).to(device)
    scales_bf16 = scale.to(torch.bfloat16).T.contiguous().to(device)               # [K//G, N]
    scaled_zeros_bf16 = (-(scale * ZERO_POINT)).to(torch.bfloat16).T.contiguous().to(device)  # [K//G, N]
    return qweight, scales_bf16, scaled_zeros_bf16
