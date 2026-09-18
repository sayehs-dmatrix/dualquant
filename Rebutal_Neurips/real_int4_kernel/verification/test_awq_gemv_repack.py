import os
import sys
import torch
_nk = os.environ.get("NUNCHAKU_DIR")
if _nk: sys.path.insert(0, _nk)
import nunchaku_min

# Reference packer, lifted verbatim from autoawq (awq/modules/linear/gemv_fast.py) --
# this is the exact packer for the interleave=4, kstride=64 TensorRT-LLM-derived
# GEMV kernel that nunchaku's gemv_awq is also derived from (same lineage, same
# kInterleave/kStride constants).
def pack_intweight(unpacked_qweight, interleave, kstride):
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]
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
    # int16 (4 nibbles/interleave group packed per element), matching autoawq's
    # own storage. nunchaku's kernel reads PACK_FACTOR=8 nibbles per int32 --
    # same total bytes, just viewed at double width -- so reinterpret (NOT
    # numeric-cast) pairs of adjacent int16 into one int32, preserving byte order.
    qweight_i16 = torch.tensor(Packed_Kernel.astype("int16")).to(unpacked_qweight.device).contiguous()
    qweight = qweight_i16.view(torch.int32)
    return qweight


def quantize_symmetric_int4(W, group_size=64):
    """W: [N, K] float. Returns unsigned nibbles [N,K] (zero-point=8, i.e. signed
    range [-8,7] stored as [0,15]) and per-group scale [K//group_size, N] (transposed,
    matching the kernel's [num_groups, OC] indexing)."""
    N, K = W.shape
    Wg = W.reshape(N, K // group_size, group_size)
    scale = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.round(Wg / scale).clamp(-8, 7)
    unsigned_q = (q + 8).to(torch.int32).reshape(N, K)
    scale = scale.squeeze(-1)  # [N, K//group_size]
    return unsigned_q, scale


dev = torch.device("cuda:0")
torch.manual_seed(0)

N, K, GROUP = 256, 128, 64  # small, kernel-shape-legal (N%8==0 for interleave, K%64==0)
W = torch.randn(N, K, device=dev) * 0.05

unsigned_q, scale = quantize_symmetric_int4(W.cpu(), group_size=GROUP)
qweight_packed = pack_intweight(unsigned_q, interleave=4, kstride=64).to(dev)  # [N/4, K] int32

num_groups = K // GROUP
scales_bf16 = scale.to(torch.bfloat16).T.contiguous().to(dev)  # [num_groups, N]
zero_point = 8.0
scaled_zeros_bf16 = (-(scale.to(torch.float32) * zero_point)).to(torch.bfloat16).T.contiguous().to(dev)  # [num_groups, N]

x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)

out = nunchaku_min.gemv_awq(x, qweight_packed, scales_bf16, scaled_zeros_bf16, 1, N, K, GROUP)

# reference: dequantize W from unsigned_q/scale (matching (q-8)*scale) and matmul
W_dequant = (unsigned_q.to(torch.float32) - zero_point).reshape(N, K // GROUP, GROUP) * scale.unsqueeze(-1)
W_dequant = W_dequant.reshape(N, K).to(dev)
y_true = x.float() @ W_dequant.T
err = (out.float() - y_true).abs().max().item()
print(f"max abs diff: {err}")
sig = (y_true**2).sum()
noise = ((out.float()-y_true)**2).sum()
print(f"SQNR: {10*torch.log10(sig/noise).item():.2f} dB")
