import os
import sys, torch
_nk = os.environ.get("NUNCHAKU_DIR")
if _nk: sys.path.insert(0, _nk)
import nunchaku_min

import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root from this file

dev = torch.device("cuda:0")
d = torch.load(_os.path.join(_REPO_ROOT, "ablation", "real_int4_kernel/wrap_cache/meta-llama__Llama-3.1-8B__layer0.o_proj__bs64__68ccec1c.pt"),
               map_location="cpu")
mat_q, beta = d["mat_q"], d["beta"]
N, K = mat_q.shape

g = nunchaku_min.QuantizedGEMM()
g.init(K, N, False, True, 0)
g.load_weight(mat_q.to(dev).to(torch.bfloat16).contiguous())
g.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))
beta_d = beta.to(dev).to(torch.bfloat16)

torch.manual_seed(0)
M = 8
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.1

act, ascales = g.quantize_probe(x, beta_d)
print(f"layer: o_proj  N={N} K={K}  M={M}")
print(f"packed activation tensor: shape={tuple(act.shape)} dtype={act.dtype}")
print(f"  -> bytes per row = {act.shape[1]}, activation elements per row = {K}")
print(f"  -> bits per activation element = {act.shape[1]*8/K:.2f}   (4.00 == INT4)")
print(f"ascales: shape={tuple(ascales.shape)} dtype={ascales.dtype}  "
      f"(one scale per {K//ascales.shape[0]} channels)")

# Unpack the two 4-bit nibbles from each byte and map to signed int4.
b = act.to(torch.int16)
lo = (b & 0xF)
hi = ((b >> 4) & 0xF)
nib = torch.stack([lo, hi], dim=-1).reshape(act.shape[0], -1)
signed = torch.where(nib > 7, nib - 16, nib)     # two's-complement int4 -> [-8, 7]

vals = torch.unique(signed[:M])
print(f"\ndistinct quantized activation levels actually used: {len(vals)}")
print(f"  min={signed[:M].min().item()}  max={signed[:M].max().item()}")
print(f"  levels: {sorted(v.item() for v in vals)}")
ok_range = signed[:M].min() >= -8 and signed[:M].max() <= 7
ok_bits = abs(act.shape[1] * 8 / K - 4.0) < 1e-9
ok_levels = len(vals) <= 16
print(f"\nfits in 4 bits (<=16 levels, range [-8,7]): {ok_range and ok_levels}")
print(f"storage is exactly 4 bits/element:            {ok_bits}")
print("\nCONCLUSION: activation operand of the GEMM is REAL INT4" if (ok_range and ok_bits and ok_levels)
      else "\nCONCLUSION: NOT int4 -- investigate")
