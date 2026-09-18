import os
import sys
import torch
_nk = os.environ.get("NUNCHAKU_DIR")
if _nk: sys.path.insert(0, _nk)
sys.path.insert(0, "/root/numrd/Quantization_Repo_July2025/Dualquant_codebase_20260508/Rebutal_Neurips/real_int4_kernel")
import nunchaku_min
from awq_gemv_pack import build_awq_gemv_weights, GROUP_SIZE

dev = torch.device("cuda:0")

def sqnr_db(ref, approx):
    sig = (ref.float() ** 2).sum()
    noise = ((approx.float() - ref.float()) ** 2).sum()
    return 10.0 * torch.log10(sig / noise).item()

d = torch.load(
    "/root/numrd/Quantization_Repo_July2025/Dualquant_codebase_20260508/Rebutal_Neurips/real_int4_kernel/wrap_cache/meta-llama__Llama-3.1-8B__layer0.o_proj__bs64__68ccec1c.pt",
    map_location="cpu",
)
mat_q, beta = d["mat_q"], d["beta"]
N, K = mat_q.shape
print(f"o_proj layer0: mat_q {mat_q.shape}, beta {beta.shape}")

qweight, scales_bf16, scaled_zeros_bf16 = build_awq_gemv_weights(mat_q, beta, dev, group_size=GROUP_SIZE)

torch.manual_seed(0)
mat_q_dev = mat_q.to(dev)
beta_dev = beta.to(dev).to(torch.bfloat16)

# reference: nunchaku's existing, already-validated forward_beta (real W4A4 path)
gemm = nunchaku_min.QuantizedGEMM()
gemm.init(K, N, False, True, 0)
gemm.load_weight(mat_q_dev.to(torch.bfloat16).contiguous())
gemm.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))

all_good = True
for i in range(10):
    x = torch.randn(1, K, dtype=torch.bfloat16, device=dev) * 0.1
    y_w4a4 = gemm.forward_beta(x, beta_dev).clone()
    y_awq = nunchaku_min.gemv_awq(x, qweight, scales_bf16, scaled_zeros_bf16, 1, N, K, GROUP_SIZE)
    y_true = (x.float() * beta_dev.float()) @ mat_q_dev.float().T
    s_w4a4 = sqnr_db(y_true, y_w4a4)
    s_awq = sqnr_db(y_true, y_awq)
    print(f"call {i}: SQNR(w4a4 vs true)={s_w4a4:.2f}dB  SQNR(awq_gemv vs true)={s_awq:.2f}dB")
    if s_awq < 10:
        all_good = False

print("ALL GOOD" if all_good else "CORRECTNESS FAILURE")
