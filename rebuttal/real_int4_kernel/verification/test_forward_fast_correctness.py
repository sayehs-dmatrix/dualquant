import os
import sys, torch
_nk = os.environ.get("NUNCHAKU_DIR")
if _nk: sys.path.insert(0, _nk)
import nunchaku_min

dev = torch.device("cuda:0")
torch.manual_seed(0)

def sqnr_db(ref, approx):
    sig = (ref.float() ** 2).sum()
    noise = ((approx.float() - ref.float()) ** 2).sum()
    return 10.0 * torch.log10(sig / noise).item()

N, K = 14336, 4096
W = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
beta = (torch.rand(K) * 3 + 1).to(torch.bfloat16).cuda()

gemm = nunchaku_min.QuantizedGEMM()
gemm.init(K, N, False, True, 0)
gemm.load_weight(W)
gemm.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))

print("=== Test 1: single-call correctness (forward_fast vs forward vs ground truth) ===")
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
x_scaled = x * beta
y_ref = gemm.forward(x_scaled)
y_fast = gemm.forward_fast(x_scaled)
y_true = (x.float() * beta.float()) @ W.float().T
print(f"SQNR(forward vs true)      = {sqnr_db(y_true, y_ref):.2f} dB")
print(f"SQNR(forward_fast vs true) = {sqnr_db(y_true, y_fast):.2f} dB")
print(f"SQNR(forward_fast vs forward) = {sqnr_db(y_ref, y_fast):.2f} dB  (should be ~inf / very high if identical)")

print()
print("=== Test 2: repeated-call correctness (simulating decode loop, catch aliasing bugs) ===")
torch.manual_seed(123)
xs = [torch.randn(1, K, dtype=torch.bfloat16, device=dev) for _ in range(10)]
results_fast = []
results_ref = []
for i, xi in enumerate(xs):
    xi_scaled = xi * beta
    y_f = gemm.forward_fast(xi_scaled)
    results_fast.append(y_f.clone())  # clone since it aliases cached_out
    y_r = gemm.forward(xi_scaled)
    results_ref.append(y_r.clone())

all_good = True
for i, (xi, yf, yr) in enumerate(zip(xs, results_fast, results_ref)):
    y_true_i = (xi.float() * beta.float()) @ W.float().T
    s_fast = sqnr_db(y_true_i, yf)
    s_ref = sqnr_db(y_true_i, yr)
    print(f"  call {i}: SQNR(fast)={s_fast:6.2f}dB  SQNR(ref)={s_ref:6.2f}dB")
    if s_fast < 10:
        all_good = False

print()
print("ALL GOOD" if all_good else "CORRECTNESS FAILURE DETECTED")
