"""Measure the ACHIEVABLE ceilings, rather than assuming 4x (decode) / 16x (prefill).

decode  = memory-bound  -> ceiling set by bytes read and achieved GB/s
prefill = compute-bound -> ceiling set by achieved OPS vs BF16 achieved FLOPS
"""
import sys, time, torch
sys.path.insert(0, "/tmp/claude-0/-root-numrd/1e334428-137f-442b-9669-4bc5945263d0/scratchpad/svdquant_repo/build/lib.linux-x86_64-cpython-312")
import nunchaku_min

dev = torch.device("cuda:0")
torch.manual_seed(0)


def bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run(N, K, M, label, iters=50):
    W = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
    beta = torch.ones(K, dtype=torch.bfloat16, device=dev)
    g = nunchaku_min.QuantizedGEMM()
    g.init(K, N, False, True, 0)
    g.load_weight(W)
    g.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)

    t_int4 = bench(lambda: g.forward_beta(x, beta), iters)
    t_bf16 = bench(lambda: x @ W.T, iters)

    flops = 2.0 * M * N * K
    print(f"  {label:<26} M={M:<5} INT4 {t_int4*1e3:8.3f} ms ({flops/t_int4/1e12:7.1f} TOP/s)   "
          f"BF16 {t_bf16*1e3:8.3f} ms ({flops/t_bf16/1e12:6.1f} TFLOP/s)   speedup {t_bf16/t_int4:5.2f}x")
    return t_bf16 / t_int4


print("=" * 118)
print("COMPUTE-BOUND regime (large M, like prefill): what is the kernel's real arithmetic ceiling?")
print("=" * 118)
for M in (512, 1312, 4096, 8192):
    run(14336, 4096, M, "gate/up_proj shape")
print()
for M in (1312, 4096):
    run(4096, 4096, M, "q/o_proj shape")

print()
print("=" * 118)
print("Peak achievable on this GPU, measured (big square GEMM, no quantization overhead in BF16 case)")
print("=" * 118)
n = 8192
A = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
B = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
t = bench(lambda: A @ B, 30)
print(f"  cuBLAS BF16 {n}^3 GEMM: {t*1e3:.3f} ms -> {2.0*n**3/t/1e12:.1f} TFLOP/s achieved")
print(f"  (RTX 4090 spec: BF16 tensor 165.2 TFLOP/s dense; INT8 660.6 TOP/s; INT4 1321 TOP/s)")

print()
print("=" * 118)
print("MEMORY-BOUND regime (M=1, like decode): achieved bandwidth")
print("=" * 118)
for (N, K, lbl) in ((14336, 4096, "gate/up_proj"), (4096, 14336, "down_proj"), (4096, 4096, "q/o_proj")):
    W = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
    beta = torch.ones(K, dtype=torch.bfloat16, device=dev)
    g = nunchaku_min.QuantizedGEMM(); g.init(K, N, False, True, 0)
    g.load_weight(W); g.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))
    x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    t4 = bench(lambda: g.forward_beta(x, beta), 300)
    tb = bench(lambda: x @ W.T, 300)
    b4 = N * K / 2 + (K // 64) * N * 2      # packed int4 weight + bf16 scales
    bb = N * K * 2
    print(f"  {lbl:<14} INT4 {t4*1e6:7.2f} us ({b4/t4/1e9:6.1f} GB/s)   "
          f"BF16 {tb*1e6:7.2f} us ({bb/tb/1e9:6.1f} GB/s)   speedup {tb/t4:5.2f}x")
