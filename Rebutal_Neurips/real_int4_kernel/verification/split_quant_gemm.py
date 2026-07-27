"""Per-shape split of the W4A4 forward into its two kernels, true GPU time.

k/v_proj moves 2.23MB and takes 28.0us; gate/up moves 31.2MB (14x more) and takes 40.2us.
A large shape-independent GPU cost is hiding in there. Attribute it with the profiler.
"""
import sys, torch
sys.path.insert(0, "/tmp/claude-0/-root-numrd/1e334428-137f-442b-9669-4bc5945263d0/scratchpad/svdquant_repo/build/lib.linux-x86_64-cpython-312")
import nunchaku_min

dev = torch.device("cuda:0")
torch.manual_seed(0)
L = 32


def run(N, K, M, label):
    gs = []
    for _ in range(L):
        W = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
        g = nunchaku_min.QuantizedGEMM()
        g.init(K, N, False, True, 0)
        g.load_weight(W)
        g.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))
        gs.append((g, torch.ones(K, dtype=torch.bfloat16, device=dev)))
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for g, b in gs:
            g.forward_static(x, b)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for g, b in gs:
            g.forward_static(x, b)
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()

    R = 10
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(R):
            gr.replay()
        torch.cuda.synchronize()

    per = {}
    for e in prof.key_averages():
        if e.self_device_time_total <= 0:
            continue
        nm = e.key
        tag = ("gemm" if "gemm_w4a4_kernel" in nm or ("gemm" in nm and "quantize" not in nm)
               else "quantize" if "quantize" in nm else "other")
        per[tag] = per.get(tag, 0.0) + e.self_device_time_total / R / L

    tot = sum(per.values())
    parts = "  ".join(f"{k}={v:6.2f}us" for k, v in sorted(per.items()))
    print(f"  {label:<22} M={M:<5} total={tot:6.2f}us   {parts}")
    del gs
    torch.cuda.empty_cache()


for M in (1, 16):
    print(f"--- M={M} ---")
    run(1024, 4096, M, "k/v_proj (N=1024)")
    run(4096, 4096, M, "q/o_proj (N=4096)")
    run(14336, 4096, M, "gate/up (N=14336)")
    run(4096, 14336, M, "down_proj (K=14336)")
    print()
