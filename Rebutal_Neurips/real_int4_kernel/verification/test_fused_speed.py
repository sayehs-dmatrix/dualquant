import sys, time, torch
sys.path.insert(0, "/tmp/claude-0/-root-numrd/1e334428-137f-442b-9669-4bc5945263d0/scratchpad/svdquant_repo/build/lib.linux-x86_64-cpython-312")
import nunchaku_min

dev = torch.device("cuda:0")
torch.manual_seed(0)

def bench(fn, iters=1000, warmup=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us/call

def run(N, K, label, M=1):
    W = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
    beta = (torch.rand(K) * 3 + 1).to(torch.bfloat16).cuda()
    gemm = nunchaku_min.QuantizedGEMM()
    gemm.init(K, N, False, True, 0)
    gemm.load_weight(W)
    gemm.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)

    custom_stream = torch.cuda.Stream()
    with torch.cuda.stream(custom_stream):
        t_graph_beta = bench(lambda: gemm.forward_graph_beta(x, beta))
        t_fused = bench(lambda: gemm.forward_graph_beta_fused(x, beta))
        torch.cuda.current_stream().synchronize()

    t_beta = bench(lambda: gemm.forward_beta(x, beta))

    print(f"--- {label} (N={N}, K={K}, M={M}) ---")
    print(f"forward_beta (no graph):         {t_beta:.2f} us")
    print(f"forward_graph_beta (2 launches): {t_graph_beta:.2f} us")
    print(f"forward_graph_beta_fused (SetParams+1 launch): {t_fused:.2f} us")
    print(f"fused vs graph_beta: {t_graph_beta/t_fused:.3f}x")
    print()

run(4096, 4096, "o_proj", M=1)
run(1024, 4096, "kv_proj (GQA)", M=1)
run(4096, 4096, "o_proj", M=16)
run(1024, 4096, "kv_proj (GQA)", M=16)
