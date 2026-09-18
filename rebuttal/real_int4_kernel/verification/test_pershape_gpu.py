"""TRUE per-shape GPU time for the W4A4 path at a given M.

Earlier per-shape microbenchmarks were host-bound: a Python->pybind->2-launch call costs
~26us of submission, which exceeded the GPU work, so every shape measured ~27us
regardless of size (2.1MB, 8.4MB and 29.4MB all took ~27-29us). Those numbers said
nothing about bandwidth.

Fix: capture 32 DISTINCT weight instances of the shape into ONE CUDA graph and time the
replay. Host cost collapses to a single graph launch, and using 32 distinct weights (like
the model's 32 layers) means the weights stream from DRAM instead of sitting hot in the
4090's 72MB L2 -- which is what the real model does.
"""
import os
import sys, time, torch
_nk = os.environ.get("NUNCHAKU_DIR")
if _nk: sys.path.insert(0, _nk)
import nunchaku_min

dev = torch.device("cuda:0")
torch.manual_seed(0)
L = 32  # distinct instances, like 32 transformer layers

BW_PEAK = 1008e9


def run(N, K, M, label):
    gs = []
    for _ in range(L):
        W = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
        g = nunchaku_min.QuantizedGEMM()
        g.init(K, N, False, True, 0)
        g.load_weight(W)
        g.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))
        gs.append((g, torch.ones(K, dtype=torch.bfloat16, device=dev), W))
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for g, b, _ in gs:
            g.forward_static(x, b)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for g, b, _ in gs:
            g.forward_static(x, b)
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()

    it = 20
    t0 = time.perf_counter()
    for _ in range(it):
        gr.replay()
    torch.cuda.synchronize()
    per_call = (time.perf_counter() - t0) / it / L

    # BF16 reference, same graph treatment
    grb = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        for _, _, W in gs:
            x @ W.T
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(grb):
        for _, _, W in gs:
            x @ W.T
    for _ in range(3):
        grb.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        grb.replay()
    torch.cuda.synchronize()
    per_call_bf16 = (time.perf_counter() - t0) / it / L

    b4 = N * K / 2 + (K // 64) * N * 2
    bb = N * K * 2
    blocks = (N // 128) * max(1, -(-M // 128))
    print(f"  {label:<22} M={M:<5} blocks={blocks:<5} "
          f"INT4 {per_call*1e6:7.2f}us ({b4/per_call/1e9:6.1f} GB/s, {b4/per_call/BW_PEAK*100:4.1f}% peak)   "
          f"BF16 {per_call_bf16*1e6:7.2f}us ({bb/per_call_bf16/1e9:6.1f} GB/s)   "
          f"speedup {per_call_bf16/per_call:5.2f}x")
    del gs
    torch.cuda.empty_cache()


for M in (1, 16):
    print(f"--- M={M} (decode) ---")
    run(4096, 4096, M, "q_proj / o_proj")
    run(1024, 4096, M, "k_proj / v_proj")
    run(14336, 4096, M, "gate_proj / up_proj")
    run(4096, 14336, M, "down_proj")
    print()
