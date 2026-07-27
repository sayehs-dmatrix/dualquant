"""Can concurrent execution of independent projections fill the GPU at M=1?

At M=1 the W4A4 grid is (1, N/BLOCK_N) = N/128 blocks, so q/o/down get 32 blocks and
k/v get 8, on a 128-SM GPU. q, k, v are independent (same input, different weights), as
are gate and up -- so running them on separate streams should overlap and recover the
idle SMs. Uses forward_static because it pushes the caller's stream onto nunchaku's
internal stream stack (forward_beta always lands on legacy stream 0, so it cannot overlap).
"""
import sys, time, torch
sys.path.insert(0, "/tmp/claude-0/-root-numrd/1e334428-137f-442b-9669-4bc5945263d0/scratchpad/svdquant_repo/build/lib.linux-x86_64-cpython-312")
import nunchaku_min

dev = torch.device("cuda:0")
torch.manual_seed(0)
K = 4096


def make(N):
    W = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
    g = nunchaku_min.QuantizedGEMM()
    g.init(K, N, False, True, 0)
    g.load_weight(W)
    g.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))
    return g, torch.ones(K, dtype=torch.bfloat16, device=dev)


def bench(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def group(shapes, label, M=1):
    gs = [make(N) for N in shapes]
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    main = torch.cuda.Stream()
    side = [torch.cuda.Stream() for _ in shapes]

    def serial():
        with torch.cuda.stream(main):
            for g, b in gs:
                g.forward_static(x, b)

    def parallel():
        with torch.cuda.stream(main):
            ev = torch.cuda.Event()
            ev.record(main)
            for s in side:
                s.wait_event(ev)          # fork
            for s, (g, b) in zip(side, gs):
                with torch.cuda.stream(s):
                    g.forward_static(x, b)
            for s in side:                 # join
                e = torch.cuda.Event()
                e.record(s)
                main.wait_event(e)

    ts, tp = bench(serial), bench(parallel)
    print(f"  {label:<34} M={M:<5} serial {ts:7.2f} us   {len(shapes)} streams {tp:7.2f} us   "
          f"speedup {ts/tp:5.2f}x")


print("Independent-projection overlap (per Llama layer group):")
group([4096, 1024, 1024], "q + k + v  (N=4096,1024,1024)", M=1)
group([14336, 14336], "gate + up  (N=14336 x2)", M=1)
group([4096, 1024, 1024], "q + k + v", M=16)
group([14336, 14336], "gate + up", M=16)
group([4096, 1024, 1024], "q + k + v", M=1312)
group([14336, 14336], "gate + up", M=1312)
