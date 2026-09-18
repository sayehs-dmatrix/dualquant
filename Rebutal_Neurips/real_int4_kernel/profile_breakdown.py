"""Per-component GPU time breakdown for all four workloads, both precisions.

Profiles the SAME CUDA-graph-captured configurations that graph_bench_all.py times, so the
breakdown adds up to the reported end-to-end numbers. Because a graph replay contains only
raw kernels (no ATen op attribution), components are identified by kernel name:

  w4a4_gemm  : nunchaku INT4xINT4 CUTLASS GEMM        (the 224 quantized linears)
  quantize   : nunchaku activation quantize            (INT4 path only)
  cublas     : cuBLAS GEMM/GEMV. INT4 path -> lm_head ONLY (it is left in BF16);
               BF16 path -> the 224 linears AND lm_head together.
  attention  : flash / memory-efficient attention kernels
  other      : RMSNorm, RoPE, residual adds, KV-cache copies, argmax, casts

Usage:  python profile_breakdown.py --mode bf16|int4
"""
import os
import argparse, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(_HERE)), _HERE,
          os.environ.get("NUNCHAKU_DIR", "")):
    if p and p not in sys.path:
        sys.path.insert(0, p)

import torch, transformers
from transformers import StaticCache
import graph_bench_all as GB

CATS = ["w4a4_gemm", "quantize", "cublas", "attention", "other"]


def categorize(name):
    n = name.lower()
    if "quantize" in n:
        return "quantize"
    if "gemm_w4a4" in n or ("nunchaku" in n and "gemm" in n):
        return "w4a4_gemm"
    if "fmha" in n or "flash" in n or "attention" in n:
        return "attention"
    if "gemv" in n or "gemm" in n or "cutlass" in n or "sgemm" in n or "dot_kernel" in n:
        return "cublas"
    return "other"


def profile_graph(g, reps, divisor):
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    out = {c: 0.0 for c in CATS}
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            out[categorize(e.key)] += e.self_device_time_total / reps / divisor / 1000.0
    return out


@torch.no_grad()
def cap_prefill(model, ids, bs):
    ids_b = ids.repeat(bs, 1) if bs > 1 else ids
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model(ids_b, use_cache=False)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        model(ids_b, use_cache=False)
    return g


@torch.no_grad()
def cap_decode(model, ids, bs, dev, extra=160):
    plen = ids.shape[1]
    ids_b = ids.repeat(bs, 1) if bs > 1 else ids
    cache = StaticCache(config=model.config, batch_size=bs, max_cache_len=plen + extra,
                        device=dev, dtype=torch.bfloat16)
    o = model(ids_b, past_key_values=cache, cache_position=torch.arange(plen, device=dev),
              use_cache=True)
    si = o.logits[:, -1:].argmax(-1).clone()
    sp = torch.tensor([plen], device=dev, dtype=torch.long)
    step = lambda: model(input_ids=si, past_key_values=cache, cache_position=sp, use_cache=True)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--mode", choices=["bf16", "int4"], required=True)
    cli = ap.parse_args()
    dev = torch.device("cuda:0")

    model, fmn = GB.build_model(cli.model, cli.mode, dev)
    tok = transformers.AutoTokenizer.from_pretrained(cli.model, use_fast=False)
    ids = tok("The quick brown fox jumps over the lazy dog. " * 8,
              return_tensors="pt").input_ids.to(dev)
    ctx = torch.cuda.stream(fmn.GRAPH_STREAM) if cli.mode == "int4" else __import__("contextlib").nullcontext()

    print(f"\n===== {cli.mode.upper()} per-component GPU time (ms) =====")
    hdr = f"{'workload':<20}" + "".join(f"{c:>12}" for c in CATS) + f"{'TOTAL':>10}"
    print(hdr)
    with ctx:
        for bs in (1, 16):
            g = cap_prefill(model, ids, bs)
            d = profile_graph(g, 10, 1)
            print(f"{'prefill b'+str(bs):<20}" + "".join(f"{d[c]:>12.3f}" for c in CATS)
                  + f"{sum(d.values()):>10.3f}")
            del g
        for bs in (1, 16):
            g = cap_decode(model, ids, bs, dev)
            d = profile_graph(g, 30, 1)
            print(f"{'decode b'+str(bs):<20}" + "".join(f"{d[c]:>12.3f}" for c in CATS)
                  + f"{sum(d.values()):>10.3f}")
            del g


if __name__ == "__main__":
    main()
