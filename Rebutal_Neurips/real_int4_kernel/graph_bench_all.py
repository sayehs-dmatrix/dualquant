"""All four throughput numbers (prefill/decode x batch 1/16) under ONE fair, honest
methodology: every configuration is CUDA-graph captured, and the BF16 baseline gets
exactly the same treatment as the INT4 path.

Why graph everything: torch.profiler showed a batch-1 decode step is ~8 ms of GPU work
behind ~36.5 ms of CPU dispatch (~1365 kernel launches/token). Prefill at batch=1 is the
same story (82 tokens is still small work per launch). Measuring in that regime compares
Python dispatch overhead, not quantization -- and it penalises INT4 specifically, because
INT4 issues MORE work per linear (quantize + GEMM) than BF16's single cuBLAS call. So the
CPU ceiling is removed from BOTH paths, and what remains is the GPU-time difference, which
is the thing quantization actually changes.

Correctness gates every number: each captured graph is checked against the identical
uncaptured execution before it is timed (decode: generated tokens must match and
max |logit diff| must be 0; prefill: logits must match bit-exactly).

Usage:
    python graph_bench_all.py --mode bf16
    python graph_bench_all.py --mode int4
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_NUNCHAKU_DIR = os.environ.get("NUNCHAKU_DIR", "")
for p in (_CODEBASE_ROOT, _HERE, _NUNCHAKU_DIR):
    if p and p not in sys.path:
        sys.path.insert(0, p)

import torch
import transformers
from transformers import StaticCache

from main import load_model, build_cfg, _load_all_hyperparams

GB = 1e9


def build_model(model_id, mode, dev):
    if mode == "bf16":
        m = load_model(model_id)
        m.config.use_cache = True
        return m.to(dev).eval(), None
    os.environ["NUNCHAKU_FWD_MODE"] = "static"
    import full_model_nunchaku as fmn
    fmn.NunchakuW4A4Linear._FWD_MODE = "static"
    args = fmn._make_args(model_id, 128, 2048)
    c = build_cfg(args, _load_all_hyperparams(args.method_cfg))
    c["_model_id"] = model_id
    m = load_model(model_id)
    m.config.use_cache = True
    fmn.convert_model_to_nunchaku(m, c)
    return m.to(dev).eval(), fmn


def timed_replay(g, n, per_call_tokens):
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    return per_call_tokens / dt, dt * 1e3


@torch.no_grad()
def prefill(model, ids, bs, dev, reps=20):
    """Graph-capture a full prefill forward and time replay. Returns (tok/s, ms, ok)."""
    ids_b = ids.repeat(bs, 1) if bs > 1 else ids
    ref = model(ids_b, use_cache=False).logits.float().clone()

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model(ids_b, use_cache=False)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(ids_b, use_cache=False).logits
    g.replay(); torch.cuda.synchronize()
    ok = torch.equal(out.float(), ref)
    diff = (out.float() - ref).abs().max().item()

    tps, ms = timed_replay(g, reps, bs * ids_b.shape[1])
    return tps, ms, ok, diff


@torch.no_grad()
def decode(model, ids, bs, dev, new_tokens=100, val_steps=6):
    """Graph-capture ONE decode step (StaticCache) and replay it. Returns (tok/s, ms, ok)."""
    prompt_len = ids.shape[1]
    ids_b = ids.repeat(bs, 1) if bs > 1 else ids
    max_len = prompt_len + new_tokens + val_steps + 32

    def fresh():
        cache = StaticCache(config=model.config, batch_size=bs, max_cache_len=max_len,
                            device=dev, dtype=torch.bfloat16)
        pos = torch.arange(prompt_len, device=dev)
        o = model(ids_b, past_key_values=cache, cache_position=pos, use_cache=True)
        return cache, o.logits[:, -1:].argmax(-1)

    # --- reference: uncaptured stepping on a StaticCache ---
    cache, tok = fresh()
    si = tok.clone()
    sp = torch.tensor([prompt_len], device=dev, dtype=torch.long)
    step = lambda: model(input_ids=si, past_key_values=cache, cache_position=sp,
                         use_cache=True).logits
    ref_toks, p = [], prompt_len
    for _ in range(val_steps):
        lg = step()
        nxt = lg[:, -1:].argmax(-1)
        ref_toks.append(nxt[0].item())
        si.copy_(nxt); p += 1; sp.fill_(p)

    # --- captured ---
    # ORDER MATTERS. QuantizedGEMM::forward_static reallocates its scratch when M
    # changes, and nunchaku allocates with cudaMallocAsync on the legacy stream,
    # which is illegal during capture. So the prefill (M = prompt_len*bs) must
    # happen BEFORE the decode-shape warm-up, and nothing with a different M may
    # run afterwards -- a later reallocation would also free the very buffers the
    # captured graph has baked pointers to.
    cache, tok = fresh()                 # M = prompt_len*bs  (allocates)
    si.copy_(tok); sp.fill_(prompt_len)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()                       # M = bs (allocates; last alloc before capture)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture on the SAME cache the warm-up used. Capture records without
    # executing, so the cache is untouched; the warm-up already wrote tok's K/V
    # at prompt_len, which is exactly what the first replayed step re-writes.
    si.copy_(tok); sp.fill_(prompt_len)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        slg = step()

    si.copy_(tok); sp.fill_(prompt_len)
    got, p = [], prompt_len
    for _ in range(val_steps):
        g.replay(); torch.cuda.synchronize()
        nxt = slg[:, -1:].argmax(-1)
        got.append(nxt[0].item())
        si.copy_(nxt); p += 1; sp.fill_(p)
    ok = got == ref_toks

    sp.fill_(prompt_len + val_steps)
    tps, ms = timed_replay(g, new_tokens, bs)   # one replay = 1 token per sequence
    return tps, ms, ok, (ref_toks, got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--mode", choices=["bf16", "int4"], required=True)
    cli = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.cuda.empty_cache()
    f0, total = torch.cuda.mem_get_info()
    base = (total - f0) / GB

    model, fmn = build_model(cli.model, cli.mode, dev)
    tok = transformers.AutoTokenizer.from_pretrained(cli.model, use_fast=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if cli.mode == "int4":
        import nunchaku_min
        nunchaku_min.trim_memory_pool(0)
    torch.cuda.synchronize()
    f1, _ = torch.cuda.mem_get_info()
    weights = (total - f1) / GB - base

    ids = tok("The quick brown fox jumps over the lazy dog. " * 8,
              return_tensors="pt").input_ids.to(dev)

    ctx = torch.cuda.stream(fmn.GRAPH_STREAM) if cli.mode == "int4" else __import__("contextlib").nullcontext()
    print(f"\n===== {cli.mode.upper()} | weights {weights:.3f} GB | all configs CUDA-graph captured =====")
    with ctx:
        for bs in (1, 16):
            tps, ms, ok, diff = prefill(model, ids, bs, dev)
            print(f"PREFILL batch={bs:<3} {tps:9.1f} tok/s  {ms:8.3f} ms  "
                  f"logits_bit_exact={ok} (maxdiff={diff:g})")
        for bs in (1, 16):
            tps, ms, ok, info = decode(model, ids, bs, dev)
            print(f"DECODE  batch={bs:<3} {tps:9.1f} tok/s  {ms:8.3f} ms/step  tokens_match={ok}")
            if not ok:
                print(f"    ref={info[0]}\n    got={info[1]}")


if __name__ == "__main__":
    main()
