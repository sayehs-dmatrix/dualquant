"""Whole-step CUDA-graph decode: the actual fix for decode throughput.

WHY THIS EXISTS (measured, not assumed). torch.profiler on a real Llama-3.1-8B
decode step showed:
    GPU (self CUDA) time  ~8.07 ms / step
    CPU time             ~36.5  ms / step   <-- the real wall-clock ceiling
i.e. decode was CPU-DISPATCH-BOUND, not GPU-bound: ~1365 kernel launches per
token, each costing ~6.5us of cudaLaunchKernel plus ATen dispatch, spread over
norms / RoPE / attention / KV-cache concat / 224 linear layers. Every
kernel-level optimization therefore hit a wall:
  - fusing the beta multiply into the GEMM call: real but small
  - folding the beta multiply into the captured graph (SetParams): ~1.5%
  - swapping in AWQ's faster weight-only GEMV: NET LOSS end-to-end (it is
    faster on the GPU but adds Python-side dispatch, which is what actually
    binds)
Under a CPU-bound ceiling, INT4 loses to BF16 for a mundane reason: it issues
MORE dispatch work per linear (beta-mul + quantize + GEMM) than BF16's single
cuBLAS call, even though it moves 2.15x less weight data.

The memory-bound floors say what SHOULD happen once that ceiling is gone
(RTX 4090, ~1008 GB/s):
    INT4 weights  7.455 GB -> 7.4 ms/step floor   (measured GPU: 8.07 ms, 1.09x floor)
    BF16 weights 16.061 GB -> 15.9 ms/step floor
=> ~2x, which is the real memory-bound INT4 win.

So: capture the ENTIRE decode step (embedding, all 32 blocks incl. attention
and norms, LM head) into ONE CUDA graph and replay it per token. That collapses
~1365 launches into 1 graph launch, removing the CPU ceiling and letting the
GPU-time difference become the throughput difference.

Requirements this forces:
  - StaticCache: a fixed-address, pre-allocated KV cache. HF's default cache
    grows via torch.cat every step (new addresses => uncapturable, since a
    graph bakes in pointers).
  - Fixed-address input/position buffers, copied into per step.
  - NUNCHAKU_FWD_MODE=static: QuantizedGEMM::forward_static, which does no
    internal graph capture (nesting is illegal), no allocation during capture
    (nunchaku's Tensor::allocate uses cudaMallocAsync on legacy stream 0), and
    pushes the capture stream onto nunchaku's internal stream stack so its
    kernels actually land in the graph.
  - Warm-up at the SAME M before capture, on a side stream.

The BF16 baseline gets the IDENTICAL graph treatment, so the comparison
measures quantization, not who got the optimization.

Usage:
    python whole_step_graph_decode.py --model meta-llama/Llama-3.1-8B --mode int4
    python whole_step_graph_decode.py --model meta-llama/Llama-3.1-8B --mode bf16
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


def build_int4_model(model_id, dev):
    # force the capture-safe forward BEFORE the module class is used
    os.environ["NUNCHAKU_FWD_MODE"] = "static"
    import full_model_nunchaku as fmn
    fmn.NunchakuW4A4Linear._FWD_MODE = "static"
    args = fmn._make_args(model_id, 128, 2048)
    cfg = build_cfg(args, _load_all_hyperparams(args.method_cfg))
    cfg["_model_id"] = model_id
    model = load_model(model_id)
    model.config.use_cache = True
    fmn.convert_model_to_nunchaku(model, cfg)
    return model.to(dev).eval()


def build_bf16_model(model_id, dev):
    model = load_model(model_id)
    model.config.use_cache = True
    return model.to(dev).eval()


@torch.no_grad()
def eager_reference_logits(model, ids, n_steps, dev):
    """Ground truth: plain eager decode with the normal dynamic cache.

    CONVENTION (must match graph_decode exactly, or the comparison is
    meaningless -- an earlier version of this file had the two functions
    disagreeing on whether the prefill's own argmax counts as generated
    token #0, which showed up as a confusing "coherent but shifted by one"
    diff): returns the first n_steps generated tokens in order, where
    token 0 is argmax(prefill logits). logits_hist[i] is the logit vector
    that PRODUCED toks[i]."""
    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits_hist = [out.logits[:, -1, :].float().clone()]
    tok = out.logits[:, -1:].argmax(-1)
    toks = [tok.item()]
    for _ in range(n_steps - 1):
        out = model(tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits_hist.append(out.logits[:, -1, :].float().clone())
        tok = out.logits[:, -1:].argmax(-1)
        toks.append(tok.item())
    return toks, logits_hist


@torch.no_grad()
def graph_decode(model, ids, n_steps, dev, max_cache_len, batch_size=1, capture=True):
    """Prefill eagerly into a StaticCache, then capture ONE graph for the
    single-token decode step and replay it n_steps times."""
    cache = StaticCache(config=model.config, batch_size=batch_size,
                        max_cache_len=max_cache_len, device=dev, dtype=torch.bfloat16)

    prefill_len = ids.shape[1]
    pos = torch.arange(prefill_len, device=dev)
    out = model(ids, past_key_values=cache, cache_position=pos, use_cache=True)

    # SAME CONVENTION as eager_reference_logits: generated token 0 is
    # argmax(prefill logits); logits_hist[i] is the logit vector that produced
    # toks[i]. The decode loop below therefore runs n_steps-1 times.
    prefill_logits = out.logits[:, -1, :].float().clone()
    tok = out.logits[:, -1:].argmax(-1)
    tok0 = tok.clone()
    cur_pos = prefill_len

    static_input = tok.clone()                                    # [B,1] fixed address
    static_pos = torch.tensor([cur_pos], device=dev, dtype=torch.long)

    def step():
        return model(input_ids=static_input, past_key_values=cache,
                     cache_position=static_pos, use_cache=True).logits

    # Warm-up on a side stream (also triggers every one-time allocation inside
    # forward_static / cuBLAS / flash-attn, which must not happen during capture).
    # Warm-up writes token tok0's K/V at position prefill_len; the real first
    # step re-feeds the same token at the same position, overwriting it
    # identically, so this leaves no stale state.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    def run_loop(advance):
        toks = [tok0.item()]
        logits_hist = [prefill_logits]
        p = cur_pos
        static_input.copy_(tok0)
        static_pos.fill_(p)
        for _ in range(n_steps - 1):
            lg = advance()
            logits_hist.append(lg[:, -1, :].float().clone())
            nxt = lg[:, -1:].argmax(-1)
            toks.append(nxt.item())
            static_input.copy_(nxt)
            p += 1
            static_pos.fill_(p)
        return toks, logits_hist

    if not capture:
        toks, logits_hist = run_loop(step)
        return toks, logits_hist, None, None, None, None

    g = torch.cuda.CUDAGraph()
    static_input.copy_(tok0)
    static_pos.fill_(cur_pos)
    with torch.cuda.graph(g):
        static_logits = step()

    def replay():
        g.replay()
        torch.cuda.synchronize()
        return static_logits

    toks, logits_hist = run_loop(replay)
    return toks, logits_hist, g, static_input, static_pos, static_logits


@torch.no_grad()
def bench_graph_replay(g, static_input, static_pos, static_logits, start_pos, n_tokens=200, warmup=20):
    """Pure replay throughput: greedy-argmax feedback per token, same as real
    decoding, but with the whole step as one graph launch."""
    pos = start_pos
    for _ in range(warmup):
        g.replay()
        static_input.copy_(static_logits[:, -1:].argmax(-1))
        pos += 1
        static_pos.fill_(pos)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_tokens):
        g.replay()
        static_input.copy_(static_logits[:, -1:].argmax(-1))
        pos += 1
        static_pos.fill_(pos)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n_tokens / elapsed, elapsed / n_tokens * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--mode", choices=["int4", "bf16"], required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--validate-steps", type=int, default=8)
    p.add_argument("--bench-tokens", type=int, default=200)
    p.add_argument("--no-capture", action="store_true",
                   help="run the same StaticCache decode WITHOUT graph capture, to isolate "
                        "whether a correctness problem is in the forward path or in the capture")
    cli = p.parse_args()

    dev = torch.device("cuda:0")
    torch.cuda.empty_cache()
    free0, total = torch.cuda.mem_get_info()
    base_gb = (total - free0) / 1e9

    print(f"Loading {cli.model} in {cli.mode} mode...")
    model = build_int4_model(cli.model, dev) if cli.mode == "int4" else build_bf16_model(cli.model, dev)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cli.model, use_fast=False)
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    weights_gb = (total - free1) / 1e9 - base_gb
    print(f"weights memory (true, mem_get_info): {weights_gb:.3f}GB")

    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    if cli.batch_size > 1:
        ids = ids.repeat(cli.batch_size, 1)
    max_cache_len = ids.shape[1] + cli.bench_tokens + cli.validate_steps + 64

    # ---- correctness FIRST ----
    # PRIMARY check (this is what actually gates the speed number): same
    # StaticCache prefill, same starting token, decode stepped WITHOUT capture
    # vs WITH whole-step graph capture+replay. This isolates exactly what
    # graphing changes. Anything else (e.g. dynamic-cache vs StaticCache
    # prefill numerics) is a pre-existing property of the model/cache, not
    # something the graph introduced, so it must not be conflated in here.
    print(f"\n[PRIMARY] {cli.validate_steps} steps: StaticCache uncaptured vs whole-step graph replay")
    nc_toks, nc_logits, *_ = graph_decode(
        model, ids[:1], cli.validate_steps, dev, max_cache_len, batch_size=1, capture=False
    )
    g_toks, g_logits, g, s_in, s_pos, s_lg = graph_decode(
        model, ids[:1], cli.validate_steps, dev, max_cache_len, batch_size=1, capture=True
    )
    print(f"  uncaptured : {nc_toks}")
    print(f"  graph      : {g_toks}")
    match = nc_toks == g_toks
    max_logit_diff = max((a - b).abs().max().item() for a, b in zip(nc_logits, g_logits))
    print(f"  tokens identical: {match}   max |logit diff|: {max_logit_diff:.6f}")
    if not match:
        print("  *** GRAPH REPLAY DIVERGES FROM THE SAME UNCAPTURED PATH -- "
              "speed numbers below are NOT valid ***")

    # SECONDARY context only: eager + dynamic cache. A first-token difference
    # here reflects dynamic-vs-static cache prefill numerics (visible on the
    # INT4 model, whose SQNR is ~19dB), NOT the graph.
    ref_toks, ref_logits = eager_reference_logits(model, ids[:1], cli.validate_steps, dev)
    print(f"  [context] eager/dynamic-cache tokens: {ref_toks}")

    if cli.no_capture:
        print("\n[--no-capture] forward-path-only check done (no graph, no speed number).")
        return

    # ---- speed ----
    start_pos = ids.shape[1] + cli.validate_steps
    s_pos.fill_(start_pos)
    tps, ms = bench_graph_replay(g, s_in, s_pos, s_lg, start_pos, n_tokens=cli.bench_tokens)
    print(f"\n[{cli.mode}] WHOLE-STEP GRAPH decode (batch={cli.batch_size}): "
          f"{tps:.2f} tok/s   {ms:.3f} ms/token   weights={weights_gb:.3f}GB   "
          f"tokens_match_eager={match}")


if __name__ == "__main__":
    main()
