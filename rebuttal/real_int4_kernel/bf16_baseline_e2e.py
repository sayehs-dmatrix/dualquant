"""BF16 baseline end-to-end numbers, measured the same way as
full_model_w4a4.py (real PPL + real generation throughput/latency/memory),
so the two are directly comparable on the same hardware / same process
structure. No quantization at all -- this is the reference row.

Read-only w.r.t. the rest of the repo -- only imports existing code.

Usage:
    python bf16_baseline_e2e.py --model meta-llama/Llama-3.1-8B
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_CODEBASE_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import transformers

from main import load_model, build_cfg, _load_all_hyperparams, evaluate_ppl
from e2e_bench_utils import prefill_bench, decode_bench

METHOD_CFG = os.path.join(_CODEBASE_ROOT, "configs", "methods.json")


def _make_args(model_id, nsamples, seqlen):
    import types
    return types.SimpleNamespace(
        model=model_id, method="dualquant", method_cfg=METHOD_CFG,
        weight_fmt="rtn_int4", weight_block_size=64, weight_scale_format="none",
        weight_clip=False, act_quant="off", act_fmt=None, act_block_size=64,
        act_scale_format="none", act_scaled_before_quant=True, act_quantize_bmm=False,
        preprocess="none", tasks=["wikitext"], nsamples=nsamples, seqlen=seqlen,
        results_path=None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--skip-ppl", action="store_true")
    cli = p.parse_args()

    dev = torch.device("cuda:0")
    args = _make_args(cli.model, cli.nsamples, cli.seqlen)
    all_hp = _load_all_hyperparams(args.method_cfg)
    cfg = build_cfg(args, all_hp)

    print(f"Loading {cli.model} (BF16, no quantization)...")
    torch.cuda.reset_peak_memory_stats()
    model = load_model(cli.model)
    model.config.use_cache = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(cli.model, use_fast=False)

    model.to(dev).eval()
    torch.cuda.synchronize()
    weights_mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after loading BF16 weights: {weights_mem_gb:.3f}GB")

    if not cli.skip_ppl:
        print("\nRunning real PPL evaluation (wikitext) on the BF16 model...")
        ppl = evaluate_ppl(model, tokenizer, args)
        print(f"\nBF16 baseline PPL: {ppl}")
    else:
        print("\n[skip-ppl] skipping PPL evaluation")

    # ── Real prefill latency/throughput/memory (same recipe as full_model_w4a4.py) ──
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    for bs in (1, 16):
        r = prefill_bench(model, ids, batch_size=bs)
        print(f"\nBF16 baseline prefill (batch={bs}, seq_len={r['seq_len']}): "
              f"latency={r['latency_ms']:.3f}ms  tokens/sec={r['tokens_per_sec']:.1f}  "
              f"peak_mem={r['peak_mem_gb']:.3f}GB")

    # ── Real decode generation throughput/latency/memory ──
    for dbs in (1, 16):
        d = decode_bench(model, ids, tokenizer, batch_size=dbs)
        print(f"\nBF16 baseline decode generation (batch={dbs}): tokens/sec={d['tokens_per_sec']:.2f}  "
              f"latency/call={d['latency_per_call_s']:.3f}s  peak_mem={d['peak_mem_gb']:.3f}GB  "
              f"weights_mem={weights_mem_gb:.3f}GB")


if __name__ == "__main__":
    main()
