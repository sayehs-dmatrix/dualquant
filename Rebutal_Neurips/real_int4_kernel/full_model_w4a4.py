"""Full-model real W4A4 kernel: convert every projection in every layer to
use the fused Triton W4A4 kernel (real packed INT4 weight + real fused INT4
activation quantisation), then run real PPL + real generation throughput.

Read-only w.r.t. the rest of the repo -- only imports existing code, writes
only into this new folder / results/.

Usage:
    python full_model_w4a4.py --model meta-llama/Llama-3.2-1B
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_KERNEL_DIR = _os.path.join(_REPO_ROOT, "vendor", "dualscale_kernel_benchmark")
for p in (_CODEBASE_ROOT, _KERNEL_DIR, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import transformers

from main import load_model, build_cfg, _load_all_hyperparams, evaluate_ppl, _ATTN_NAMES, _MLP_NAMES
from methods import make_method
from fused_dual_scale_kernel import pack_int4_weights
from fused_w4a4_kernel import fused_w4a4_gemm
from e2e_bench_utils import prefill_bench, decode_bench
from wrap_cache import wrap_cached

import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root from this file

METHOD_CFG = os.path.join(_CODEBASE_ROOT, "configs", "methods.json")
GROUP_SIZE = 64   # matches block_size=64, the default used for the paper's reported PPL numbers


class RealW4A4Linear(nn.Module):
    """Drop-in replacement for a quantised nn.Linear: packs DualQuant's real
    mat_q into the kernel's INT4 format once, then runs the real fused
    kernel (packed weight + fused activation quant) at every forward call."""

    def __init__(self, mat_q: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        w_packed, scale1 = pack_int4_weights(mat_q.float(), GROUP_SIZE)
        self.register_buffer("w_packed", w_packed)
        self.register_buffer("scale1", scale1.to(torch.bfloat16))
        self.register_buffer("beta", beta.to(torch.bfloat16))

    def forward(self, x):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).to(torch.bfloat16)
        y2d = fused_w4a4_gemm(x2d, self.beta, self.w_packed, self.scale1, GROUP_SIZE)
        return y2d.reshape(*orig_shape[:-1], -1).to(x.dtype)


def _make_args(model_id, nsamples, seqlen):
    import types
    return types.SimpleNamespace(
        model=model_id, method="dualquant", method_cfg=METHOD_CFG,
        weight_fmt="rtn_int4", weight_block_size=GROUP_SIZE, weight_scale_format="none",
        weight_clip=False, act_quant="on", act_fmt="rtn_int4", act_block_size=GROUP_SIZE,
        act_scale_format="none", act_scaled_before_quant=True, act_quantize_bmm=False,
        preprocess="none", tasks=["wikitext"], nsamples=nsamples, seqlen=seqlen,
        results_path=None,
    )


def convert_model_to_real_kernel(model, cfg):
    """Quantise every projection with real DualQuant, then replace each with
    a RealW4A4Linear using the actual quantised weight + beta."""
    method = make_method("dualquant")
    n_layers = len(model.model.layers)
    for layer_idx, block in enumerate(model.model.layers):
        for proj_name in _ATTN_NAMES:
            parent = block.self_attn
            module = getattr(parent, proj_name)
            K = module.weight.shape[1]
            if K % GROUP_SIZE != 0:
                print(f"  [SKIP kernel-conv] layer{layer_idx}.{proj_name}: K={K} not div by {GROUP_SIZE}")
                continue
            beta = wrap_cached(method, module, cfg, f"layer{layer_idx}.{proj_name}", cfg["_model_id"], GROUP_SIZE)
            real_module = RealW4A4Linear(module.weight.detach(), beta.detach())
            setattr(parent, proj_name, real_module)
        for proj_name in _MLP_NAMES:
            parent = block.mlp
            module = getattr(parent, proj_name)
            K = module.weight.shape[1]
            if K % GROUP_SIZE != 0:
                print(f"  [SKIP kernel-conv] layer{layer_idx}.{proj_name}: K={K} not div by {GROUP_SIZE}")
                continue
            beta = wrap_cached(method, module, cfg, f"layer{layer_idx}.{proj_name}", cfg["_model_id"], GROUP_SIZE)
            real_module = RealW4A4Linear(module.weight.detach(), beta.detach())
            setattr(parent, proj_name, real_module)
        print(f"  layer {layer_idx}/{n_layers} converted to real W4A4 kernel", flush=True)


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
    cfg["_model_id"] = cli.model

    print(f"Loading {cli.model}...")
    model = load_model(cli.model)
    model.config.use_cache = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(cli.model, use_fast=False)

    print("Converting all projections to real fused W4A4 kernel...")
    t0 = time.time()
    convert_model_to_real_kernel(model, cfg)
    print(f"Conversion done in {time.time()-t0:.1f}s")

    torch.cuda.reset_peak_memory_stats()
    model.to(dev).eval()
    torch.cuda.synchronize()
    weights_mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after loading packed W4A4 weights: {weights_mem_gb:.3f}GB")

    if not cli.skip_ppl:
        print("\nRunning real PPL evaluation (wikitext) through the real kernel...")
        ppl = evaluate_ppl(model, tokenizer, args)
        print(f"\nReal-kernel W4A4 PPL: {ppl}")
    else:
        print("\n[skip-ppl] skipping PPL evaluation")

    # ── Real prefill latency/throughput/memory (compute/bandwidth-bound, batch>1) ──
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    for bs in (1, 16):
        r = prefill_bench(model, ids, batch_size=bs)
        print(f"\nReal-kernel prefill (batch={bs}, seq_len={r['seq_len']}): "
              f"latency={r['latency_ms']:.3f}ms  tokens/sec={r['tokens_per_sec']:.1f}  "
              f"peak_mem={r['peak_mem_gb']:.3f}GB")

    # ── Real decode generation throughput/latency/memory (batch=1, autoregressive) ──
    d = decode_bench(model, ids, tokenizer)
    print(f"\nReal-kernel decode generation: tokens/sec={d['tokens_per_sec']:.2f}  "
          f"latency/call={d['latency_per_call_s']:.3f}s  peak_mem={d['peak_mem_gb']:.3f}GB  "
          f"weights_mem={weights_mem_gb:.3f}GB")


if __name__ == "__main__":
    main()
