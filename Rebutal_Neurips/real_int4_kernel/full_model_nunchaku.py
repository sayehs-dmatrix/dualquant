"""Full-model real W4A4 kernel using MIT's nunchaku (SVDQuant) CUTLASS
INT4 GEMM (github.com/mit-han-lab/nunchaku), instead of the hand-rolled
Triton kernel in fused_w4a4_kernel.py. Same DualQuant BCD optimization
(method.wrap) produces mat_q + beta per projection; beta is loaded as
nunchaku's "smooth" factor (smooth = 1/beta, since nunchaku divides
activations by smooth while DualQuant's beta is meant to multiply).

Usage:
    python full_model_nunchaku.py --model meta-llama/Llama-3.1-8B
"""
import argparse
import contextlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_NUNCHAKU_DIR = ("/tmp/claude-0/-root-numrd/1e334428-137f-442b-9669-4bc5945263d0/"
                 "scratchpad/svdquant_repo/build/lib.linux-x86_64-cpython-312")
for p in (_CODEBASE_ROOT, _HERE, _NUNCHAKU_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import transformers
import nunchaku_min

from main import load_model, build_cfg, _load_all_hyperparams, evaluate_ppl, _ATTN_NAMES, _MLP_NAMES
from methods import make_method
from e2e_bench_utils import prefill_bench, decode_bench
from wrap_cache import wrap_cached

METHOD_CFG = os.path.join(_CODEBASE_ROOT, "configs", "methods.json")
GROUP_SIZE = 64


class NunchakuW4A4Linear(nn.Module):
    """Drop-in replacement for a quantised nn.Linear using nunchaku's real
    CUTLASS W4A4 GEMM. DualQuant's mat_q is packed via nunchaku's own
    quantize_w4a4_wgt kernel (correct tiled layout, no hand-rolled packing).

    IMPORTANT: nunchaku's internal "smooth" mechanism was built for
    SmoothQuant-style mild per-channel rescaling; DualQuant's real beta
    spans up to ~34x per layer, which collapses nunchaku's per-64-group
    INT4 activation quantization when routed through smooth (verified:
    full-model PPL went from 6.73 to 26905 -- a broken model). Fix: use
    nunchaku PURELY as the INT4x INT4 GEMM engine (smooth left at 1, a
    no-op) and apply DualQuant's beta ourselves as a plain elementwise
    multiply before quantization, exactly like the validated Triton path.
    Verified on a real layer: 19.19dB SQNR (clean, matches quant noise floor).

    NOTE on a path tried and reverted: nunchaku also bundles an AWQ/TensorRT-
    LLM-derived weight-only INT4 GEMV kernel (gemv_awq), purpose-built for
    small-M decode and measured up to 4.5x over BF16 in isolation on real
    Llama-3.1-8B MLP shapes (vs W4A4's 1.1-1.7x there) by skipping activation
    quantization entirely. Wiring it in as an M<=4 fast path was a NET
    REGRESSION end-to-end (decode 28.69 vs 32.94 tok/s, plus +4GB weight
    memory for the duplicate AWQ-packed copy): torch.profiler on a real decode
    step showed GPU kernel time is only ~8ms/step against ~36.5ms/step of CPU
    time, i.e. decode is CPU-dispatch-bound, not GPU-kernel-bound -- the eager
    gemv_awq call added Python-level orchestration overhead that outweighed
    its GPU-side win, while forward_graph_beta's single graph-launch call
    stays cheaper on the CPU side despite doing "more" GPU work per call. See
    awq_gemv_pack.py (kept as a validated reference/artifact, not imported
    here) for the kernel-level finding and the beta-folding trick used."""

    def __init__(self, mat_q: torch.Tensor, beta: torch.Tensor, device_id=0):
        super().__init__()
        N, K = mat_q.shape
        assert N % 128 == 0 and K % 128 == 0, f"nunchaku needs N,K multiples of 128, got {N},{K}"
        dev = torch.device(f"cuda:{device_id}")
        self.gemm = nunchaku_min.QuantizedGEMM()
        self.gemm.init(K, N, False, True, device_id)
        self.gemm.load_weight(mat_q.to(dev).to(torch.bfloat16).contiguous())
        self.gemm.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))
        self.register_buffer("beta", beta.to(dev).to(torch.bfloat16).contiguous())

    _FWD_MODE = os.environ.get("NUNCHAKU_FWD_MODE", "graph_beta")

    def forward(self, x):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).to(torch.bfloat16).contiguous()
        if self._FWD_MODE == "nocache":
            # Memory-lean path for long-context evaluation (e.g. PPL at seqlen 8192).
            # The fast paths keep PERSISTENT per-layer scratch (cached_act / cached_out
            # etc.) sized for the current M; at M=8192 cached_out alone is
            # 8192*14336*2 = 235 MB for gate/up, which across 224 layers is ~15 GB and
            # OOMs a 24GB card. QuantizedGEMM::forward allocates and frees per call
            # instead, so peak memory stays ~one layer's worth. beta is applied here as
            # a plain multiply (this is the originally validated arrangement, 19.0 dB
            # SQNR); it costs an extra kernel per layer, which is irrelevant for an
            # accuracy run.
            y2d = self.gemm.forward((x2d * self.beta).contiguous())
        elif self._FWD_MODE == "static":
            # For use INSIDE an outer whole-model CUDA graph capture
            # (whole_step_graph_decode.py). forward_static pushes the caller's
            # current stream onto nunchaku's internal stream stack so its
            # kernels land on the capture stream and are actually recorded.
            # Using forward_beta here instead would silently launch them on
            # legacy stream 0, leaving them OUT of the graph -- which shows up
            # as garbage output plus an impossibly fast replay (below the
            # INT4 memory-bound floor).
            y2d = self.gemm.forward_static(x2d, self.beta)
        elif self._FWD_MODE == "graph_beta":
            # forward_graph_beta captures a CUDA graph, which cannot happen on
            # the legacy/default stream 0 -- it requires (and asserts) that
            # the caller's current stream is a real, non-default
            # torch.cuda.Stream(). main() wraps the whole forward pass in
            # `with torch.cuda.stream(GRAPH_STREAM):` for exactly this reason.
            y2d = self.gemm.forward_graph_beta(x2d, self.beta)
        else:
            y2d = self.gemm.forward_beta(x2d, self.beta)
        return y2d.reshape(*orig_shape[:-1], -1).to(x.dtype)


# Shared across all NunchakuW4A4Linear instances: forward_graph_beta requires
# a single consistent non-default stream (each layer's QuantizedGEMM captures
# its own graph against whichever stream is current at first use).
GRAPH_STREAM = torch.cuda.Stream()


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


def convert_model_to_nunchaku(model, cfg):
    method = make_method("dualquant")
    n_layers = len(model.model.layers)
    for layer_idx, block in enumerate(model.model.layers):
        for proj_name in _ATTN_NAMES:
            parent = block.self_attn
            module = getattr(parent, proj_name)
            N, K = module.weight.shape
            if K % 128 != 0 or N % 128 != 0:
                print(f"  [SKIP] layer{layer_idx}.{proj_name}: shape=({N},{K}) not div by 128")
                continue
            beta = wrap_cached(method, module, cfg, f"layer{layer_idx}.{proj_name}",
                                cfg["_model_id"], GROUP_SIZE)
            real_module = NunchakuW4A4Linear(module.weight.detach(), beta.detach())
            setattr(parent, proj_name, real_module)
        for proj_name in _MLP_NAMES:
            parent = block.mlp
            module = getattr(parent, proj_name)
            N, K = module.weight.shape
            if K % 128 != 0 or N % 128 != 0:
                print(f"  [SKIP] layer{layer_idx}.{proj_name}: shape=({N},{K}) not div by 128")
                continue
            beta = wrap_cached(method, module, cfg, f"layer{layer_idx}.{proj_name}",
                                cfg["_model_id"], GROUP_SIZE)
            real_module = NunchakuW4A4Linear(module.weight.detach(), beta.detach())
            setattr(parent, proj_name, real_module)
        print(f"  layer {layer_idx}/{n_layers} converted to nunchaku W4A4", flush=True)


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

    torch.cuda.empty_cache()
    free0, total = torch.cuda.mem_get_info()
    baseline_used_gb = (total - free0) / 1e9

    print(f"Loading {cli.model}...")
    model = load_model(cli.model)
    model.config.use_cache = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(cli.model, use_fast=False)

    print("Converting all projections to nunchaku real W4A4 kernel...")
    t0 = time.time()
    convert_model_to_nunchaku(model, cfg)
    print(f"Conversion done in {time.time()-t0:.1f}s")

    torch.cuda.reset_peak_memory_stats()
    model.to(dev).eval()
    torch.cuda.synchronize()
    # NOTE: torch.cuda.memory_allocated() only tracks PyTorch's own allocator.
    # nunchaku's Tensor::allocate uses cudaMallocAsync directly, invisible to
    # that counter. Use mem_get_info (whole-GPU accounting) for the true figure.
    free1, _ = torch.cuda.mem_get_info()
    true_used_gb = (total - free1) / 1e9
    weights_mem_gb = true_used_gb - baseline_used_gb
    print(f"GPU memory after loading nunchaku W4A4 weights (true, via mem_get_info): "
          f"{weights_mem_gb:.3f}GB  (torch-allocator-only would have shown "
          f"{torch.cuda.memory_allocated()/1e9:.3f}GB -- wrong, misses nunchaku's cudaMallocAsync buffers)")

    # forward_graph_beta requires a real non-default stream to capture its
    # CUDA graph on (asserts this and throws otherwise). forward_beta must
    # NOT be run under one: its quantize/gemm calls go through nunchaku's own
    # internal stream stack (empty -> legacy stream 0) regardless of
    # PyTorch's current stream, so forcing a custom stream around it would
    # introduce a real cross-stream race between its beta-multiply (which DOES
    # follow PyTorch's current stream) and those internal calls.
    stream_ctx = torch.cuda.stream(GRAPH_STREAM) if NunchakuW4A4Linear._FWD_MODE == "graph_beta" else contextlib.nullcontext()

    with stream_ctx:
        if not cli.skip_ppl:
            print("\nRunning real PPL evaluation (wikitext) through nunchaku kernel...")
            ppl = evaluate_ppl(model, tokenizer, args)
            print(f"\nNunchaku-kernel W4A4 PPL: {ppl}")
        else:
            print("\n[skip-ppl] skipping PPL evaluation")

        prompt = "The quick brown fox jumps over the lazy dog. " * 8
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
        for bs in (1, 16):
            r = prefill_bench(model, ids, batch_size=bs)
            print(f"\nNunchaku-kernel prefill (batch={bs}, seq_len={r['seq_len']}): "
                  f"latency={r['latency_ms']:.3f}ms  tokens/sec={r['tokens_per_sec']:.1f}  "
                  f"peak_mem={r['peak_mem_gb']:.3f}GB")

        for dbs in (1, 16):
            d = decode_bench(model, ids, tokenizer, batch_size=dbs)
            print(f"\nNunchaku-kernel decode generation (batch={dbs}): tokens/sec={d['tokens_per_sec']:.2f}  "
                  f"latency/call={d['latency_per_call_s']:.3f}s  peak_mem={d['peak_mem_gb']:.3f}GB  "
                  f"weights_mem={weights_mem_gb:.3f}GB")


if __name__ == "__main__":
    main()
