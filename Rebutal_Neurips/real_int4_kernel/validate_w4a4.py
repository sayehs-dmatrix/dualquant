"""Validate the fused A4W4 kernel: correctness vs ground truth / existing
fake-quant, and real end-to-end latency (fused single-launch vs the unfused
weight-kernel-plus-separate-activation-quantise pipeline vs plain bf16).

Usage:
    python validate_w4a4.py --model meta-llama/Llama-3.2-1B
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_KERNEL_DIR = ("/home/coder/numrd/Quantization_Repo_July2025/"
               "MSE_Reduction_Two_approache_All_DataFormats_20260410/"
               "__Baselines_with_the_same_fils_as_MSE/DualScale_Kernel_Benchmark")
for p in (_CODEBASE_ROOT, _KERNEL_DIR, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from main import load_model, build_cfg, _load_all_hyperparams
from methods import make_method
from formats import make_format
from fused_dual_scale_kernel import pack_int4_weights, fused_beta_int4_gemm, cuda_timer
from fused_w4a4_kernel import fused_w4a4_gemm

METHOD_CFG = os.path.join(_CODEBASE_ROOT, "configs", "methods.json")
GROUP_SIZE = 32


def _make_args(model_id):
    import types
    return types.SimpleNamespace(
        model=model_id, method="dualquant", method_cfg=METHOD_CFG,
        weight_fmt="rtn_int4", weight_block_size=GROUP_SIZE, weight_scale_format="none",
        weight_clip=False, act_quant="on", act_fmt="rtn_int4", act_block_size=GROUP_SIZE,
        act_scale_format="none", act_scaled_before_quant=True, act_quantize_bmm=False,
    )


def sqnr_db(ref, approx):
    sig = (ref ** 2).sum()
    noise = ((approx - ref) ** 2).sum()
    return 10.0 * torch.log10(sig / noise).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--proj", default="q_proj", choices=["q_proj", "k_proj", "v_proj", "o_proj"])
    p.add_argument("--batch-size", type=int, default=16)
    cli = p.parse_args()

    dev = torch.device("cuda:0")
    args = _make_args(cli.model)
    all_hp = _load_all_hyperparams(args.method_cfg)
    cfg = build_cfg(args, all_hp)

    print(f"Loading {cli.model}...")
    model = load_model(cli.model)
    block = model.model.layers[cli.layer]
    module = getattr(block.self_attn, cli.proj)

    W_true = module.weight.detach().clone().to(torch.float32)
    N, K = W_true.shape
    print(f"Projection: layer{cli.layer}.{cli.proj}  shape=({N},{K})")

    method = make_method("dualquant")
    col_scales = method.wrap(module, cfg, layer_key=f"layer{cli.layer}.{cli.proj}")
    mat_q = module.weight.detach().clone().to(torch.float32)
    beta = col_scales.detach().clone().to(torch.float32)

    w_packed, scale1 = pack_int4_weights(mat_q, GROUP_SIZE)
    w_packed, scale1 = w_packed.to(dev), scale1.to(torch.bfloat16).to(dev)
    beta_bf16 = beta.to(torch.bfloat16).to(dev)
    ones_beta = torch.ones_like(beta_bf16)

    torch.manual_seed(0)
    M = cli.batch_size
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)

    y_true = x.float() @ W_true.to(dev).T

    # ── Reference W4A4 (unfused): our existing act_fmt.cast, then weight-only kernel ──
    act_fmt = make_format("rtn_int4", block_size=GROUP_SIZE)
    x_scaled = x.float() * beta.to(dev)
    x_quantised = act_fmt.cast(x_scaled).to(torch.bfloat16)
    y_unfused_kernel = fused_beta_int4_gemm(x_quantised, ones_beta, w_packed, scale1, GROUP_SIZE).float()

    # ── NEW: fully fused W4A4 kernel, single launch ──
    y_fused_kernel = fused_w4a4_gemm(x, beta_bf16, w_packed, scale1, GROUP_SIZE).float()

    print()
    print("=== Correctness: fused-kernel W4A4 vs ground truth vs unfused-pipeline W4A4 ===")
    print(f"  SQNR(unfused_pipeline vs true) = {sqnr_db(y_true, y_unfused_kernel):.2f} dB")
    print(f"  SQNR(fused_kernel     vs true) = {sqnr_db(y_true, y_fused_kernel):.2f} dB")
    print(f"  SQNR(fused_kernel vs unfused_pipeline) = {sqnr_db(y_unfused_kernel, y_fused_kernel):.2f} dB")

    # ── Real latency: three paths ──
    def fused_call():
        return fused_w4a4_gemm(x, beta_bf16, w_packed, scale1, GROUP_SIZE)

    def unfused_call():
        # separate (unfused) activation quantise step, then weight kernel
        xs = x.float() * beta.to(dev)
        xq = act_fmt.cast(xs).to(torch.bfloat16)
        return fused_beta_int4_gemm(xq, ones_beta, w_packed, scale1, GROUP_SIZE)

    def bf16_baseline_call():
        return x @ W_true.to(dev).to(torch.bfloat16).T

    t_fused = cuda_timer(fused_call)
    t_unfused = cuda_timer(unfused_call)
    t_baseline = cuda_timer(bf16_baseline_call)

    print()
    print("=== Real measured latency (single projection, includes activation quant) ===")
    print(f"  BF16 baseline (no quant)               : {t_baseline:.2f} us")
    print(f"  Unfused (separate act-quant + kernel)   : {t_unfused:.2f} us  "
          f"(speedup vs baseline: {t_baseline / t_unfused:.2f}x)")
    print(f"  Fused W4A4 kernel (single launch)       : {t_fused:.2f} us  "
          f"(speedup vs baseline: {t_baseline / t_fused:.2f}x)")
    print(f"  Fusion benefit (unfused / fused)        : {t_unfused / t_fused:.2f}x")


if __name__ == "__main__":
    main()
