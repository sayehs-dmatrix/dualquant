"""Validate DualQuant's real, packed INT4 Triton kernel against fake-quant.

Read-only w.r.t. the rest of the repo: only imports from Dualquant_codebase_20260508
(main.py, methods/) and the existing Triton kernel file in
MSE_Reduction_Two_approache_All_DataFormats_20260410/.../DualScale_Kernel_Benchmark/
fused_dual_scale_kernel.py. Nothing outside this new folder is modified.

Mapping (verified against layer_wrapper_dualquant.py:530-536):
  DualQuant's wrap(..., act_quant_flag=True) sets layer.weight = mat_q (real-valued,
  "prescaled" weight) and returns col_scales, meant to be multiplied into the
  ACTIVATION at inference. This is exactly the kernel's model:
      y = (x * beta) @ dequant(W_int4, scale1)^T
  with beta = col_scales, and dequant(W_int4, scale1) approximating mat_q.
  So: pack mat_q (= module.weight.data after wrap()) via the kernel's own
  pack_int4_weights(), and pass col_scales straight through as beta.

Usage:
    python validate_real_kernel.py --model meta-llama/Llama-3.2-1B
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))          # Dualquant_codebase_20260508
_KERNEL_DIR = ("/home/coder/numrd/Quantization_Repo_July2025/"
               "MSE_Reduction_Two_approache_All_DataFormats_20260410/"
               "__Baselines_with_the_same_fils_as_MSE/DualScale_Kernel_Benchmark")
for p in (_CODEBASE_ROOT, _KERNEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import transformers

from main import load_model, build_cfg, _load_all_hyperparams
from methods import make_method
from formats import make_format
from fused_dual_scale_kernel import pack_int4_weights, fused_beta_int4_gemm, cuda_timer

METHOD_CFG = os.path.join(_CODEBASE_ROOT, "configs", "methods.json")
GROUP_SIZE = 32   # must match the kernel's hardcoded BLOCK_K


def _make_args(model_id):
    import types
    return types.SimpleNamespace(
        model=model_id,
        method="dualquant",
        method_cfg=METHOD_CFG,
        weight_fmt="rtn_int4",
        weight_block_size=GROUP_SIZE,     # match kernel's GROUP_SIZE requirement
        weight_scale_format="none",
        weight_clip=False,
        act_quant="on",
        act_fmt="rtn_int4",
        act_block_size=GROUP_SIZE,
        act_scale_format="none",
        act_scaled_before_quant=True,
        act_quantize_bmm=False,
    )


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

    W_true = module.weight.detach().clone().to(torch.float32)   # ground truth, pre-quant
    N, K = W_true.shape
    print(f"Projection: layer{cli.layer}.{cli.proj}  shape=({N},{K})")
    assert K % GROUP_SIZE == 0, f"K={K} not divisible by GROUP_SIZE={GROUP_SIZE}"

    # ── Real DualQuant quantisation (exactly as used everywhere else this session) ──
    method = make_method("dualquant")
    col_scales = method.wrap(module, cfg, layer_key=f"layer{cli.layer}.{cli.proj}")
    mat_q = module.weight.detach().clone().to(torch.float32)     # DualQuant's real "mat_q"
    beta = col_scales.detach().clone().to(torch.float32)         # DualQuant's real beta
    print(f"col_scales (beta): min={beta.min():.4f} max={beta.max():.4f} mean={beta.mean():.4f}")

    # ── Pack mat_q into the kernel's real INT4 format ──
    w_packed, scale1 = pack_int4_weights(mat_q, GROUP_SIZE)
    w_packed = w_packed.to(dev)
    scale1 = scale1.to(torch.bfloat16).to(dev)
    beta_bf16 = beta.to(torch.bfloat16).to(dev)

    # ── Real input activations (random, standing in for a real hidden state) ──
    torch.manual_seed(0)
    M = cli.batch_size
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)

    # ── Ground truth (no quantisation at all) ──
    y_true = (x.float() @ W_true.to(dev).T)

    def sqnr_db(ref, approx):
        sig = (ref ** 2).sum()
        noise = ((approx - ref) ** 2).sum()
        return 10.0 * torch.log10(sig / noise).item()

    # ═══════════════════ WEIGHT-ONLY (A16W4) — kernel as found ═══════════════════
    y_fake_quant_w = ((x.float() * beta.to(dev)) @ mat_q.to(dev).T)
    y_real_kernel_w = fused_beta_int4_gemm(x, beta_bf16, w_packed, scale1, GROUP_SIZE).float()

    print()
    print("=== Weight-only (A16W4): real kernel vs ground truth vs fake-quant ===")
    print(f"  SQNR(fake_quant   vs true) = {sqnr_db(y_true, y_fake_quant_w):.2f} dB")
    print(f"  SQNR(real_kernel  vs true) = {sqnr_db(y_true, y_real_kernel_w):.2f} dB")
    print(f"  SQNR(real_kernel  vs fake_quant) = {sqnr_db(y_fake_quant_w, y_real_kernel_w):.2f} dB")

    # ═══════════════════ FULL W4A4 — matches our paper's actual config ═══════════
    # DualQuant's real order: scale activation by beta, THEN quantise (act_fmt.cast).
    # Reusing the SAME rtn_int4 cast function used everywhere else in this codebase.
    act_fmt = make_format("rtn_int4", block_size=GROUP_SIZE)
    x_scaled = x.float() * beta.to(dev)
    x_quantised = act_fmt.cast(x_scaled).to(torch.bfloat16)   # real act quant noise applied

    ones_beta = torch.ones_like(beta_bf16)   # scaling already applied above -> no-op inside kernel

    y_fake_quant_wa = (x_quantised.float() @ mat_q.to(dev).T)
    y_real_kernel_wa = fused_beta_int4_gemm(
        x_quantised, ones_beta, w_packed, scale1, GROUP_SIZE
    ).float()

    print()
    print("=== Full W4A4 (matches paper's reported config): real kernel vs ground truth vs fake-quant ===")
    print(f"  SQNR(fake_quant   vs true) = {sqnr_db(y_true, y_fake_quant_wa):.2f} dB")
    print(f"  SQNR(real_kernel  vs true) = {sqnr_db(y_true, y_real_kernel_wa):.2f} dB")
    print(f"  SQNR(real_kernel  vs fake_quant) = {sqnr_db(y_fake_quant_wa, y_real_kernel_wa):.2f} dB"
          f"   <- how close the REAL kernel is to what we've been reporting as PPL/accuracy")

    # ── Real latency: fused kernel vs plain bf16 matmul (no quantisation) ──
    # NOTE: latency reflects the weight-side kernel (memory-bandwidth savings from
    # packed INT4 weights); the activation quantisation above is a numerical-fidelity
    # check only, computed outside the timed kernel call.
    def real_kernel_call():
        return fused_beta_int4_gemm(x, beta_bf16, w_packed, scale1, GROUP_SIZE)

    def bf16_baseline_call():
        return x @ W_true.to(dev).to(torch.bfloat16).T

    t_kernel = cuda_timer(real_kernel_call)
    t_baseline = cuda_timer(bf16_baseline_call)
    print()
    print("=== Real measured latency (single projection matmul; weight-side kernel benefit) ===")
    print(f"  BF16 baseline matmul : {t_baseline:.2f} us")
    print(f"  Real INT4 kernel     : {t_kernel:.2f} us")
    print(f"  Speedup              : {t_baseline / t_kernel:.2f}x")


if __name__ == "__main__":
    main()
