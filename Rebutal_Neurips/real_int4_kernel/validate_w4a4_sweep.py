"""Sweep the fused W4A4 kernel validation across multiple projections/layers,
loading the model once. Reports correctness (SQNR) and latency per (layer, proj).

Usage:
    python validate_w4a4_sweep.py --model meta-llama/Llama-3.2-1B
"""
import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_KERNEL_DIR = _os.path.join(_REPO_ROOT, "vendor", "dualscale_kernel_benchmark")
for p in (_CODEBASE_ROOT, _KERNEL_DIR, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from main import load_model, build_cfg, _load_all_hyperparams
from methods import make_method
from formats import make_format
from fused_dual_scale_kernel import pack_int4_weights, fused_beta_int4_gemm, cuda_timer
from fused_w4a4_kernel import fused_w4a4_gemm

import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root from this file

METHOD_CFG = os.path.join(_CODEBASE_ROOT, "configs", "methods.json")
GROUP_SIZE = 32
PROJECTIONS = [
    ("q_proj", "self_attn"), ("k_proj", "self_attn"),
    ("v_proj", "self_attn"), ("o_proj", "self_attn"),
]


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


def _target_layers(n_layers):
    mid = n_layers // 2
    idxs = {0, 1, mid - 1, mid, n_layers - 2, n_layers - 1}
    return sorted(i for i in idxs if 0 <= i < n_layers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--batch-size", type=int, default=16)
    cli = p.parse_args()

    dev = torch.device("cuda:0")
    args = _make_args(cli.model)
    all_hp = _load_all_hyperparams(args.method_cfg)
    cfg = build_cfg(args, all_hp)
    act_fmt = make_format("rtn_int4", block_size=GROUP_SIZE)

    print(f"Loading {cli.model}...")
    model = load_model(cli.model)
    n_layers = len(model.model.layers)
    target_layers = _target_layers(n_layers)
    print(f"Target layers: {target_layers}")

    torch.manual_seed(0)
    rows = []

    for layer_idx in target_layers:
        block = model.model.layers[layer_idx]
        for proj_name, parent_attr in PROJECTIONS:
            module = getattr(getattr(block, parent_attr), proj_name)
            W_true = module.weight.detach().clone().to(torch.float32)
            N, K = W_true.shape
            if K % GROUP_SIZE != 0:
                print(f"  [SKIP] layer{layer_idx}.{proj_name}: K={K} not divisible by {GROUP_SIZE}")
                continue

            method = make_method("dualquant")
            col_scales = method.wrap(module, cfg, layer_key=f"layer{layer_idx}.{proj_name}")
            mat_q = module.weight.detach().clone().to(torch.float32)
            beta = col_scales.detach().clone().to(torch.float32)

            w_packed, scale1 = pack_int4_weights(mat_q, GROUP_SIZE)
            w_packed, scale1 = w_packed.to(dev), scale1.to(torch.bfloat16).to(dev)
            beta_bf16 = beta.to(torch.bfloat16).to(dev)
            ones_beta = torch.ones_like(beta_bf16)

            M = cli.batch_size
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            y_true = x.float() @ W_true.to(dev).T

            x_scaled = x.float() * beta.to(dev)
            x_quantised = act_fmt.cast(x_scaled).to(torch.bfloat16)
            y_unfused = fused_beta_int4_gemm(x_quantised, ones_beta, w_packed, scale1, GROUP_SIZE).float()
            y_fused = fused_w4a4_gemm(x, beta_bf16, w_packed, scale1, GROUP_SIZE).float()

            sqnr_unfused = sqnr_db(y_true, y_unfused)
            sqnr_fused = sqnr_db(y_true, y_fused)

            def fused_call():
                return fused_w4a4_gemm(x, beta_bf16, w_packed, scale1, GROUP_SIZE)

            def bf16_call():
                return x @ W_true.to(dev).to(torch.bfloat16).T

            t_fused = cuda_timer(fused_call, warmup=20, repeats=100)
            t_base = cuda_timer(bf16_call, warmup=20, repeats=100)

            row = dict(
                layer=layer_idx, proj=proj_name, N=N, K=K,
                sqnr_unfused=round(sqnr_unfused, 2), sqnr_fused=round(sqnr_fused, 2),
                t_fused_us=round(t_fused, 2), t_baseline_us=round(t_base, 2),
                speedup=round(t_base / t_fused, 2),
            )
            rows.append(row)
            print(f"  layer{layer_idx:>2} {proj_name:<8} N={N:5d} K={K:5d}  "
                  f"SQNR(unfused)={sqnr_unfused:6.2f}dB  SQNR(fused)={sqnr_fused:6.2f}dB  "
                  f"speedup={row['speedup']:.2f}x", flush=True)

    out_csv = os.path.join(_HERE, "w4a4_sweep_results.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_sqnr_unfused = sum(r["sqnr_unfused"] for r in rows) / len(rows)
    mean_sqnr_fused = sum(r["sqnr_fused"] for r in rows) / len(rows)
    mean_speedup = sum(r["speedup"] for r in rows) / len(rows)
    print()
    print(f"=== Summary over {len(rows)} (layer, proj) combos ===")
    print(f"  mean SQNR(unfused vs true) = {mean_sqnr_unfused:.2f} dB")
    print(f"  mean SQNR(fused   vs true) = {mean_sqnr_fused:.2f} dB")
    print(f"  mean speedup (fused vs bf16 baseline) = {mean_speedup:.2f}x")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
