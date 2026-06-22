"""Sweep SmoothQuant alpha with quantization across 6 models and 4 scenarios.

Scenarios (all use RTN method; INT4 = per-group-64, MXFP4 = block-32):
  0  SQ + INT4  W4A16   weights only
  1  SQ + INT4  W4A4    weights + activations
  2  SQ + MXFP4 W4A16   weights only
  3  SQ + MXFP4 W4A4    weights + activations

Fresh model load for every (model, scenario, alpha) run — necessary because
W4A4 installs PermLinear wrappers that structurally change the model.

Results: results_sq_alpha_sweep/sq_alpha_quant_sweep.csv
         columns: model, scenario, alpha, ppl_wikitext

Usage:
    python sweep_sq_alpha_quant.py
    python sweep_sq_alpha_quant.py --models meta-llama/Llama-3.2-1B Qwen/Qwen3-0.6B
    python sweep_sq_alpha_quant.py --alphas 0.3 0.5 0.7
    python sweep_sq_alpha_quant.py --scenarios 0 2   # W4A16 only
"""

import sys, os as _os
_CODEBASE = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_CODEBASE, _os.path.join(_CODEBASE, "legacy_vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import csv
import gc
import os
import types

import torch
import transformers

import datautils
import eval_utils
from main import load_model, build_cfg, quantise_model, _load_all_hyperparams
from preprocess.smoothquant import apply_smoothquant

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CODEBASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

_ALL_MODELS = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-14B",
]

_SQ_SCALES = {
    "meta-llama/Llama-3.1-8B":   "smoothquant/act_scales/Llama-3.1-8B_seq_len_8192_4bit.pt",
    "meta-llama/Llama-3.2-1B":   "smoothquant/act_scales/Llama-3.2-1B_seq_len_8192_4bit.pt",
    "meta-llama/Llama-2-7b-hf":  "smoothquant/act_scales/Meta-Llama-2-7B_seqlen_4096_4bit.pt",
    "Qwen/Qwen3-0.6B":           "smoothquant/act_scales/Qwen3-0.6B_seq_len_8192_4bit.pt",
    "Qwen/Qwen2.5-7B":           "smoothquant/act_scales/Qwen2.5-7B_seq_len_8192_4bit.pt",
    "Qwen/Qwen3-14B":            "smoothquant/act_scales/Qwen3-14B_seq_len_8192_4bit.pt",
}

# Each scenario dict maps 1-to-1 to the CLI flags that main.py would receive.
SCENARIOS = [
    dict(
        label="SQ+INT4 (W4A16)",
        method="rtn",
        weight_fmt="rtn_int4", weight_block_size=64, weight_scale_format="none",
        act_quant="off",  act_fmt=None,      act_block_size=64, act_scale_format="none",
        act_scaled_before_quant=False, act_quantize_bmm=False,
    ),
    dict(
        label="SQ+INT4 (W4A4)",
        method="rtn",
        weight_fmt="rtn_int4", weight_block_size=64, weight_scale_format="none",
        act_quant="on",   act_fmt="rtn_int4", act_block_size=64, act_scale_format="none",
        act_scaled_before_quant=True,  act_quantize_bmm=False,
    ),
    dict(
        label="SQ+MXFP4 (W4A16)",
        method="rtn",
        weight_fmt="mxfp4",    weight_block_size=32, weight_scale_format="e8m0",
        act_quant="off",  act_fmt=None,       act_block_size=32, act_scale_format="e8m0",
        act_scaled_before_quant=False, act_quantize_bmm=False,
    ),
    dict(
        label="SQ+MXFP4 (W4A4)",
        method="rtn",
        weight_fmt="mxfp4",    weight_block_size=32, weight_scale_format="e8m0",
        act_quant="on",   act_fmt="mxint4",   act_block_size=32, act_scale_format="e8m0",
        act_scaled_before_quant=True,  act_quantize_bmm=False,
    ),
]

METHOD_CFG  = os.path.join(CODEBASE_DIR, "configs", "methods.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_sq_alpha_sweep")
NSAMPLES    = 128
SEQLEN      = 2048


def _csv_path(model_id):
    """One CSV per model, e.g. results_sq_alpha_sweep/meta-llama_Llama-3.1-8B.csv"""
    return os.path.join(RESULTS_DIR, model_id.replace("/", "_") + ".csv")


def _make_args(model_id, scen):
    """Build the SimpleNamespace that build_cfg / quantise_model expect."""
    return types.SimpleNamespace(
        model=model_id,
        method=scen["method"],
        attn_layers=True,
        mlp_layers=True,
        preprocess="smoothquant",       # prevents the quarot branch in quantise_model
        weight_fmt=scen["weight_fmt"],
        weight_block_size=scen["weight_block_size"],
        weight_scale_format=scen["weight_scale_format"],
        weight_clip=False,
        act_quant=scen["act_quant"],
        act_fmt=scen["act_fmt"],
        act_block_size=scen["act_block_size"],
        act_scale_format=scen["act_scale_format"],
        act_scaled_before_quant=scen["act_scaled_before_quant"],
        act_quantize_bmm=scen["act_quantize_bmm"],
    )


def _eval_ppl(model, tokenizer, model_id):
    dev = torch.device("cuda:0")
    _, _, _, testenc, _ = datautils.get_loaders(
        "wikitext", model_id, nsamples=NSAMPLES, seed=0, seqlen=SEQLEN
    )
    return float(eval_utils.evaluator_newVersion(
        model, tokenizer, testenc, dev, None, dataset="test"
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=_ALL_MODELS,
                   help="HF model ids to sweep (default: all 6)")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[round(a * 0.1, 1) for a in range(11)],
                   help="Alpha values (default: 0.0 0.1 … 1.0)")
    p.add_argument("--scenarios", nargs="+", type=int,
                   default=list(range(len(SCENARIOS))),
                   help="Scenario indices: 0=INT4 W4A16  1=INT4 W4A4  "
                        "2=MXFP4 W4A16  3=MXFP4 W4A4  (default: all)")
    args = p.parse_args()

    all_hyperparams = _load_all_hyperparams(METHOD_CFG)
    selected = [SCENARIOS[i] for i in args.scenarios]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total = len(args.models) * len(selected) * len(args.alphas)
    count = 0

    for model_id in args.models:
        csv_path = _csv_path(model_id)
        write_header = not os.path.exists(csv_path)
        sq_scales_path = os.path.join(CODEBASE_DIR, _SQ_SCALES[model_id])
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, use_fast=False
        )

        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["model", "scenario", "alpha", "ppl_wikitext"]
            )
            if write_header:
                writer.writeheader()

            for scen in selected:
                for alpha in args.alphas:
                    count += 1
                    print(f"\n{'='*62}")
                    print(f" [{count}/{total}]  {scen['label']}  |  "
                          f"model={model_id}  alpha={alpha:.2f}")
                    print(f"{'='*62}")

                    model = load_model(model_id)
                    apply_smoothquant(model, {
                        "act_scales_path": sq_scales_path,
                        "alpha": alpha,
                        "weight_stat": "max_abs",
                    })

                    quant_args = _make_args(model_id, scen)
                    cfg = build_cfg(quant_args, all_hyperparams)
                    quantise_model(model, tokenizer, quant_args, cfg)
                    gc.collect()
                    torch.cuda.empty_cache()  # release calibration tensors before eval

                    ppl = _eval_ppl(model, tokenizer, model_id)
                    print(f"  WikiText-2 PPL = {ppl:.4f}")

                    writer.writerow({
                        "model": model_id,
                        "scenario": scen["label"],
                        "alpha": alpha,
                        "ppl_wikitext": round(ppl, 4),
                    })
                    fh.flush()

                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

        print(f"\nModel done. Results in: {csv_path}")

    print(f"\nAll done. Results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
