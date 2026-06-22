"""Downstream-task evaluation entrypoint.

Same quantisation pipeline as main.py, but evaluates with lm-evaluation-harness
instead of perplexity.

Usage:
    python main_downstream.py \\
        --model meta-llama/Llama-3.2-1B \\
        --method rtn \\
        --weight-fmt rtn_int4 --weight-block-size 64 --weight-scale-format none \\
        --tasks arc_easy arc_challenge hellaswag winogrande piqa \\
        --nshots 0 \\
        --results-path results/downstream_rtn_int4.csv

    python main_downstream.py \\
        --model meta-llama/Llama-3.2-1B \\
        --method dualquant --method-cfg configs/methods.json \\
        --weight-fmt rtn_int4 --weight-block-size 64 --weight-scale-format none \\
        --preprocess smoothquant --sq-scales smoothquant/act_scales/Llama-3.2-1B_seq_len_8192_4bit.pt \\
        --tasks arc_easy arc_challenge hellaswag winogrande piqa \\
        --nshots 0 \\
        --results-path results/downstream_dq.csv
"""

import _legacy_path  # noqa: F401

import argparse
import csv
import json
import os

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from formats import make_format
from methods import make_method
from preprocess import apply_preprocess
from activations import install_perm_linear
from data import calib_acts_for, collect_calib_activations

import lm_eval
from lm_eval.models.huggingface import HFLM


_ATTN_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_NAMES  = ("up_proj", "down_proj", "gate_proj")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()

    # ---- model ----
    p.add_argument("--model", required=True)

    # ---- method ----
    p.add_argument("--method", required=True)
    p.add_argument("--method-cfg", default=None)

    # ---- weight format ----
    p.add_argument("--weight-fmt", required=True)
    p.add_argument("--weight-block-size", type=int, default=32)
    p.add_argument("--weight-scale-format", default="e8m0",
                   choices=["e8m0", "e4m3", "e4m4", "e5m3", "none"])
    p.add_argument("--weight-clip", action="store_true")

    # ---- activation quantisation ----
    p.add_argument("--act-quant", choices=["on", "off"], default="off")
    p.add_argument("--act-fmt", default=None)
    p.add_argument("--act-block-size", type=int, default=32)
    p.add_argument("--act-scale-format", default="e8m0",
                   choices=["e8m0", "e4m3", "e4m4", "e5m3", "none"])
    p.add_argument("--act-scaled-before-quant", action="store_true")
    p.add_argument("--act-quantize-bmm", action="store_true")

    # ---- preprocess ----
    p.add_argument("--preprocess", default="none",
                   choices=["none", "smoothquant", "quarot"])
    p.add_argument("--sq-scales", default=None)

    # ---- downstream eval ----
    p.add_argument("--tasks", nargs="+",
                   default=["arc_easy", "arc_challenge", "hellaswag",
                            "winogrande", "piqa"],
                   help="lm_eval task names")
    p.add_argument("--nshots", type=int, default=0,
                   help="few-shot count passed to lm_eval (default: 0)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="batch size for lm_eval inference")

    # ---- output ----
    p.add_argument("--results-path", default=None)

    return p.parse_args()


def _load_hyperparams(path):
    if path is None:
        return {}
    with open(path) as fh:
        return json.load(fh)


def build_cfg(args, all_hyperparams):
    method_cfg = all_hyperparams.get(args.method, {})
    weight_fmt  = make_format(args.weight_fmt, block_size=args.weight_block_size,
                              mse=args.weight_clip)
    act_quant = {"enabled": False}
    if args.act_quant == "on":
        if args.act_fmt is None:
            raise SystemExit("--act-quant on requires --act-fmt")
        act_quant = {
            "enabled": True,
            "fmt": make_format(args.act_fmt, block_size=args.act_block_size),
            "scale_format": args.act_scale_format,
            "scaled_before_quant": args.act_scaled_before_quant,
            "quantize_bmm": args.act_quantize_bmm,
        }
    return {
        "method_cfg":         method_cfg,
        "weight_fmt":         weight_fmt,
        "weight_scale_format": args.weight_scale_format,
        "act_quant":          act_quant,
    }


def load_model(model_id):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    if not hasattr(cfg, "mlp_bias"):
        cfg.mlp_bias = False
    if "llama" in model_id.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_id, config=cfg, torch_dtype="auto", low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=cfg, torch_dtype="auto", low_cpu_mem_usage=True
        )
    model.config.use_cache = False
    model.eval()
    return model


def quantise_model(model, tokenizer, args, cfg):
    method = make_method(args.method)

    if hasattr(method, "quantise_model"):
        method.quantise_model(model, tokenizer, args, cfg)
        return

    activations = None
    if method.needs_calib_acts:
        print(f"collecting calibration activations for {method.name}...")
        activations = collect_calib_activations(model, tokenizer)
        print(f"collected activations for {len(activations)} layers")

    act_cfg    = cfg["act_quant"]
    act_enabled = act_cfg.get("enabled", False)

    def _process(parent, attr_name, sublayer_path):
        module = getattr(parent, attr_name, None)
        if module is None:
            return
        layer_key = f"layer{layer_idx}.{sublayer_path}"
        if method.needs_calib_acts:
            acts      = calib_acts_for(layer_idx, sublayer_path, activations)
            col_scales = method.wrap(module, cfg, calib_acts=acts, layer_key=layer_key)
        else:
            col_scales = method.wrap(module, cfg, layer_key=layer_key)
        if act_enabled:
            is_qkv = sublayer_path in (
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            )
            install_perm_linear(
                parent, attr_name,
                act_fmt=act_cfg["fmt"],
                col_scales=col_scales,
                quantize_bmm_input=(is_qkv and act_cfg.get("quantize_bmm", False)),
            )

    with torch.no_grad():
        for layer_idx, block in enumerate(model.model.layers):
            for name in _ATTN_NAMES:
                _process(block.self_attn, name, f"self_attn.{name}")
            for name in _MLP_NAMES:
                _process(block.mlp, name, f"mlp.{name}")
            print(f"layer {layer_idx} quantised | method={method.name} "
                  f"weight_fmt={cfg['weight_fmt'].name} "
                  f"act_fmt={(act_cfg['fmt'].name if act_enabled else 'off')}")

        if hasattr(method, "save_scales"):
            num_iter     = cfg["method_cfg"].get("num_iter", 15)
            scale_option = cfg["method_cfg"]["scale_option"]
            row_init     = cfg["method_cfg"]["row_init"]
            col_init     = cfg["method_cfg"]["col_init"]
            saved = method.save_scales(
                args.model, num_iter,
                os.path.join(SCRIPT_DIR, "scales_dualquant_rtn_int4"),
                scale_option=scale_option, row_init=row_init, col_init=col_init,
            )
            if saved:
                print(f"scales saved to {saved}")

        if args.preprocess == "quarot":
            from quarot import install_online_hadamards
            if not act_enabled:
                from activations import PermLinear
                dummy_fmt = cfg["weight_fmt"]
                for block in model.model.layers:
                    for parent, names in ((block.self_attn, _ATTN_NAMES),
                                          (block.mlp, _MLP_NAMES)):
                        for n in names:
                            if getattr(parent, n, None) is not None:
                                install_perm_linear(parent, n, act_fmt=dummy_fmt,
                                                    col_scales=None,
                                                    quantize_bmm_input=False)
                for m in model.modules():
                    if isinstance(m, PermLinear):
                        m.set_act_quant_enabled(False)
            install_online_hadamards(model, verbose=True)


def evaluate_downstream(model, tokenizer, args):
    """Run lm_eval on the quantised model and return a dict of results."""
    print(f"\nRunning downstream eval: tasks={args.tasks}  nshots={args.nshots}")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        add_bos_token=False,
    )
    raw = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.nshots,
        log_samples=False,
    )
    # Flatten: {task: {metric: value}}
    out = {}
    for task, metrics in raw["results"].items():
        row = {}
        for k, v in metrics.items():
            if k == "alias":
                continue
            # keys look like "acc,none", "acc_norm,none" — keep the prefix
            short_key = k.split(",")[0]
            if isinstance(v, float):
                row[short_key] = round(v, 4)
        out[task] = row
        for k, v in row.items():
            print(f"  {task:<30} {k:<12} {v:.4f}")
    return out


def write_results_csv(args, results):
    """One row per (run × task × metric)."""
    if not args.results_path:
        return
    csv_path = args.results_path if args.results_path.endswith(".csv") \
               else args.results_path + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    act_fmt_label = args.act_fmt if args.act_quant == "on" else "off"
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "model", "method", "preprocess", "weight_fmt", "act_fmt",
            "task", "nshots", "metric", "value",
        ])
        if write_header:
            writer.writeheader()
        for task, metrics in results.items():
            for metric, value in metrics.items():
                writer.writerow({
                    "model":      args.model,
                    "method":     args.method,
                    "preprocess": args.preprocess,
                    "weight_fmt": args.weight_fmt,
                    "act_fmt":    act_fmt_label,
                    "task":       task,
                    "nshots":     args.nshots,
                    "metric":     metric,
                    "value":      value,
                })
    print(f"appended rows to {csv_path}")


def main():
    args           = parse_args()
    all_hyperparams = _load_hyperparams(args.method_cfg)
    cfg            = build_cfg(args, all_hyperparams)
    model          = load_model(args.model)
    tokenizer      = transformers.AutoTokenizer.from_pretrained(
                         args.model, use_fast=False)

    if args.preprocess != "none":
        preprocess_cfg = dict(all_hyperparams.get(args.preprocess, {}))
        preprocess_cfg.setdefault("_act_quant_enabled", args.act_quant == "on")
        if args.preprocess == "smoothquant" and args.sq_scales is not None:
            preprocess_cfg["act_scales_path"] = args.sq_scales
        print(f"applying preprocess={args.preprocess}")
        apply_preprocess(args.preprocess, model, preprocess_cfg)

    quantise_model(model, tokenizer, args, cfg)

    results = evaluate_downstream(model, tokenizer, args)
    if args.results_path:
        write_results_csv(args, results)


if __name__ == "__main__":
    main()
