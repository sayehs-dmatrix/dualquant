"""Single entrypoint: load model, quantise, evaluate PPL.

Orthogonal axes (model, weight format, activation format, preprocess) are
CLI flags. Method-specific hyperparameters come from a tiny per-method JSON
(--method-cfg path).

Usage:
    python main.py \\
        --model Qwen/Qwen3-0.6B \\
        --method dualquant --method-cfg configs/dualquant.json \\
        --weight-fmt mxfp4 --weight-block-size 32 --weight-scale-format e8m0 \\
        --tasks wikitext --nsamples 128 --seqlen 2048 \\
        --results-path results/dualquant_mxfp4.csv

    python main.py \\
        --model Qwen/Qwen3-0.6B \\
        --method rtn \\
        --weight-fmt rtn_int4 --weight-block-size 128 --weight-scale-format none \\
        --tasks wikitext

Activation quantisation (Stage 2):
    --act-quant on --act-fmt mxfp8_e4m3 --act-block-size 32 --act-scale-format e8m0
    [--act-scaled-before-quant] [--act-quantize-bmm]

Preprocess (Stage 2):
    --preprocess smoothquant --preprocess-cfg configs/smoothquant.json
"""

import _legacy_path  # noqa: F401  - lets us reuse legacy eval_utils, datautils

import argparse
import csv
import json
import os

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import datautils         # legacy
import eval_utils        # legacy
from activations import install_perm_linear
from data import calib_acts_for, collect_calib_activations
from formats import make_format
from methods import make_method
from preprocess import apply_preprocess


_ATTN_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_NAMES = ("up_proj", "down_proj", "gate_proj")

SCRIPT_DIR_OR_RESULTS = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()

    # ---- model + eval ----
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--tasks", nargs="+", default=["wikitext"], choices=["wikitext", "c4"])
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--results-path", default=None, help="CSV to append a row to")

    # ---- which sub-layers to quantise (default: both on) ----
    # BooleanOptionalAction auto-generates --no-attn-layers / --no-mlp-layers
    # for disabling, so the positive flags ("do this thing") match their meaning.
    p.add_argument("--attn-layers", action=argparse.BooleanOptionalAction, default=True,
                   help="quantise self-attention proj layers (default: enabled)")
    p.add_argument("--mlp-layers", action=argparse.BooleanOptionalAction, default=True,
                   help="quantise MLP proj layers (default: enabled)")

    # ---- method ----
    p.add_argument("--method", required=True, help="method name: rtn | dualquant | gptq | awq | sinq")
    p.add_argument("--method-cfg", default=None,
                   help="path to a single JSON with hyperparams for every method+preprocess "
                        "(top-level keys are method/preprocess names). Optional if no method/preprocess "
                        "in this run has hyperparameters.")

    # ---- weight format ----
    p.add_argument("--weight-fmt", required=True,
                   help="mxfp4 | mxint4 | mxfp8_e4m3 | mxfp8_e5m2 | mxint8 | nvfp4 | sfp4 | rtn_int4 | rtn_int8")
    p.add_argument("--weight-block-size", type=int, default=32)
    p.add_argument("--weight-scale-format", default="e8m0",
                   choices=["e8m0", "e4m3", "e4m4", "e5m3", "none"])
    p.add_argument("--weight-clip", action="store_true",
                   help="MSE-optimal weight clipping for RTN (80-step grid search, "
                        "equivalent to QuaRot --w_clip). Only affects rtn_int4/rtn_int8.")

    # ---- activation quantisation (orthogonal to weight format) ----
    p.add_argument("--act-quant", choices=["on", "off"], default="off")
    p.add_argument("--act-fmt", default=None, help="independent of --weight-fmt")
    p.add_argument("--act-block-size", type=int, default=32)
    p.add_argument("--act-scale-format", default="e8m0", choices=["e8m0", "e4m3", "e4m4", "e5m3", "none"])
    p.add_argument("--act-scaled-before-quant", action="store_true")
    p.add_argument("--act-quantize-bmm", action="store_true")

    # ---- preprocess ----
    p.add_argument("--preprocess", default="none", choices=["none", "smoothquant", "quarot"])
    # preprocess hyperparams come from the same --method-cfg JSON, under the preprocess name.
    # SmoothQuant per-model scales path. Overrides JSON's "act_scales_path" if set;
    # the bash sweep uses this to route each model to its matching .pt under
    # smoothquant/act_scales/. Ignored unless --preprocess smoothquant.
    p.add_argument("--sq-scales", default=None,
                   help="Path to SmoothQuant act_scales .pt (overrides JSON value).")

    # ---- diagnostics ----
    # If set, run PPL on the model AFTER preprocess but BEFORE quantisation. Used
    # to verify that the preprocess (e.g. QuaRot R1) is FP-lossless. Compare to
    # the no-preprocess fp16/bf16 baseline; if they differ, the rotation pipeline
    # has a numerical or correctness bug; if equal, any subsequent PPL hit is
    # purely from quantisation interacting with rotated weights.
    p.add_argument("--debug-post-preprocess-ppl", action="store_true",
                   help="Eval PPL right after preprocess (skips quantisation + writes).")

    return p.parse_args()


def _load_all_hyperparams(path):
    if path is None:
        return {}
    with open(path) as fh:
        return json.load(fh)


def build_cfg(args, all_hyperparams):
    """Assemble the cfg dict that methods.wrap consumes."""
    method_cfg = all_hyperparams.get(args.method, {})

    weight_fmt = make_format(args.weight_fmt, block_size=args.weight_block_size,
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
        "method_cfg": method_cfg,
        "weight_fmt": weight_fmt,
        "weight_scale_format": args.weight_scale_format,
        "act_quant": act_quant,
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

    # Methods that own their own orchestration (e.g. sequential GPTQ) expose
    # a quantise_model() and handle weight quant + PermLinear install + any
    # post-processing themselves. We dispatch and return early.
    if hasattr(method, "quantise_model"):
        method.quantise_model(model, tokenizer, args, cfg)
        return

    activations = None
    if method.needs_calib_acts:
        print(f"collecting calibration activations for {method.name}...")
        activations = collect_calib_activations(model, tokenizer)
        print(f"collected activations for {len(activations)} layers")
        print(f"  sample keys: {list(activations.keys())[:4]}")
        # sanity-check: verify the key pattern matches what calib_acts_for expects
        expected = "model.layers.0.self_attn.q_proj"
        print(f"  key '{expected}' present: {expected in activations}")
        if expected in activations:
            a = activations[expected]
            print(f"  activation shape={a.shape}  mean={a.float().mean():.4f}  std={a.float().std():.4f}  allzero={a.eq(0).all().item()}")

    act_cfg = cfg["act_quant"]
    act_enabled = act_cfg.get("enabled", False)

    def _process(parent, attr_name, sublayer_path):
        module = getattr(parent, attr_name, None)
        if module is None:
            return
        layer_key = f"layer{layer_idx}.{sublayer_path}"

        if method.needs_calib_acts:
            acts = calib_acts_for(layer_idx, sublayer_path, activations)
            col_scales = method.wrap(module, cfg, calib_acts=acts, layer_key=layer_key)
        else:
            col_scales = method.wrap(module, cfg, layer_key=layer_key)
        if act_enabled:
            # For attention q/k/v we also quantise the output (it feeds the BMM);
            # o_proj output goes back to the residual stream so we don't.
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
            if args.attn_layers:
                for name in _ATTN_NAMES:
                    _process(block.self_attn, name, f"self_attn.{name}")
            if args.mlp_layers:
                for name in _MLP_NAMES:
                    _process(block.mlp, name, f"mlp.{name}")
            print(f"layer {layer_idx} quantised | method={method.name} "
                  f"weight_fmt={cfg['weight_fmt'].name} "
                  f"act_fmt={(act_cfg['fmt'].name if act_enabled else 'off')}")
        if hasattr(method, "save_scales"):
            num_iter = cfg["method_cfg"].get("num_iter", 15)
            scale_option = cfg["method_cfg"]["scale_option"]
            row_init = cfg["method_cfg"]["row_init"]
            col_init = cfg["method_cfg"]["col_init"]
            saved = method.save_scales(args.model, num_iter, os.path.join(SCRIPT_DIR_OR_RESULTS, "scales_dualquant_rtn_int4"), scale_option=scale_option, row_init=row_init, col_init=col_init)
            if saved: print(f"scales saved to {saved}")

        # QuaRot R2/R4 — install the online Hadamards on o_proj / down_proj PermLinear
        # wrappers, the second half of the offline+online identity. The reference
        # paper relies on the H @ H = I cancellation; without this call, the offline
        # R2/R4 folds drift the network's output.
        #
        # act_quant=on:  PermLinears already installed above (with real act_fmt,
        #                act_quant_enabled=True). Just set the online-had flags.
        # act_quant=off: No PermLinears exist yet. Install pass-through ones
        #                (act_quant_enabled=False) so the online Hadamard can
        #                fire and cancel the offline R2/R4 weight folds while
        #                leaving activations unquantised.
        if args.preprocess == "quarot":
            from quarot import install_online_hadamards
            if not act_enabled:
                from activations import PermLinear  # install_perm_linear already at module level
                dummy_fmt = cfg["weight_fmt"]  # act_quant_enabled=False → never called
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


def evaluate_ppl(model, tokenizer, args):
    dev = torch.device("cuda:0")
    out = {}
    if "wikitext" in args.tasks:
        _, _, _, testenc, _ = datautils.get_loaders(
            "wikitext", args.model, nsamples=args.nsamples, seed=0, seqlen=args.seqlen
        )
        out["wikitext"] = float(eval_utils.evaluator_newVersion(model, tokenizer, testenc, dev, args, dataset="test"))
        print(f"WikiText-2 PPL: {out['wikitext']:.4f}")
    if "c4" in args.tasks:
        try:
            c4enc = datautils.get_c4_testenc(args.model, args.seqlen)
            out["c4"] = float(eval_utils.evaluator_newVersion(model, tokenizer, c4enc, dev, args, dataset="test"))
            print(f"C4 PPL: {out['c4']:.4f}")
        except Exception as exc:
            print(f"C4 evaluation failed: {exc}")
    return out


def write_results_csv(args, ppl):
    csv_path = args.results_path if args.results_path.endswith(".csv") else args.results_path + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "method", "preprocess", "weight_fmt", "act_fmt",
                        "ppl_wikitext", "ppl_c4"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model": args.model,
            "method": args.method,
            "preprocess": args.preprocess,
            "weight_fmt": args.weight_fmt,
            "act_fmt": args.act_fmt if args.act_quant == "on" else "off",
            "ppl_wikitext": round(ppl["wikitext"], 4) if "wikitext" in ppl else "N/A",
            "ppl_c4": round(ppl["c4"], 4) if "c4" in ppl else "N/A",
        })
    print(f"appended row to {csv_path}")


def main():
    args = parse_args()
    all_hyperparams = _load_all_hyperparams(args.method_cfg)
    cfg = build_cfg(args, all_hyperparams)
    model = load_model(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if args.preprocess != "none":
        preprocess_cfg = dict(all_hyperparams.get(args.preprocess, {}))
        # Auto-signal to QuaRot whether activation quant is on so it knows
        # whether to apply R2/R4 (offline-only halves drift the model without
        # the online halves installed in PermLinear).
        preprocess_cfg.setdefault("_act_quant_enabled", args.act_quant == "on")
        # SmoothQuant per-model scales path: --sq-scales wins over JSON if both set.
        if args.preprocess == "smoothquant" and args.sq_scales is not None:
            preprocess_cfg["act_scales_path"] = args.sq_scales
        print(f"applying preprocess={args.preprocess}")
        apply_preprocess(args.preprocess, model, preprocess_cfg)

    if args.debug_post_preprocess_ppl:
        # For QuaRot + act_quant=on, R2/R4 offline halves are uncancelled until
        # PermLinear's online halves fire at forward time. Install the wrappers
        # with act_quant_enabled=False so this eval measures rotation cancellation
        # in isolation (no quant noise). Without this, the PPL is garbage.
        if args.preprocess == "quarot" and args.act_quant == "on":
            from activations import PermLinear
            from quarot import install_online_hadamards
            act_fmt = cfg["act_quant"]["fmt"]
            for block in model.model.layers:
                for parent, names in ((block.self_attn, _ATTN_NAMES),
                                      (block.mlp, _MLP_NAMES)):
                    for n in names:
                        if getattr(parent, n, None) is not None:
                            install_perm_linear(parent, n, act_fmt=act_fmt,
                                                col_scales=None,
                                                quantize_bmm_input=False)
            for m in model.modules():
                if isinstance(m, PermLinear):
                    m.set_act_quant_enabled(False)
            install_online_hadamards(model, verbose=True)
        print(f"[debug] post-preprocess PPL (no quantisation):")
        ppl = evaluate_ppl(model, tokenizer, args)
        print(f"[debug] result: {ppl}")
        return

    quantise_model(model, tokenizer, args, cfg)
    ppl = evaluate_ppl(model, tokenizer, args)
    if args.results_path:
        write_results_csv(args, ppl)


if __name__ == "__main__":
    main()
