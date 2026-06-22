"""Per-layer output SQNR — FP vs DualQuant vs SmoothQuant.

For each transformer decoder layer, runs the same calibration sequences
through three versions of the model (FP, DQ-quantised, SQ-quantised),
captures the layer's residual-stream output via a forward hook, and reports
the SQNR (in dB) of the quantised output relative to the FP reference.

Higher SQNR = closer to FP. The per-layer view shows which layers each
method handles well and which it doesn't — complements the per-channel
scale-correlation tables in the other Rebutal_Neurips scripts.

Standalone — uses the main codebase's modules (via the same imports main.py
uses) but does not modify anything in the codebase.

Defaults:
    --act-quant off    → compares the *weight-quantisation* effect of each
                         method in isolation. DualQuant's column scales
                         (1/β) are folded into the stored weight at
                         quantisation time, so the conventional forward
                         pass produces the full DQ result without any
                         runtime activation step. SmoothQuant's
                         per-channel migration is likewise pre-applied
                         to weights (and LayerNorm gammas). No activation
                         quant noise on top.

    Pass --act-quant on if you want the full W4A4 picture (DQ then has 1/β
    applied at runtime via PermLinear, both DQ and SQ also get activation
    quantisation noise).

Usage:
    python Rebutal_Neurips/sqnr_per_layer.py --model meta-llama/Llama-3.2-1B
    python Rebutal_Neurips/sqnr_per_layer.py --model Qwen/Qwen3-0.6B --nsamples 2
    python Rebutal_Neurips/sqnr_per_layer.py --model meta-llama/Llama-3.2-1B \\
        --weight-fmt mxint4 --weight-block-size 32 --weight-scale-format e8m0 \\
        --act-fmt mxint4 --act-quant on
"""
import argparse
import json
import os
import sys

print("[import-debug] entering script", flush=True)
import torch
print("[import-debug] torch OK", flush=True)

# Make the parent codebase importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _CODEBASE_ROOT not in sys.path:
    sys.path.insert(0, _CODEBASE_ROOT)

import _legacy_path  # noqa: F401
print("[import-debug] _legacy_path OK", flush=True)

import datautils  # legacy
print("[import-debug] datautils OK", flush=True)
from transformers import AutoTokenizer
print("[import-debug] transformers.AutoTokenizer OK", flush=True)

print("[import-debug] trying eval_utils", flush=True)
import eval_utils  # noqa: F401
print("[import-debug] eval_utils OK", flush=True)

print("[import-debug] trying formats.base", flush=True)
from formats.base import FormatSpec  # noqa: F401
print("[import-debug] formats.base OK", flush=True)

print("[import-debug] trying formats.mx (imports dmx.compressor)", flush=True)
from formats import mx  # noqa: F401
print("[import-debug] formats.mx OK", flush=True)

print("[import-debug] trying formats.nvfp4", flush=True)
from formats import nvfp4  # noqa: F401
print("[import-debug] formats.nvfp4 OK", flush=True)

print("[import-debug] trying formats.rtn_int", flush=True)
from formats import rtn_int  # noqa: F401
print("[import-debug] formats.rtn_int OK", flush=True)

print("[import-debug] trying formats.sfp4", flush=True)
from formats import sfp4  # noqa: F401
print("[import-debug] formats.sfp4 OK", flush=True)

print("[import-debug] trying formats.make_format", flush=True)
from formats import make_format  # noqa: F401
print("[import-debug] formats.make_format OK", flush=True)

print("[import-debug] trying activations.install_perm_linear", flush=True)
from activations import install_perm_linear  # noqa: F401
print("[import-debug] activations.install_perm_linear OK", flush=True)

print("[import-debug] trying data", flush=True)
from data import calib_acts_for, collect_calib_activations  # noqa: F401
print("[import-debug] data OK", flush=True)

print("[import-debug] trying methods.make_method", flush=True)
from methods import make_method  # noqa: F401
print("[import-debug] methods.make_method OK", flush=True)

print("[import-debug] trying preprocess.apply_preprocess", flush=True)
from preprocess import apply_preprocess
print("[import-debug] preprocess.apply_preprocess OK", flush=True)

print("[import-debug] trying main.*", flush=True)
from main import quantise_model, build_cfg, load_model
print("[import-debug] main.* OK", flush=True)


# Mirror of run_sweep.sh:sq_scales_for() — model id → SmoothQuant .pt filename.
SQ_SCALES = {
    "meta-llama/Llama-3.1-8B":   "Llama-3.1-8B_seq_len_8192_4bit.pt",
    "meta-llama/Llama-3.2-1B":   "Llama-3.2-1B_seq_len_8192_4bit.pt",
    "meta-llama/Llama-2-7b-hf":  "Meta-Llama-2-7B_seqlen_4096_4bit.pt",
    "meta-llama/Llama-2-13b-hf": "Meta-Llama-2-13B_seqlen_4096_4bit.pt",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B-v01_seq_len_8192_4bit.pt",
    "Qwen/Qwen2-1.5B":           "Qwen2-1.5B_seq_len_8192_4bit.pt",
    "Qwen/Qwen2.5-0.5B":         "Qwen2.5-0.5B_seq_len_8192_4bit.pt",
    "Qwen/Qwen2.5-7B":           "Qwen2.5-7B_seq_len_8192_4bit.pt",
    "Qwen/Qwen2.5-14B":          "Qwen2.5-14B_seq_len_8192_4bit.pt",
    "Qwen/Qwen3-0.6B":           "Qwen3-0.6B_seq_len_8192_4bit.pt",
    "Qwen/Qwen3-1.7B":           "Qwen3-1.7B_seq_len_8192_4bit.pt",
    "Qwen/Qwen3-14B":            "Qwen3-14B_seq_len_8192_4bit.pt",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B",
                   choices=sorted(SQ_SCALES.keys()),
                   help="Which model to evaluate")
    p.add_argument("--nsamples", type=int, default=4,
                   help="Calibration sequences to average SQNR over (keep low for big models)")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--weight-fmt", default="rtn_int4")
    p.add_argument("--weight-block-size", type=int, default=64)
    p.add_argument("--weight-scale-format", default="none",
                   choices=["e8m0", "e4m3", "e4m4", "e5m3", "none"])
    p.add_argument("--act-fmt", default=None, help="Defaults to --weight-fmt")
    p.add_argument("--act-block-size", type=int, default=None,
                   help="Defaults to --weight-block-size")
    p.add_argument("--act-scale-format", default=None,
                   choices=["e8m0", "e4m3", "e4m4", "e5m3", "none"],
                   help="Defaults to --weight-scale-format")
    p.add_argument("--act-quant", default="off", choices=["on", "off"],
                   help="off (default): weight-only comparison. on: full W4A4-style.")
    p.add_argument("--act-scaled-before-quant", action="store_true",
                   help="Apply column scales to activations before quantising (matches run_sweep.sh).")
    p.add_argument("--act-quantize-bmm", action="store_true",
                   help="Also quantise BMM inputs (matches --act-quantize-bmm in run_sweep.sh).")
    p.add_argument("--attn-layers", action=argparse.BooleanOptionalAction, default=True,
                   help="Quantise attention projection layers (default: on).")
    p.add_argument("--mlp-layers", action=argparse.BooleanOptionalAction, default=True,
                   help="Quantise MLP projection layers (default: on).")
    p.add_argument("--method-cfg",
                   default=os.path.join(_CODEBASE_ROOT, "configs/methods.json"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--csv-out", default=None,
                   help="Optional CSV path to also write the per-layer table to")
    p.add_argument("--pdf-out", default=None,
                   help="Optional PDF path for result plots. Defaults to <csv-out>.pdf if --csv-out is set.")
    p.add_argument("--isolate-layer", type=int, default=None,
                   help="If set, quantise ONLY this block index; all others are restored to FP. "
                        "Removes accumulated error so block SQNR reflects only this layer.")
    p.add_argument("--data-split", default="test",
                   choices=["test", "test_indiv", "sq_calib"],
                   help="test (default): WikiText-2 test split, chunked into seqlen tokens. "
                        "test_indiv: WikiText-2 test split, individual text tokenisation "
                        "(same approach as sq_calib — use to compare splits fairly). "
                        "sq_calib: WikiText-2 train split, shuffle(seed=42), individual "
                        "tokenisation — exact same texts as SmoothQuant calibration.")
    cli = p.parse_args()
    if cli.act_fmt is None:
        cli.act_fmt = cli.weight_fmt
    if cli.act_block_size is None:
        cli.act_block_size = cli.weight_block_size
    if cli.act_scale_format is None:
        cli.act_scale_format = cli.weight_scale_format
    return cli


def make_main_args(method, preprocess, cli):
    """Construct a Namespace that mimics main.py's parse_args output."""
    return argparse.Namespace(
        model=cli.model,
        method=method,
        method_cfg=cli.method_cfg,
        weight_fmt=cli.weight_fmt,
        weight_block_size=cli.weight_block_size,
        weight_scale_format=cli.weight_scale_format,
        act_quant=cli.act_quant,
        act_fmt=cli.act_fmt,
        act_block_size=cli.act_block_size,
        act_scale_format=cli.act_scale_format,
        act_scaled_before_quant=cli.act_scaled_before_quant,
        act_quantize_bmm=cli.act_quantize_bmm,
        attn_layers=cli.attn_layers,
        mlp_layers=cli.mlp_layers,
        nsamples=cli.nsamples,
        seqlen=cli.seqlen,
        tasks=["wikitext"],
        results_path=None,
        sq_scales=None,
        preprocess=preprocess,
        debug_post_preprocess_ppl=False,
        weight_clip=False,
    )


# Ordered list of (short_name, parent_attr, proj_attr) for all projections.
PROJ_PATHS = [
    ("self_attn.q_proj", "self_attn", "q_proj"),
    ("self_attn.k_proj", "self_attn", "k_proj"),
    ("self_attn.v_proj", "self_attn", "v_proj"),
    ("self_attn.o_proj", "self_attn", "o_proj"),
    ("mlp.gate_proj",    "mlp",       "gate_proj"),
    ("mlp.up_proj",      "mlp",       "up_proj"),
    ("mlp.down_proj",    "mlp",       "down_proj"),
]


class _Capturer:
    """Attach forward (post) and forward-pre hooks; store outputs by key.

    hooks format: list of (kind, module, key)
      kind='post' — register_forward_hook,      captures module OUTPUT
      kind='pre'  — register_forward_pre_hook,  captures module INPUT[0]
    """
    def __init__(self, hooks):
        self.outputs = {}
        self.handles = []
        for kind, m, k in hooks:
            if kind == 'post':
                self.handles.append(m.register_forward_hook(self._post_hook(k)))
            else:
                self.handles.append(m.register_forward_pre_hook(self._pre_hook(k)))

    def _post_hook(self, key):
        def fn(_mod, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            self.outputs[key] = x.detach().to("cpu", dtype=torch.float32)
        return fn

    def _pre_hook(self, key):
        def fn(_mod, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            self.outputs[key] = x.detach().to("cpu", dtype=torch.float32)
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def _build_hooks(model):
    """Return [(kind, module, key)] for decoder blocks, sub-block residuals, and projections.

    Keys:
      i                        — full block output:        xᵢ + aᵢ + mᵢ
      (i, 'after_attn')        — after attention residual: xᵢ + aᵢ  (pre-hook on post_attn_ln)
      (i, proj_name)           — individual projection output
    """
    hooks = []
    for i, block in enumerate(model.model.layers):
        # Full block output (post-hook on decoder layer)
        hooks.append(('post', block, i))

        # After-attention residual = input to post_attention_layernorm
        post_ln = getattr(block, 'post_attention_layernorm', None)
        if post_ln is not None:
            hooks.append(('pre', post_ln, (i, 'after_attn')))

        # Individual projections
        for proj_name, parent_attr, attr in PROJ_PATHS:
            parent = getattr(block, parent_attr, None)
            if parent is None:
                continue
            module = getattr(parent, attr, None)
            if module is None:
                continue
            hooks.append(('post', module, (i, proj_name)))

    # Final logits: lm_head output after the final layer norm.
    # Shape (batch, seq, vocab_size) — captured as scalar sums only (never stored).
    # This directly predicts PPL.
    if hasattr(model, 'lm_head'):
        hooks.append(('post', model.lm_head, 'logits'))

    return hooks


@torch.no_grad()
def measure_sqnr_online(fp_model, dq_model, sq_model, calib_ids, device):
    """Compute per-block and per-projection SQNR without storing activation tensors.

    All three models are kept on-device simultaneously. For each calibration
    sample: run FP → move outputs to CPU → run DQ → compare → run SQ → compare →
    accumulate scalar (signal / noise) sums → discard tensors immediately.

    Returns:
        block_sqnr: {layer_idx:  (dq_db, sq_db)}
        proj_sqnr:  {(layer_idx, proj_name): (dq_db, sq_db)}
    """
    for m in (fp_model, dq_model, sq_model):
        m.to(device).eval()

    fp_cap = _Capturer(_build_hooks(fp_model))
    dq_cap = _Capturer(_build_hooks(dq_model))
    sq_cap = _Capturer(_build_hooks(sq_model))

    # acc[key] = {'dq': [per-sample SQNR], 'sq': [per-sample SQNR]}
    acc = {}
    eps = 1e-12

    def _db(sig, noise):
        import math
        return 10.0 * math.log10(max(sig, eps) / max(noise, eps))

    for i, ids in enumerate(calib_ids):
        ids_gpu = ids.to(device)

        # ── FP reference forward ──
        fp_cap.outputs.clear()
        fp_model(ids_gpu)
        fp_cpu = {k: v for k, v in fp_cap.outputs.items()}

        # ── DQ noise — compute per-sample SQNR immediately ──
        dq_cap.outputs.clear()
        dq_model(ids_gpu)
        for k, fp_v in fp_cpu.items():
            sig      = float((fp_v ** 2).sum())
            noise_dq = float(((fp_v - dq_cap.outputs[k]) ** 2).sum())
            acc.setdefault(k, {'dq': [], 'sq': []})['dq'].append(_db(sig, noise_dq))

        # ── SQ noise — compute per-sample SQNR immediately ──
        sq_cap.outputs.clear()
        sq_model(ids_gpu)
        for k, fp_v in fp_cpu.items():
            sig      = float((fp_v ** 2).sum())
            noise_sq = float(((fp_v - sq_cap.outputs[k]) ** 2).sum())
            acc[k]['sq'].append(_db(sig, noise_sq))

        print(f"  sample {i + 1}/{len(calib_ids)} done", flush=True)

    for cap in (fp_cap, dq_cap, sq_cap):
        cap.remove()

    def _stats(vals):
        import math, statistics as _st
        if not vals:
            return float('nan'), float('nan')
        mean = sum(vals) / len(vals)
        std  = _st.stdev(vals) if len(vals) > 1 else float('nan')
        return mean, std

    # Return format: (dq_mean, dq_std, sq_mean, sq_std)
    block_sqnr, proj_sqnr, logit_sqnr = {}, {}, {}
    for k, entry in acc.items():
        dq_mean, dq_std = _stats(entry['dq'])
        sq_mean, sq_std = _stats(entry['sq'])
        result = (dq_mean, dq_std, sq_mean, sq_std)
        if k == 'logits':
            logit_sqnr = {'dq': dq_mean, 'dq_std': dq_std,
                          'sq': sq_mean, 'sq_std': sq_std}
        elif isinstance(k, int):
            block_sqnr[k] = result
        else:
            proj_sqnr[k] = result

    return block_sqnr, proj_sqnr, logit_sqnr



def _isolate_to_layer(quant_model, fp_model, k):
    """Restore all decoder blocks except k to their original FP weights.

    After quantise_model has run on all layers, this resets every block
    except k back to FP, so that only block k introduces quantisation error.
    The input to block k (Δxₖ = 0) is therefore the clean FP hidden state.

    Also handles act_quant=on: PermLinear wrappers on non-k blocks have their
    activation quantization disabled and act_scale (column scale buffer) cleared,
    so they behave identically to the original FP Linear.
    """
    from activations import PermLinear
    for i, (q_block, fp_block) in enumerate(
        zip(quant_model.model.layers, fp_model.model.layers)
    ):
        if i == k:
            continue
        # Copy FP weights (parameters only — buffers like act_scale are separate)
        for q_p, fp_p in zip(q_block.parameters(), fp_block.parameters()):
            q_p.data.copy_(fp_p.data)
        # If PermLinear was installed (act_quant=on), disable activation quant
        # and clear the column-scale buffer so the restored block is pure FP.
        for m in q_block.modules():
            if isinstance(m, PermLinear):
                m.set_act_scale(None)
                m.set_act_quant_enabled(False)


def main():

    cli = parse_args()
    print(f"Model:       {cli.model}")
    print(f"Weights:     {cli.weight_fmt} (block={cli.weight_block_size}, scale={cli.weight_scale_format})")
    print(f"Activations: {cli.act_fmt} (block={cli.act_block_size}, scale={cli.act_scale_format}, quant={cli.act_quant})")
    print(f"Calib:       {cli.nsamples} × {cli.seqlen} tokens  "
          f"(wikitext/{cli.data_split})")

    tokenizer = AutoTokenizer.from_pretrained(cli.model, use_fast=False)

    if cli.data_split == "sq_calib":
        # Exact same texts as SmoothQuant calibration:
        # WikiText-2 train → shuffle(seed=42) → first nsamples non-empty →
        # tokenise individually with max_length=seqlen, truncation=True.
        # Replicates smoothquant_calib_llama3.py + calibration.py data loading.
        from datasets import load_dataset as _hf_load
        _ds = _hf_load("wikitext", "wikitext-2-raw-v1", split="train")
        _ds = _ds.shuffle(seed=42)
        calib_ids = []
        for sample in _ds:
            if len(calib_ids) >= cli.nsamples:
                break
            text = sample["text"]
            if not text or not text.strip():
                continue
            ids = tokenizer(
                text, return_tensors="pt",
                max_length=cli.seqlen, truncation=True
            ).input_ids
            if ids.shape[1] == 0:
                continue
            calib_ids.append(ids)
        n_use = len(calib_ids)
        print(f"  [sq_calib] loaded {n_use} samples from WikiText-2 train "
              f"(shuffle seed=42) — exactly matches SQ calibration data", flush=True)
    elif cli.data_split == "test_indiv":
        # WikiText-2 test split, individual text tokenisation.
        # Same tokenisation as sq_calib — only the split differs.
        # Use this to compare train vs test without tokenisation confound.
        from datasets import load_dataset as _hf_load
        _ds = _hf_load("wikitext", "wikitext-2-raw-v1", split="test")
        calib_ids = []
        for sample in _ds:
            if len(calib_ids) >= cli.nsamples:
                break
            text = sample["text"]
            if not text or not text.strip():
                continue
            ids = tokenizer(
                text, return_tensors="pt",
                max_length=cli.seqlen, truncation=True
            ).input_ids
            if ids.shape[1] == 0:
                continue
            calib_ids.append(ids)
        n_use = len(calib_ids)
        print(f"  [test_indiv] loaded {n_use} samples from WikiText-2 test "
              f"(individual text tokenisation — comparable to sq_calib)", flush=True)
    else:
        # test (default): chunked tokenisation, WikiText-2 test split.
        _, _, _, testenc, _ = datautils.get_loaders(
            "wikitext", cli.model, nsamples=cli.nsamples, seed=0, seqlen=cli.seqlen,
        )
        T = testenc.input_ids.shape[1]
        n_use = min(cli.nsamples, T // cli.seqlen)
        import random as _random
        _random.seed(0)
        starts = _random.sample(range(T - cli.seqlen), n_use)
        calib_ids = [
            testenc.input_ids[:, s:s + cli.seqlen]
            for s in starts
        ]

    with open(cli.method_cfg) as f:
        all_hp = json.load(f)

    # ── Load and quantise all three models ────────────────────────────────────
    print(f"\n[1/3] Loading FP model...", flush=True)
    fp_model = load_model(cli.model)

    print(f"\n[2/3] Loading + quantising DualQuant model...", flush=True)
    dq_args = make_main_args(method="dualquant", preprocess="none", cli=cli)
    dq_cfg  = build_cfg(dq_args, all_hp)
    dq_model = load_model(cli.model)
    quantise_model(dq_model, tokenizer, dq_args, dq_cfg)
    if cli.isolate_layer is not None:
        _isolate_to_layer(dq_model, fp_model, cli.isolate_layer)
        print(f"  [DQ] isolated to layer {cli.isolate_layer} — all others restored to FP", flush=True)

    print(f"\n[3/3] Loading + quantising SmoothQuant+RTN model...", flush=True)
    sq_scales_path = os.path.join(
        _CODEBASE_ROOT, "smoothquant", "act_scales", SQ_SCALES[cli.model]
    )
    sq_args = make_main_args(method="rtn", preprocess="smoothquant", cli=cli)
    sq_args.sq_scales = sq_scales_path
    sq_cfg  = build_cfg(sq_args, all_hp)
    sq_model = load_model(cli.model)
    if cli.attn_layers or cli.mlp_layers:
        sq_pp_cfg = dict(all_hp.get("smoothquant", {}))
        sq_pp_cfg["act_scales_path"] = sq_scales_path
        apply_preprocess("smoothquant", sq_model, sq_pp_cfg)
        print(f"  [SQ] preprocess applied", flush=True)
    else:
        print(f"  [SQ] skipping preprocess (attn_layers=off, mlp_layers=off)", flush=True)
    quantise_model(sq_model, tokenizer, sq_args, sq_cfg)
    if cli.isolate_layer is not None:
        _isolate_to_layer(sq_model, fp_model, cli.isolate_layer)
        print(f"  [SQ] isolated to layer {cli.isolate_layer} — all others restored to FP", flush=True)
    # ── Online SQNR measurement — no large tensors retained ──────────────────
    print(f"\nRunning online SQNR measurement ({n_use} samples × {cli.seqlen} tokens)...", flush=True)
    block_sqnr, proj_sqnr, logit_sqnr = measure_sqnr_online(
        fp_model, dq_model, sq_model, calib_ids, cli.device
    )
    del fp_model, dq_model, sq_model
    torch.cuda.empty_cache()

    # ── Logit SQNR — most direct proxy for PPL ───────────────────────────────────
    print("\n" + "=" * 60)
    print(f"LOGIT SQNR (dB)  —  after lm_head, directly predicts PPL")
    print(f"Averaged over {n_use} sequences × {cli.seqlen} tokens")
    print("=" * 60)
    if logit_sqnr:
        dq_l = logit_sqnr['dq']
        sq_l = logit_sqnr['sq']
        dq_std_l = logit_sqnr.get('dq_std', float('nan'))
        sq_std_l = logit_sqnr.get('sq_std', float('nan'))
        print(f"  DualQuant  : {dq_l:>8.2f} ± {dq_std_l:.2f} dB")
        print(f"  SmoothQuant: {sq_l:>8.2f} ± {sq_std_l:.2f} dB")
        print(f"  DQ - SQ    : {dq_l - sq_l:>+8.2f} dB")
        winner = "DualQuant" if dq_l > sq_l else "SmoothQuant"
        print(f"  → {winner} has higher logit SQNR (closer to FP logits)")
    else:
        print("  [no logit hook found — lm_head missing?]")

    # ── Residual-stream SQNR table (after_attn and full block) ──────────────────
    n_layers = max(block_sqnr) + 1

    def _print_residual_table(title, row_data):
        # row_data: list of (layer, dq_mean, dq_std, sq_mean, sq_std)
        print("\n" + "=" * 76)
        print(f"{title}")
        print(f"Averaged over {n_use} sequences × {cli.seqlen} tokens  (mean ± std)")
        print("=" * 76)
        print(f"{'Layer':>5} | {'DualQuant mean±std':>20} | {'SmoothQuant mean±std':>22} | {'DQ-SQ':>7}")
        print("-" * 64)
        for l, d, d_std, s, s_std in row_data:
            dq_str = f"{d:.2f} ± {d_std:.2f}" if d_std == d_std else f"{d:.2f}"
            sq_str = f"{s:.2f} ± {s_std:.2f}" if s_std == s_std else f"{s:.2f}"
            print(f"{l:>5d} | {dq_str:>20} | {sq_str:>22} | {d-s:>+7.2f}")
        print("-" * 64)
        mean_dq = sum(r[1] for r in row_data) / len(row_data)
        mean_sq = sum(r[3] for r in row_data) / len(row_data)
        print(f"{'Mean':>5} | {mean_dq:>20.2f} | {mean_sq:>22.2f} | {mean_dq - mean_sq:>+7.2f}")

    # After-attention residual: xᵢ + aᵢ
    after_attn_rows = []
    for l in range(n_layers):
        key = (l, 'after_attn')
        if key in proj_sqnr:
            d, d_std, s, s_std = proj_sqnr[key]
            after_attn_rows.append((l, d, d_std, s, s_std))
    if after_attn_rows:
        _print_residual_table(
            "After-attention SQNR (dB)  —  xᵢ + aᵢ  (residual stream after attention, before MLP)",
            after_attn_rows,
        )

    # Full block output: xᵢ + aᵢ + mᵢ
    rows = [(l, *block_sqnr[l]) for l in range(n_layers)]
    _print_residual_table(
        "Full-block SQNR (dB)  —  xᵢ + aᵢ + mᵢ  (residual stream after attention + MLP)",
        rows,
    )

    # ── Per-projection SQNR table ──
    proj_names = [p[0] for p in PROJ_PATHS]
    proj_names = [p for p in proj_names if (0, p) in proj_sqnr and p != 'after_attn']

    print("\n" + "=" * 80)
    print(f"Per-projection SQNR (dB)  —  individual layer output, higher = closer to FP")
    print(f"{n_use} sequences × {cli.seqlen} tokens  (mean ± std)")
    print("=" * 80)
    col_w = 14
    header = f"{'Layer':>5}  {'Projection':<22}" + f"{'DQ mean±std':>{col_w}}" + f"{'SQ mean±std':>{col_w}}" + f"{'DQ-SQ':>8}"
    print(header)
    print("-" * len(header))
    proj_rows = []
    for l in range(n_layers):
        for proj in proj_names:
            key = (l, proj)
            if key not in proj_sqnr:
                continue
            d, d_std, s, s_std = proj_sqnr[key]
            proj_rows.append((l, proj, d, d_std, s, s_std))
            dq_str = f"{d:.1f}±{d_std:.1f}"
            sq_str = f"{s:.1f}±{s_std:.1f}"
            print(f"{l:>5d}  {proj:<22}{dq_str:>{col_w}}{sq_str:>{col_w}}{d-s:>+8.2f}")
        print()

    print("-" * len(header))
    for proj in proj_names:
        vals = [(d, d_std, s, s_std) for (l, p, d, d_std, s, s_std) in proj_rows if p == proj]
        if not vals:
            continue
        md     = sum(d      for d, _, s, __ in vals) / len(vals)
        ms     = sum(s      for _, __, s, ___ in vals) / len(vals)
        md_std = sum(d_std  for _, d_std, __, ___ in vals) / len(vals)
        ms_std = sum(s_std  for _, __, ___, s_std in vals) / len(vals)
        dq_str = f"{md:.1f}±{md_std:.1f}"
        sq_str = f"{ms:.1f}±{ms_std:.1f}"
        print(f"{'Mean':>5}  {proj:<22}{dq_str:>{col_w}}{sq_str:>{col_w}}{md-ms:>+8.2f}")

    if cli.csv_out:
        import csv
        base, ext = cli.csv_out.rsplit(".", 1) if "." in cli.csv_out else (cli.csv_out, "csv")
        # Block-level CSV (original)
        with open(cli.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "dq_mean", "dq_std", "sq_mean", "sq_std", "dq_minus_sq"])
            for l, d, d_std, s, s_std in rows:
                w.writerow([l, f"{d:.4f}", f"{d_std:.4f}", f"{s:.4f}", f"{s_std:.4f}", f"{d-s:.4f}"])
        print(f"\nWrote block-level CSV: {cli.csv_out}")
        # Per-projection CSV
        proj_csv = f"{base}_per_proj.{ext}"
        with open(proj_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "projection", "dq_mean", "dq_std", "sq_mean", "sq_std", "dq_minus_sq"])
            for l, proj, d, d_std, s, s_std in proj_rows:
                w.writerow([l, proj, f"{d:.4f}", f"{d_std:.4f}", f"{s:.4f}", f"{s_std:.4f}", f"{d-s:.4f}"])
        print(f"Wrote per-projection CSV: {proj_csv}")
        # Logit SQNR summary row
        logit_csv = f"{base}_logits.{ext}"
        with open(logit_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "dq_mean", "dq_std", "sq_mean", "sq_std", "dq_minus_sq"])
            if logit_sqnr:
                dq_l  = logit_sqnr['dq'];  dq_std_l = logit_sqnr.get('dq_std', float('nan'))
                sq_l  = logit_sqnr['sq'];  sq_std_l = logit_sqnr.get('sq_std', float('nan'))
                w.writerow(["logits", f"{dq_l:.4f}", f"{dq_std_l:.4f}",
                            f"{sq_l:.4f}", f"{sq_std_l:.4f}", f"{dq_l-sq_l:.4f}"])
        print(f"Wrote logit SQNR CSV:     {logit_csv}")

    # ── PDF ───────────────────────────────────────────────────────────────────
    # Priority: --pdf-out > derived from --csv-out > auto-generated default
    pdf_path = cli.pdf_out
    if pdf_path is None and cli.csv_out:
        pdf_path = cli.csv_out.rsplit(".", 1)[0] + ".pdf"
    if pdf_path is None:
        model_tag = cli.model.replace("/", "_")
        iso   = f"_isolate{cli.isolate_layer}" if cli.isolate_layer is not None else ""
        split = f"_{cli.data_split}" if cli.data_split != "test" else ""
        pdf_path = os.path.join(
            _HERE, f"sqnr_{model_tag}_{cli.weight_fmt}_act{cli.act_quant}{iso}{split}.pdf"
        )
    if pdf_path:
        _save_pdf(pdf_path, cli, n_layers, n_use,
                  block_sqnr, proj_sqnr, logit_sqnr, proj_names, after_attn_rows)
        print(f"Wrote PDF: {pdf_path}")


def _save_pdf(path, cli, n_layers, n_use,
              block_sqnr, proj_sqnr, logit_sqnr, proj_names, after_attn_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    model_tag = cli.model.replace("/", "_")
    layers = list(range(n_layers))
    iso_tag   = f"  |  isolate_layer={cli.isolate_layer}" if cli.isolate_layer is not None else ""
    split_tag = f"  |  data={cli.data_split}"
    title_base = (f"{model_tag}  |  {cli.weight_fmt}  |  act_quant={cli.act_quant}"
                  f"{iso_tag}{split_tag}  |  {n_use} samples × {cli.seqlen} tokens")

    def _make_table(fig, ax, header, rows, title):
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=10)
        tbl = ax.table(
            cellText=rows,
            colLabels=header,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(list(range(len(header))))
        # Header row styling
        for j in range(len(header)):
            tbl[0, j].set_facecolor("#2c3e50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # Alternate row shading
        for i in range(1, len(rows) + 1):
            for j in range(len(header)):
                tbl[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")

    with PdfPages(path) as pdf:

        # ── Page 1: logit SQNR summary table ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.suptitle(title_base, fontsize=10)
        header = ["Metric", "DQ mean ± std (dB)", "SQ mean ± std (dB)", "DQ − SQ (dB)"]
        if logit_sqnr:
            dq_l = logit_sqnr['dq'];  dq_std_l = logit_sqnr.get('dq_std', float('nan'))
            sq_l = logit_sqnr['sq'];  sq_std_l = logit_sqnr.get('sq_std', float('nan'))
            rows = [["Logit SQNR",
                     f"{dq_l:.2f} ± {dq_std_l:.2f}",
                     f"{sq_l:.2f} ± {sq_std_l:.2f}",
                     f"{dq_l-sq_l:+.2f}"]]
        else:
            rows = [["Logit SQNR", "N/A", "N/A", "N/A"]]
        _make_table(fig, ax, header, rows, "Logit SQNR  —  directly predicts PPL")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: per-block SQNR table ──────────────────────────────────────
        iso = cli.isolate_layer
        display_layers = [iso] if iso is not None else layers

        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(display_layers) * 0.38 + 2.5)))
        fig.suptitle(title_base, fontsize=10)

        header = ["Layer", "DQ mean ± std", "SQ mean ± std", "DQ − SQ"]
        rows = [[str(l),
                 f"{block_sqnr[l][0]:.2f} ± {block_sqnr[l][1]:.2f}",
                 f"{block_sqnr[l][2]:.2f} ± {block_sqnr[l][3]:.2f}",
                 f"{block_sqnr[l][0]-block_sqnr[l][2]:+.2f}"]
                for l in display_layers]
        if iso is None:
            mean_dq = sum(block_sqnr[l][0] for l in layers) / n_layers
            mean_sq = sum(block_sqnr[l][2] for l in layers) / n_layers
            rows.append(["Mean", f"{mean_dq:.2f}", f"{mean_sq:.2f}", f"{mean_dq-mean_sq:+.2f}"])
        _make_table(fig, axes[0], header, rows, "Full-block SQNR  (xᵢ + aᵢ + mᵢ)")

        if after_attn_rows:
            disp_attn = [(l, d, d_std, s, s_std) for l, d, d_std, s, s_std in after_attn_rows
                         if iso is None or l == iso]
            rows2 = [[str(l),
                      f"{d:.2f} ± {d_std:.2f}",
                      f"{s:.2f} ± {s_std:.2f}",
                      f"{d-s:+.2f}"]
                     for l, d, d_std, s, s_std in disp_attn]
            if iso is None:
                m_dq = sum(r[1] for r in after_attn_rows) / len(after_attn_rows)
                m_sq = sum(r[3] for r in after_attn_rows) / len(after_attn_rows)
                rows2.append(["Mean", f"{m_dq:.2f}", f"{m_sq:.2f}", f"{m_dq-m_sq:+.2f}"])
            _make_table(fig, axes[1], header, rows2, "After-attention SQNR  (xᵢ + aᵢ)")
        else:
            axes[1].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 3: per-projection mean SQNR table ────────────────────────────
        if proj_names:
            fig, ax = plt.subplots(figsize=(11, max(4, len(proj_names) * 0.5 + 1.5)))
            fig.suptitle(title_base, fontsize=10)
            proj_layers = display_layers
            label = f"layer {iso}" if iso is not None else "mean over all layers"
            header = ["Projection", "DQ mean ± std", "SQ mean ± std", "DQ − SQ"]
            rows = []
            for p in proj_names:
                entries = [proj_sqnr[(l, p)] for l in proj_layers if (l, p) in proj_sqnr]
                if not entries:
                    continue
                md    = sum(e[0] for e in entries) / len(entries)
                ms    = sum(e[2] for e in entries) / len(entries)
                d_std = sum(e[1] for e in entries) / len(entries)  # mean of per-layer stds
                s_std = sum(e[3] for e in entries) / len(entries)
                rows.append([p, f"{md:.2f} ± {d_std:.2f}", f"{ms:.2f} ± {s_std:.2f}", f"{md-ms:+.2f}"])
            _make_table(fig, ax, header, rows, f"Per-projection SQNR  ({label})")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Pages 4+: one table per projection, DQ vs SQ side by side ──────────
        if proj_names:
            header = ["Layer", "DQ mean ± std", "SQ mean ± std", "DQ − SQ"]
            pairs = [proj_names[i:i+2] for i in range(0, len(proj_names), 2)]
            for pair in pairs:
                ncols = len(pair)
                fig, axes = plt.subplots(1, ncols,
                                         figsize=(8 * ncols, max(5, len(display_layers) * 0.38 + 2.5)))
                fig.suptitle(title_base, fontsize=10)
                if ncols == 1:
                    axes = [axes]
                for ax, p in zip(axes, pair):
                    rows = []
                    for l in display_layers:
                        if (l, p) in proj_sqnr:
                            d, d_std, s, s_std = proj_sqnr[(l, p)]
                            rows.append([str(l),
                                         f"{d:.2f} ± {d_std:.2f}",
                                         f"{s:.2f} ± {s_std:.2f}",
                                         f"{d-s:+.2f}"])
                        else:
                            rows.append([str(l), "-", "-", "-"])
                    if iso is None:
                        entries = [proj_sqnr[(l,p)] for l in layers if (l,p) in proj_sqnr]
                        if entries:
                            md = sum(e[0] for e in entries)/len(entries)
                            ms = sum(e[2] for e in entries)/len(entries)
                            rows.append(["Mean", f"{md:.2f}", f"{ms:.2f}", f"{md-ms:+.2f}"])
                    _make_table(fig, ax, header, rows, f"Projection: {p}")
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


if __name__ == "__main__":
    main()
