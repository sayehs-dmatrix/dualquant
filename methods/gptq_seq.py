"""Sequential GPTQ — port of QuaRot fake_quant/gptq_utils.gptq_fwrd (Apache-2.0).

Unlike the existing `GPTQMethod` (parallel calibration), this method handles
its own layer iteration. Each layer's calibration activations are captured
AFTER upstream layers are quantised, so the Hessian reflects what the layer
will actually see at inference time. This is required for low-bit weight
formats (rtn_int4) where the unquantised-vs-quantised activation drift would
otherwise compound to catastrophic PPL.

The orchestrator pattern is byte-for-byte from QuaRot's gptq_fwrd:
  1. Capture inputs to layer 0 via a Catcher module that intercepts forward.
  2. For each block in order:
     a. For each sub-group (k/v/q  →  o  →  up/gate  →  down):
        - register hooks on the sub-group's linears
        - forward this layer with the calibration batch
        - hooks populate per-linear GPTQ Hessians
        - call fasterquant on each linear (uses our format quantizers)
        - remove hooks
     b. Re-forward this layer (now fully quantised) to populate `outs`.
  3. Swap inps/outs — next block reads quantised output of this block.

Adapter glue vs the reference:
  - Reference uses `quant_utils.WeightQuantizer` (INT-only). We use
    `_gptq_quantizer_for(format_name, block_size)` from `methods/gptq.py`
    so the same code works for rtn_int4, mxfp4, nvfp4, mxint4, etc.
  - Reference iterates ActQuantWrapper-wrapped Linears (names ending
    `.module`). We iterate the raw `nn.Linear`s before any wrapping —
    PermLinear gets installed AFTER weight quantisation in main.py.
"""

import _legacy_path  # noqa: F401

import torch
import torch.nn as nn

from awq import get_calib_dataset                     # legacy
from layer_wrapper_baseline_data_formats_methods import GPTQ as _LegacyGPTQ  # legacy

from .base import QuantMethod
from .gptq import _fq_groupsize_for, _gptq_quantizer_for


class GPTQSequential(QuantMethod):
    """Sequential GPTQ. Use this for INT4 weight formats; use plain `gptq`
    for block formats (mxfp4 etc.) where parallel calibration is tolerable
    and ~10× faster."""

    name = "gptq_seq"
    needs_calib_acts = False    # we collect our own calibration sequentially

    def wrap(self, *args, **kwargs):
        raise NotImplementedError(
            "gptq_seq runs as a whole-model orchestration via .quantise_model(); "
            "the per-linear .wrap() API does not apply."
        )

    @torch.no_grad()
    def quantise_model(self, model, tokenizer, args, cfg):
        """End-to-end quantisation: weights via sequential GPTQ, then (if
        --act-quant on) PermLinear install + QuaRot online-Hadamard install.

        Owns the whole orchestration so main.py only needs to dispatch here.

        Ordering note: when QuaRot is on AND activation quant is enabled, the
        offline R2/R4 halves have already been baked into o_proj/down_proj
        weights by apply_r1. Those halves only cancel when the online halves
        ALSO fire at forward time — i.e. only after PermLinear + online-Had
        flags are installed. So we install PermLinear + online Hadamards
        FIRST, then run sequential GPTQ. The GPTQ hooks target the inner
        nn.Linear inside each PermLinear, so the captured Hessian reflects
        the true inference-time input (post-act-quant + post-online-Had).
        Matches spcl/QuaRot's reference flow (wrap layers, then gptq_fwrd
        hooks `<name>.module`).
        """
        weight_fmt = cfg["weight_fmt"]
        act_cfg = cfg["act_quant"]
        act_enabled = bool(act_cfg.get("enabled", False))
        method_cfg = cfg["method_cfg"]
        blocksize = int(method_cfg.get("blocksize", 128))
        percdamp = float(method_cfg.get("percdamp", 0.01))

        nsamples = int(args.nsamples)
        seqlen = int(args.seqlen)
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        quarot_with_acts = (args.preprocess == "quarot" and act_enabled)

        # ── Pre-Phase 1 (quarot + act-quant only): install PermLinear with
        #    online R2/R4 BEFORE sequential GPTQ so calibration sees the true
        #    inference pipeline. See ordering note above. ─────────────────────
        if quarot_with_acts:
            _install_perm_linears(model, args, act_cfg)
            from quarot import install_online_hadamards  # local: avoid circular
            install_online_hadamards(model, verbose=True)

            # Match QuaRot's reference flow byte-for-byte: during GPTQ their
            # ActQuantizer is at bits=16 (no-op), so the hook sees post-Hadamard
            # but PRE-act-quant inputs. We toggle off `act_quant_enabled` so
            # `_quantise_act` skips `act_fmt.cast` while online Hadamards still
            # fire. Re-enabled after GPTQ for inference.
            from activations.perm_linear import PermLinear  # local: avoid circular
            for mod in model.modules():
                if isinstance(mod, PermLinear):
                    mod.set_act_quant_enabled(False)

        # ── Phase 1: sequential GPTQ on weights ──────────────────────────────
        calib = get_calib_dataset(
            tokenizer=tokenizer, n_samples=max(nsamples * 4, 64), max_seq_len=seqlen
        )
        calib = calib[:nsamples]
        if not calib:
            raise RuntimeError("gptq_seq: get_calib_dataset returned no samples")

        _gptq_fwrd_adapter(
            model, calib, dev,
            weight_fmt=weight_fmt,
            blocksize=blocksize,
            percdamp=percdamp,
            nsamples=len(calib),
            seqlen=seqlen,
            linear_name_suffix=".layer" if quarot_with_acts else "",
        )

        # ── Post-GPTQ: re-enable act-quant on every PermLinear (only the
        #    quarot+act path turned it off above). ─────────────────────────
        if quarot_with_acts:
            from activations.perm_linear import PermLinear  # local: avoid circular
            for mod in model.modules():
                if isinstance(mod, PermLinear):
                    mod.set_act_quant_enabled(True)

        # ── Phase 2: install PermLinear (only if we didn't already do it
        #    in pre-Phase 1 above). Pass col_scales=None — GPTQ doesn't
        #    compute per-channel scales. ───────────────────────────────────
        if act_enabled and not quarot_with_acts:
            _install_perm_linears(model, args, act_cfg)


def _install_perm_linears(model, args, act_cfg):
    """Wrap every block linear with PermLinear. Used both pre-GPTQ (QuaRot +
    act-quant) and post-GPTQ (act-quant without QuaRot)."""
    from activations import install_perm_linear  # local: avoid circular
    attn_names = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_names = ("up_proj", "down_proj", "gate_proj")
    for block in model.model.layers:
        if getattr(args, "attn_layers", True):
            for name in attn_names:
                is_qkv = name in ("q_proj", "k_proj", "v_proj")
                install_perm_linear(
                    block.self_attn, name,
                    act_fmt=act_cfg["fmt"],
                    col_scales=None,
                    quantize_bmm_input=(is_qkv and act_cfg.get("quantize_bmm", False)),
                )
        if getattr(args, "mlp_layers", True):
            for name in mlp_names:
                install_perm_linear(
                    block.mlp, name,
                    act_fmt=act_cfg["fmt"],
                    col_scales=None,
                    quantize_bmm_input=False,
                )


# ── inner orchestrator (byte-faithful to QuaRot fake_quant/gptq_utils.gptq_fwrd
#    where the structure can be preserved verbatim) ──────────────────────────
def _gptq_fwrd_adapter(model, dataloader, dev, *, weight_fmt, blocksize,
                       percdamp, nsamples, seqlen, linear_name_suffix=""):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Move the whole model to dev. The QuaRot reference (written for
    # transformers 4.38) used a partial-move pattern (embed + norm + layers[0])
    # because rotary_emb was per-attention back then. In transformers 5.3+
    # rotary_emb lives at model.model.rotary_emb and runs BEFORE the layer
    # loop. With partial moves it stays on CPU while hidden_states is on GPU
    # → mixed-device matmul produces garbage position_embeddings → corrupted
    # calibration → catastrophic PPL. Whole-model-on-dev is version-robust
    # and fits comfortably on A100 for models up to ~8B at bf16.
    model.to(dev)
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    hidden_size = int(model.config.hidden_size)
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            # Forward missing attribute lookups to the wrapped layer.
            # transformers 5.x reads layer-level metadata from the parent's
            # forward — Qwen3 uses `decoder_layer.attention_type` to pick the
            # causal-mask variant, Llama-4 / Gemma-3 similarly. We need every
            # such read to find the value on the underlying decoder block,
            # not the bare Catcher. nn.Module's __getattr__ already handles
            # the registered submodule lookup (self.module), so we only fall
            # back to attribute proxy for things it doesn't know about.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    fwd_kwargs = cache["kwargs"] or {}

    # Sub-group order matches the QuaRot reference and IST-DASLab's llama_sequential.
    # Within each sub-group, the linears share the same INPUT (so their
    # Hessians can be collected in one forward); the next sub-group's inputs
    # depend on the previously-quantised sub-group's outputs.
    sequential = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    for i in range(len(layers)):
        print(f"\nLayer {i} (gptq_seq):", flush=True, end=" ")
        layer = layers[i]  # already on dev (whole model is on dev)

        # When PermLinear is installed before GPTQ (QuaRot + act-quant path),
        # the inner nn.Linear is at `<name>.layer`. We register hooks on the
        # inner Linear so the captured input is post-act-quant + post-online-
        # Hadamard — i.e. the true inference-time distribution.
        full = {name: mod for name, mod in layer.named_modules()
                if isinstance(mod, nn.Linear)}

        for names in sequential:
            subset = {n: full[f"{n}{linear_name_suffix}"]
                      for n in names if f"{n}{linear_name_suffix}" in full}
            if not subset:
                continue

            gptq = {}
            for name, lin in subset.items():
                print(f"{name}", end="  ", flush=True)
                gptq[name] = _LegacyGPTQ(lin)
                gptq[name].quantizer = _gptq_quantizer_for(
                    weight_fmt.name, weight_fmt.block_size
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = [subset[n].register_forward_hook(add_batch(n))
                       for n in subset]

            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **fwd_kwargs)[0]

            for h in handles:
                h.remove()

            for name in subset:
                fq_groupsize = _fq_groupsize_for(
                    weight_fmt.name, weight_fmt.block_size, gptq[name].columns
                )
                gptq[name].fasterquant(
                    blocksize=blocksize, percdamp=percdamp, groupsize=fq_groupsize,
                    actorder=True,
                )
                w = subset[name].weight
                if torch.isnan(w).any() or torch.isinf(w).any():
                    print(f"  *** NaN/Inf in layer {i} {name} after GPTQ ***")
                gptq[name].free()

        # Re-forward through the fully-quantised layer to populate outs;
        # next iteration's inps = this layer's quantised output.
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kwargs)[0]

        out0 = outs[0]
        print(f"  layer {i} re-fwd: mean={out0.float().mean():.4f}  std={out0.float().std():.4f}  nan={torch.isnan(out0).any().item()}  inf={torch.isinf(out0).any().item()}")

        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print()  # finish the trailing per-layer line
