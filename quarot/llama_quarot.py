"""R1 + R2 + R4 orchestrator for QuaRot on Llama-style models.

Calls the byte-identical vendored functions (`rotation_utils`, `model_utils`,
`hadamard_utils`) from spcl/QuaRot, with two adaptations the reference
doesn't need:

  1. We skip the **embedding mean-subtraction** step at the top of the
     reference's `fuse_layer_norms`. That step is only correct for LayerNorm-
     based models (OPT). For RMSNorm models (Llama / Qwen / Mistral) it
     actively breaks pre-quantisation equivalence: RMSNorm doesn't subtract
     the mean, so removing the mean from the embedding shifts the residual
     stream and changes the RMSNorm output.

  2. If `model.config.tie_word_embeddings=True` (e.g. Llama-3.2-1B), the
     reference would corrupt the embedding when it fuses the final RMSNorm γ
     into lm_head. We untie before any modification.

R3 (online Q/K Hadamard for K-cache quant) is intentionally skipped — there
is no KV-cache quantisation in this codebase.

R1 is mathematically lossless. R2 and R4 introduce small additional rotations
that are not exactly cancelled in the forward pass (the QuaRot paper accepts
this — it's swamped by the gain on outlier-smoothing post-quant). So the
"R1 alone is bit-equivalent" smoke test no longer holds once R2 + R4 fire.
"""

import torch
import tqdm
import transformers

# Trigger sys.modules stubs + sys.path setup before importing the vendored files.
from . import vendor  # noqa: F401  (ensures `quarot/__init__.py` ran)

from .vendor import model_utils, rotation_utils, utils  # noqa: E402


# Extend model_type_extractor to recognize Qwen2/Qwen3 and Mistral as
# Llama-shaped. Each rotation_utils.rotate_* function only ever compares the
# returned model_type against the constant `model_utils.LLAMA_MODEL`, so
# returning LLAMA_MODEL for these families routes them through the same
# rotation paths. Their attribute layout (q/k/v/o_proj, up/gate/down_proj,
# input_layernorm, post_attention_layernorm, model.norm, embed_tokens,
# lm_head) is structurally identical to Llama; Qwen3's extra per-head
# q_norm / k_norm sit on the head_dim axis (downstream of q/k_proj OUTPUT)
# and are not touched by R1.
#
# Import dance: the vendored rotation_utils.py does `import model_utils` —
# a top-level import resolved via sys.path (quarot/__init__.py inserts
# `vendor/` onto sys.path). When THIS file does `from .vendor import
# model_utils`, Python loads the same source file again under the relative
# name `quarot.vendor.model_utils`, creating a SECOND module object in
# sys.modules. Patching only the relative one leaves rotation_utils still
# bound to the top-level (unpatched) one. We patch both copies (and
# `get_model_type` too) so every reachable caller sees the extended
# behaviour.
import sys as _sys
_module_objects_to_patch = {model_utils, _sys.modules.get("model_utils", model_utils)}
_ORIG_EXTRACTOR = model_utils.model_type_extractor
_ORIG_GET_MODEL_TYPE = model_utils.get_model_type


def _llama_family_extractor(model):
    cls_name = type(model).__name__
    if "Qwen" in cls_name or "Mistral" in cls_name:
        return model_utils.LLAMA_MODEL
    return _ORIG_EXTRACTOR(model)


def _llama_family_get_model_type(model):
    cls_name = type(model).__name__
    if "Qwen" in cls_name or "Mistral" in cls_name:
        return model_utils.LLAMA_MODEL
    return _ORIG_GET_MODEL_TYPE(model)


for _mu in _module_objects_to_patch:
    _mu.model_type_extractor = _llama_family_extractor
    _mu.get_model_type = _llama_family_get_model_type


def _untie_lm_head(model) -> None:
    """Force embed_tokens.weight and lm_head.weight to use distinct storage."""
    cloned = model.lm_head.weight.data.clone()
    model.lm_head.weight = torch.nn.Parameter(cloned)
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
    # Verify the two weights now have different storage. If they share storage,
    # any in-place modification of one (e.g. rotate_embeddings doing W @ Q on
    # embed_tokens.weight.data) will silently mutate the other too — and the
    # subsequent rotate_head(W @ Q) would re-rotate the already-rotated tensor,
    # giving (W @ Q²) instead of (W @ Q). That's exactly the "double rotation"
    # failure mode we want to rule out for Llama-3.2-1B (tied embeddings).
    emb = model.model.embed_tokens.weight
    head = model.lm_head.weight
    if emb.data_ptr() == head.data_ptr():
        raise RuntimeError(
            "_untie_lm_head failed: embed_tokens.weight and lm_head.weight "
            "still share storage. rotate_embeddings + rotate_head would "
            "double-apply Q to the same tensor."
        )


class _RMSNbf16Safe(torch.nn.Module):
    """RMSNorm replacement byte-for-byte equivalent to HF `LlamaRMSNorm` with
    gamma=1 (which is the post-fuse state).

    Two changes vs the vendored `model_utils.RMSN`:

    1. dtype cast: vendored casts only for fp16, leaving bf16 inputs in bf16
       and destroying variance precision (bf16 has ~7 mantissa bits; summing
       hidden-size squared values overflows the significand). We cast for any
       non-fp32 dtype, matching HF's `hidden_states.to(torch.float32)`.

    2. variance reduction: we use `.mean(-1)` instead of `.sum(-1) / mean_dim`.
       Mathematically the two are equal when `mean_dim == x.shape[-1]`, but the
       hardcoded form silently produces a wrong scale if a smaller / different
       last-axis tensor is ever passed through (which would break in subtle
       ways under any future architecture change or padding-aware path).
       `.mean(-1)` is dynamic and matches HF exactly.

    `mean_dim` is kept on `self` for state_dict compatibility with the vendored
    RMSN but is no longer used in forward.

    `self.weight` is a dummy scalar Parameter kept for state_dict compatibility
    with the vendored RMSN; gamma is baked into the adjacent linear's weight
    by `fuse_ln_linear` before we replace, so this Parameter is intentionally
    unused.
    """
    def __init__(self, mean_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if input_dtype != torch.float32:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def _fuse_layer_norms_rmsnorm_only(model) -> None:
    """Llama/Qwen/Mistral-shape RMSNorm fusion.

    Equivalent to `rotation_utils.fuse_layer_norms` for Llama EXCEPT we skip
    the OPT-only embedding mean-subtraction step at the top.

    Replacement strategy: target the three residual-stream RMSNorms by
    attribute path (input_layernorm, post_attention_layernorm, model.norm)
    rather than by class. The reference uses `replace_modules(model,
    LlamaRMSNorm, ...)` which walks the module tree by type — that works
    for Llama because every RMSNorm is on the residual stream. Qwen3
    (and Llama-4) add per-head q_norm / k_norm inside self_attn that
    normalize head_dim, NOT the residual; those have their own gamma and
    are not touched by R1, so they must keep their original class and
    weights. Targeting by attribute path lets us replace the right three
    norms on any RMSNorm-based decoder without inadvertently eating
    per-head norms.
    """
    for layer in model.model.layers:
        rotation_utils.fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj],
        )
        rotation_utils.fuse_ln_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )

    rotation_utils.fuse_ln_linear(model.model.norm, [model.lm_head])

    rms_eps = float(getattr(model.config, "rms_norm_eps", 1e-5))
    hidden = int(model.config.hidden_size)
    # Use the bf16-safe RMSN, not the vendored one — see _RMSNbf16Safe docstring.
    for layer in model.model.layers:
        layer.input_layernorm = _RMSNbf16Safe(hidden, eps=rms_eps)
        layer.post_attention_layernorm = _RMSNbf16Safe(hidden, eps=rms_eps)
    model.model.norm = _RMSNbf16Safe(hidden, eps=rms_eps)


# `@torch.no_grad()` rather than `@torch.inference_mode()`: in torch 2.7+ the
# tensors created inside inference_mode are marked "inference tensors" and
# raise "Inference tensors do not track version counter" when later passed
# through F.linear during calibration / eval. no_grad has no such side-effect.
@torch.no_grad()
def apply_r1(model, rotate_mode: str = "random", apply_r2_r4: bool = True,
             verbose: bool = False) -> None:
    """In-place: fuse RMSNorms then apply R1 (and optionally R2 + R4) rotations.

    R1 alone is mathematically lossless. R2 + R4 are only lossless if their
    online counterparts fire at forward time (via `install_online_hadamards`
    setting flags on PermLinear). When running weight-only quant (no PermLinear
    installed), pass `apply_r2_r4=False` so the offline R2/R4 folds don't
    drift the model output.

    Args:
        model: a Llama-style HF model (LlamaForCausalLM / Qwen / Mistral).
        rotate_mode: 'random' (QR-of-Gaussian) or 'hadamard' (randomized
                     Hadamard: H_n times a random ±1 diagonal). Hadamard
                     requires hidden_size be a power of 2 or one of the
                     special sizes recognised by `get_hadK`.
        apply_r2_r4: if False, only R1 is applied. Set False for weight-only
                     quant or whenever activation quantisation is OFF.
        verbose: print progress.
    """
    if getattr(model.config, "tie_word_embeddings", False):
        if verbose:
            print("[quarot] untying lm_head from embed_tokens before R1")
        _untie_lm_head(model)

    if verbose:
        print("[quarot] fusing RMSNorms into adjacent linears (Llama/RMSNorm path)")
    _fuse_layer_norms_rmsnorm_only(model)

    Q = rotation_utils.get_orthogonal_matrix(model.config.hidden_size, rotate_mode)
    if verbose:
        print(f"[quarot] R1 Q: {tuple(Q.shape)}  mode={rotate_mode}  device={Q.device}")

    model_type = model_utils.model_type_extractor(model)
    rotation_utils.rotate_embeddings(model, Q)
    rotation_utils.rotate_head(model, Q)
    utils.cleanup_memory()

    cfg = model.config
    num_heads = int(cfg.num_attention_heads)
    # head_dim is not always hidden_size / num_heads. Qwen3 (and Llama-4 /
    # Gemma-3) set head_dim explicitly on the config — e.g. Qwen3-0.6B is
    # hidden_size=1024, num_heads=16, head_dim=128, so q_proj outputs
    # num_heads*head_dim=2048, NOT hidden_size. The vendored QuaRot reference
    # was written for LLaMA-2/3 + OPT where the Llama convention always held
    # and it just divides. We prefer the config value when the model exposes
    # it; otherwise fall back to the Llama convention.
    head_dim = int(getattr(cfg, "head_dim", None) or (int(cfg.hidden_size) // num_heads))

    desc = "QuaRot R1+R2+R4" if apply_r2_r4 else "QuaRot R1"
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer",
                                          desc=desc, disable=not verbose)):
        # R1 — residual-stream rotation on every block linear
        rotation_utils.rotate_attention_inputs(layer, Q, model_type)
        rotation_utils.rotate_attention_output(layer, Q, model_type)
        rotation_utils.rotate_mlp_input(layer, Q, model_type)
        if apply_r2_r4:
            # R1 + R4: `rotate_mlp_output` applies Q^T to down_proj weight (R1)
            # AND calls apply_exact_had_to_linear (R4 offline half).
            rotation_utils.rotate_mlp_output(layer, Q, model_type)
            # R2: per-head Hadamard on v_proj output + full H on o_proj input
            rotation_utils.rotate_ov_proj(layer, model_type, num_heads, head_dim)
        else:
            # R1-only down_proj: replicates the R1 portion of rotate_mlp_output
            # without the R4 apply_exact_had_to_linear step. Math copied
            # verbatim from rotation_utils.rotate_mlp_output:171-176.
            W = layer.mlp.down_proj
            dtype = W.weight.dtype
            dev = W.weight.device
            W_ = W.weight.data.to(device=Q.device, dtype=torch.float64)
            W.weight.data = torch.matmul(Q.T, W_).to(device=dev, dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=Q.device, dtype=torch.float64)
                W.bias.data = torch.matmul(Q.T, b).to(device=dev, dtype=dtype)

    if verbose:
        print(f"[quarot] {desc} applied")
