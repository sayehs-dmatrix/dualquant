"""PermLinear: runtime activation-quantisation wrapper for nn.Linear.

Installed by main.py when --act-quant on. Wraps the (already weight-quantised)
nn.Linear and applies activation quantisation at forward time.

Forward path:
    1. If act_scale is set (per-input-column tensor, e.g. 1/beta from dualquant
       or SINQ), pre-scale the input by it.
    2. If an online Hadamard is configured (QuaRot R2/R4), apply it to the
       input before activation quantisation. Combined with the offline
       Hadamard that the QuaRot preprocess already baked into the linear's
       weight, the two cancel mathematically (H @ H = I) — so the network
       output is unchanged pre-quant, but the activation quant sees the
       outlier-smoothed (rotated) value rather than the original.
    3. Quantise the input with the activation FormatSpec.
    4. Run the wrapped linear.
    5. If quantize_bmm_input, quantise the output.

Methods that produce a per-input-column scale (dualquant scale_option=row_column,
SINQ) return that scale from .wrap(); main.py passes it here via set_act_scale.
Methods that don't (RTN, AWQ, GPTQ, QuaRot) leave act_scale as None.

The online-Hadamard branch is byte-for-byte the same as the reference
spcl/QuaRot ActQuantWrapper.forward at quant_utils.py:216-243 (Apache-2.0).
Off by default; install_online_hadamards (quarot/online_had_install.py)
sets the flags on o_proj / down_proj wrappers after QuaRot preprocess.
"""

import math

import torch
import torch.nn as nn

# Trigger quarot package init (registers sys.modules stubs so the vendored
# hadamard_utils + fast_hadamard_transform pure-torch impl are available).
import quarot  # noqa: F401
import hadamard_utils
import fast_hadamard_transform


class PermLinear(nn.Module):
    def __init__(self, layer: nn.Linear, act_fmt, quantize_bmm_input: bool = False):
        super().__init__()
        self.layer = layer
        self.dtype = layer.weight.dtype
        self.act_fmt = act_fmt                              # FormatSpec
        self.quantize_bmm_input = quantize_bmm_input
        self.register_buffer("act_scale", None)             # set later via set_act_scale

        # QuaRot online Hadamard state. Defaults make _quantise_act behave
        # exactly as before for non-QuaRot paths.
        self.online_full_had = False
        self.online_partial_had = False
        self.had_K = None
        self.K = 1
        self.had_dim = 0
        self.fp32_had = False

        # When False, act_fmt.cast is skipped (online Hadamard still fires
        # if configured). Mirrors QuaRot's ActQuantizer with bits=16: the
        # wrapper exists so its online-Hadamard branch runs, but the
        # quantizer itself is a pass-through. Used to keep GPTQ calibration
        # at FP precision while the offline/online Hadamards still cancel.
        self.act_quant_enabled = True

    def set_act_scale(self, act_scale):
        if act_scale is not None:
            act_scale = act_scale.detach().clone().to(torch.float32)
        self.act_scale = act_scale

    def set_act_quant_enabled(self, enabled: bool) -> None:
        """Toggle the act_fmt.cast step on/off. Online Hadamard branch is
        unaffected. Used by gptq_seq to disable during calibration."""
        self.act_quant_enabled = bool(enabled)

    def _apply_online_hadamard(self, x):
        """Online Hadamard rotation. Byte-for-byte from spcl/QuaRot
        ActQuantWrapper.forward (quant_utils.py:220-242)."""
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:

            if self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),
                                                               scale=1/math.sqrt(init_shape[-1]//self.had_dim)).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        return x

    def _quantise_act(self, x):
        # Only upcast to fp32 when we have a numerically sensitive intermediate
        # to perform (act_scale multiplication, Hadamard rotation). Casting
        # unnecessarily before act_fmt.cast changes the rounding behaviour at
        # the INT4 grid boundaries — the legacy `modeling_perm_llama.PermLinear`
        # quantises activations in the input dtype (bf16 for Llama-3.2-1B),
        # and that path gives PPL ~12 on Llama-3.2-1B W4-A4 RTN. Upcasting
        # to fp32 here drifts PPL to ~14 because borderline values round to
        # different INT4 levels under bf16 vs fp32 storage. Match legacy.
        needs_fp32 = (
            self.act_scale is not None
            or self.online_full_had
            or self.online_partial_had
        )
        if needs_fp32:
            x32 = x.to(torch.float32)
            if self.act_scale is not None:
                x32 = x32 * self.act_scale
            if self.online_full_had or self.online_partial_had:
                x32 = self._apply_online_hadamard(x32)
            if self.act_quant_enabled:
                x32 = self.act_fmt.cast(x32)
            return x32.to(self.dtype)
        else:
            if self.act_quant_enabled:
                return self.act_fmt.cast(x).to(self.dtype)
            return x

    def forward(self, x):
        x_q = self._quantise_act(x)
        out = self.layer(x_q)
        if self.quantize_bmm_input:
            out = self.act_fmt.cast(out.to(torch.float32)).to(self.dtype)
        return out
