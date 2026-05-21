"""GPTQ (Hessian-based weight quantisation).

Adapter over the legacy `GPTQ` class + per-format quantizer factories. The
heavy lifting (Hessian accumulation, Cholesky solve, column-by-column /
block-by-block quantisation with error propagation) is unchanged.

method_cfg keys (all optional):
    blocksize  (int, default 128)   — outer GPTQ block size
    percdamp   (float, default 0.01) — Hessian damping percentage
"""

import _legacy_path  # noqa: F401

import torch
from layer_wrapper_baseline_data_formats_methods import (    # legacy
    GPTQ,
    mxfp4_quantizer_cls,
    mxfp8_e4m3_quantizer_cls,
    mxfp8_e5m2_quantizer_cls,
    mxint4_quantizer_cls,
    mxint8_quantizer_cls,
    nBits_quantizer,
    nvfp4_quantizer_cls,
    sfp4_quantizer_cls,
)

from .base import QuantMethod


def _gptq_quantizer_for(format_name, block_size):
    if format_name == "sfp4":
        q = sfp4_quantizer_cls(block_size=block_size)
        q.configure(bits=4)
        return q
    if format_name == "nvfp4":
        return nvfp4_quantizer_cls(block_size=block_size)
    if format_name == "mxfp4":
        return mxfp4_quantizer_cls(block_size=block_size)
    if format_name == "mxint4":
        return mxint4_quantizer_cls(block_size=block_size)
    if format_name == "mxint8":
        return mxint8_quantizer_cls(block_size=block_size)
    if format_name == "mxfp8_e4m3":
        return mxfp8_e4m3_quantizer_cls(block_size=block_size)
    if format_name == "mxfp8_e5m2":
        return mxfp8_e5m2_quantizer_cls(block_size=block_size)
    if format_name == "rtn_int4":
        return nBits_quantizer(bits=4, group_size=-1, sym=True)
    if format_name == "rtn_int8":
        return nBits_quantizer(bits=8, group_size=-1, sym=True)
    raise ValueError(f"GPTQ: unsupported weight format {format_name!r}")


def _fq_groupsize_for(format_name, block_size, columns):
    if format_name == "rtn_int8":
        return columns                          # per-channel affine (one scale per row)
    if format_name == "rtn_int4":
        return block_size                       # standard grouped INT4
    if format_name in ("sfp4", "nvfp4"):
        return 16                               # both formats are fixed at 16-element blocks
    return block_size                           # MX formats: block_size from config


class GPTQMethod(QuantMethod):
    name = "gptq"
    needs_calib_acts = True

    def wrap(self, layer, cfg, calib_acts=None, layer_key=None):
        # layer_key is consumed by Dualquant for scale-saving; ignored here.
        if calib_acts is None:
            raise ValueError(f"{self.name!r} requires calibration activations")
        weight_fmt = cfg["weight_fmt"]
        method_cfg = cfg["method_cfg"]
        blocksize = method_cfg.get("blocksize", 128)
        percdamp = method_cfg.get("percdamp", 0.01)

        with torch.no_grad():
            gptq_obj = GPTQ(layer)
            gptq_obj.quantizer = _gptq_quantizer_for(weight_fmt.name, weight_fmt.block_size)

            gptq_obj.add_batch(calib_acts.to(gptq_obj.dev), None)

            fq_groupsize = _fq_groupsize_for(
                weight_fmt.name, weight_fmt.block_size, gptq_obj.columns
            )
            gptq_obj.fasterquant(
                blocksize=blocksize, percdamp=percdamp, groupsize=fq_groupsize
            )
            gptq_obj.free()
        return None
