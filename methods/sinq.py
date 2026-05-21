"""SINQ (Sinkhorn-based dual scale quantisation).

Thin adapter over the legacy `wrap_layer_with_sinq_method`. SINQ produces a
quantised weight plus a per-row scale tensor; if act_quant.scaled_before_quant
is True the per-row scale is returned for runtime activation pre-scaling,
otherwise it's folded into the stored weight.
"""

import _legacy_path  # noqa: F401

import torch
from sinq_functions import wrap_layer_with_sinq_method     # legacy

from .base import QuantMethod


class SINQ(QuantMethod):
    name = "sinq"
    needs_calib_acts = False

    def wrap(self, layer, cfg, calib_acts=None, layer_key=None):
        # layer_key consumed only by Dualquant for scale-saving; ignored here.
        weight_fmt = cfg["weight_fmt"]
        act_cfg = cfg["act_quant"]
        act_quant_flag = bool(act_cfg.get("enabled") and act_cfg.get("scaled_before_quant"))

        with torch.no_grad():
            dtype = layer.weight.dtype
            W = layer.weight.clone()
            Wq, weight_reduce_dim_scales = wrap_layer_with_sinq_method(
                W, calib_acts, weight_fmt.name, weight_fmt.block_size,
            )
            if act_quant_flag:
                layer.weight.copy_(Wq.to(dtype))
                return weight_reduce_dim_scales
            layer.weight.copy_((Wq * weight_reduce_dim_scales).to(dtype))
        return None
