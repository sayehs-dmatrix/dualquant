"""Plain round-to-nearest. No calibration, no method_cfg, no alpha/beta — just format.cast()."""

import torch

from .base import QuantMethod


class RTN(QuantMethod):
    name = "rtn"
    needs_calib_acts = False

    def wrap(self, layer, cfg, calib_acts=None, layer_key=None):
        weight_fmt = cfg["weight_fmt"]
        with torch.no_grad():
            W = layer.weight.detach().to(torch.float32).cpu()
            # breakpoint()
            Wq = weight_fmt.cast(W)
            layer.weight.copy_(Wq.to(layer.weight.dtype))
        return None
