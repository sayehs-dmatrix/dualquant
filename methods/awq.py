"""AWQ (activation-aware weight quantisation).

Uses calibration activations to find a per-input-channel scale s that
minimises output error after quantising W*s, then folds s back out so the
runtime path is unchanged.

Math is delegated to the legacy `compute_awq_scale`; the inner fake-quant
is the chosen weight format's block cast.
"""

import _legacy_path  # noqa: F401

import torch
from awq import compute_awq_scale         # legacy

from .base import QuantMethod


class AWQ(QuantMethod):
    name = "awq"
    needs_calib_acts = True

    def wrap(self, layer, cfg, calib_acts=None, layer_key=None):
        # layer_key consumed only by Dualquant for scale-saving; ignored here.
        if calib_acts is None:
            raise ValueError(f"{self.name!r} requires calibration activations")
        weight_fmt = cfg["weight_fmt"]

        with torch.no_grad():
            dtype = layer.weight.dtype
            W = layer.weight.detach().to(torch.float32).cpu()
            X = calib_acts.cpu().float()

            def fake_quant_fn(W_scaled):
                return weight_fmt.cast(W_scaled.to("cuda")).cpu()

            scale = compute_awq_scale(W, X, fake_quant_fn)         # (in_features,)

            # Scale input channels up, quantise, fold scale back out
            W_q = weight_fmt.cast((W * scale.unsqueeze(0)).to("cuda")).cpu()
            W_q = W_q / scale.unsqueeze(0)

            layer.weight.copy_(W_q.to(dtype))
        return None
