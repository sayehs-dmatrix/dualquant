"""Activations package: runtime activation-quantisation.

Public surface:
    PermLinear(layer, act_fmt, quantize_bmm_input)
        — nn.Module wrapping an already-weight-quantised nn.Linear.
    install_perm_linear(parent, attr_name, act_fmt, col_scales, quantize_bmm_input)
        — replace parent.<attr_name> (an nn.Linear) with a PermLinear that
          wraps it, optionally seeding act_scale with col_scales.
"""

import torch.nn as nn

from .perm_linear import PermLinear


def install_perm_linear(parent: nn.Module, attr_name: str, act_fmt,
                        col_scales=None, quantize_bmm_input: bool = False) -> PermLinear:
    """Wrap parent.<attr_name> with a PermLinear in place.

    `col_scales` is the per-input-column scale returned by methods like
    dualquant / SINQ. Pass None for methods that don't compute one.
    """
    layer = getattr(parent, attr_name)
    perm = PermLinear(layer, act_fmt=act_fmt, quantize_bmm_input=quantize_bmm_input)
    perm.set_act_scale(col_scales)
    setattr(parent, attr_name, perm)
    return perm


__all__ = ["PermLinear", "install_perm_linear"]
