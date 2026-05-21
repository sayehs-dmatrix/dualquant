"""FormatSpec: numeric encoding of a tensor (weight or activation).

Three orthogonal concerns are separated in this codebase:

  1. METHOD          (rtn, gptq, awq, dualquant, sinq) — how scales/transforms
                     are computed.
  2. FORMAT          (this file) — how a tensor's numeric values are encoded.
  3. SCALE FORMAT    (../scale_formats.py) — how per-row/per-block scales are
                     stored (e8m0, e4m3, e4m4, e5m3, none).

A FormatSpec exposes two casts:

  cast(W)              — block-format quant. Dispatches an internal block
                         scale on top of the element values. Used at the
                         end to produce the stored weight ("what the hardware
                         sees").
  cast_element_only(X) — element-only quant; assumes X is already pre-scaled.
                         Used inside the dualquant iteration when
                         dualquant_to_element_tensor=True.

`max_val` is the maximum representable element value (used by some methods
to initialise per-row scales).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class FormatSpec:
    name: str
    block_size: int
    max_val: float
    cast: Callable[[torch.Tensor], torch.Tensor]
    cast_element_only: Callable[[torch.Tensor], torch.Tensor]
