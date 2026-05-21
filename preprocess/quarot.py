"""QuaRot preprocess — calibration-free rotation of the residual stream.

R1 is always applied (mathematically lossless). R2 + R4 are applied only
when the activation-quant pipeline is on, because their offline halves need
the online halves (installed in PermLinear by main.py) to cancel and stay
lossless.

preprocess_cfg keys (all optional):
    rotate_mode  (str, default "random")  — "random" (QR of Gaussian) or
                                            "hadamard" (randomized Hadamard
                                            à la QuIP: H_n times a random ±1
                                            diagonal; requires hidden_size be
                                            a power of 2 or one of the special
                                            sizes in get_hadK)
    seed         (int, default None)      — global torch seed used right
                                            before generating Q
    r2_r4        (bool, default auto)     — explicit override. If unset,
                                            defaults to `act_quant_enabled`
                                            (injected by main.py).
    verbose      (bool, default False)    — print progress

Architecture: this preprocess rotates the model in-place; subsequent
quantisation runs unchanged on the rotated weights. The "include GPTQ on top"
toggle is just which `--method` flag the user supplies (`rtn` = no GPTQ,
`gptq` = with GPTQ). Same weight formats (mxfp4, mxint4, mxfp8, nvfp4, etc.)
work as for the other quantisation paths.
"""

import os
import sys

import torch

# Make the sibling `quarot/` package importable regardless of cwd.
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODEBASE_ROOT not in sys.path:
    sys.path.insert(0, _CODEBASE_ROOT)

from quarot import apply_r1


def apply_quarot(model, preprocess_cfg):
    rotate_mode = preprocess_cfg.get("rotate_mode", "random")
    seed = preprocess_cfg.get("seed", None)
    verbose = bool(preprocess_cfg.get("verbose", False))

    # R2/R4 default = act_quant_enabled (auto-detect from main.py). User can
    # override via preprocess_cfg["r2_r4"].
    act_quant_enabled = bool(preprocess_cfg.get("_act_quant_enabled", False))
    apply_r2_r4 = bool(preprocess_cfg.get("r2_r4", act_quant_enabled))

    if seed is not None:
        torch.manual_seed(int(seed))
    apply_r1(model, rotate_mode=rotate_mode, apply_r2_r4=apply_r2_r4, verbose=verbose)
    return model
