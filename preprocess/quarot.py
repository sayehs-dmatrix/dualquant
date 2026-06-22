"""QuaRot preprocess — calibration-free rotation of the residual stream.

R1 is always applied (mathematically lossless). R2 + R4 are now also always
applied by default. Their offline halves are folded into the weights here;
main.py installs the matching online halves in PermLinear so the H @ H = I
cancellation holds at forward time — with act_quant_enabled=False for weight-
only quant (activations pass through unquantised) or act_quant_enabled=True
for full W+A quant.

preprocess_cfg keys (all optional):
    rotate_mode  (str, default "random")  — "random" (QR of Gaussian) or
                                            "hadamard" (randomized Hadamard
                                            à la QuIP: H_n times a random ±1
                                            diagonal; requires hidden_size be
                                            a power of 2 or one of the special
                                            sizes in get_hadK)
    seed         (int, default None)      — global torch seed used right
                                            before generating Q
    r2_r4        (bool, default True)     — set False to apply R1 only
                                            (disables the online Hadamard
                                            install in main.py too).
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

    # R2/R4 default = True (always apply). The online halves are installed by
    # main.py via install_online_hadamards — for act_quant=off via pass-through
    # PermLinears (act_quant_enabled=False), for act_quant=on via the normal
    # PermLinear path. User can override via preprocess_cfg["r2_r4"] = false.
    apply_r2_r4 = bool(preprocess_cfg.get("r2_r4", True))

    if seed is not None:
        torch.manual_seed(int(seed))
    apply_r1(model, rotate_mode=rotate_mode, apply_r2_r4=apply_r2_r4, verbose=verbose)
    return model
