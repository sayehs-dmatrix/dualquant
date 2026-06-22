"""SmoothQuant preprocess.

Migrates magnitude from activations into weights via per-channel scales,
making activations easier to quantise. Applied to the model in-place
BEFORE per-layer quantisation runs.

preprocess_cfg keys:
    act_scales_path  (str)   — path to a .pt file of per-tensor activation
                                statistics (saved by the upstream calibration
                                run; lives in legacy/act_scales/<model>.pt).
    alpha            (float, default 0.5) — migration strength.
"""

import _legacy_path  # noqa: F401

import os
import torch
# from smoothquant.smooth import smooth_lm     # legacy
from smoothquant.smooth_with_scale_dict import smooth_lm

_CODEBASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCALES_DIR = os.path.join(_CODEBASE_ROOT, "scales_smoothquant_rtn_int4")


def apply_smoothquant(model, preprocess_cfg):
    if "act_scales_path" not in preprocess_cfg:
        raise ValueError("smoothquant preprocess requires preprocess_cfg['act_scales_path']")
    act_scales = torch.load(preprocess_cfg["act_scales_path"])
    alpha = preprocess_cfg.get("alpha", 0.5)
    weight_stat = preprocess_cfg.get("weight_stat", "max_abs")
    saved_scales = smooth_lm(model, act_scales, alpha, weight_stat=weight_stat)
    ############## Save debugging information ################
    model_name = model.config._name_or_path
    safe_model_name = model_name.replace("/", "_")
    scales_dict = {
        "model_name": model_name,
        "alpha": alpha,
        "weight_stat": weight_stat,
        "saved_scales": saved_scales,
        "act_scales": act_scales
    }
    os.makedirs(_SCALES_DIR, exist_ok=True)
    filename = os.path.join(_SCALES_DIR, f"SQ_scales_{safe_model_name}_alpha_{alpha}_wstat_{weight_stat}.pt")
    torch.save(scales_dict, filename)
    print(f"Saved to: {filename}")
    ##########################################################
    return model
