"""Dualquant adapter — calls layer_wrapper_dualquant.wrap_dualquant_layer.

The heavy lifting (alpha/beta iteration, splits, scale formats) lives in
../layer_wrapper_dualquant.py. This adapter just translates the cfg dict
into the kwargs that wrapper expects.
"""

import os
import sys
import torch 

# Add parent dir of this file (the dualquant codebase root) so the sibling
# module layer_wrapper_dualquant.py is importable.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from layer_wrapper_dualquant import wrap_dualquant_layer

from .base import QuantMethod


class Dualquant(QuantMethod):
    name = "dualquant"
    needs_calib_acts = False

    def __init__(self):
        self.collected_scales = {}   # {layer_key: column_scales tensor}

    def wrap(self, layer, cfg, calib_acts=None, layer_key=None):
        method_cfg = cfg["method_cfg"]
        weight_fmt = cfg["weight_fmt"]
        act_cfg = cfg["act_quant"]

        legacy_cfg = {
            "num_splits": method_cfg.get("num_splits", 1),
            "scale_option": method_cfg["scale_option"],         # 'row_column' | 'only_column' | 'only_row'
            "col_init": method_cfg.get("col_init", "l1_norm"),
            "row_init": method_cfg.get("row_init", "max_abs"),
            "block_size": weight_fmt.block_size,
            "num_iter": method_cfg.get("num_iter", 5),
            "scale_format": cfg["weight_scale_format"],
            "dualquant_to_element_tensor": method_cfg.get("dualquant_to_element_tensor", True),
        }
        act_quant_flag = bool(act_cfg.get("enabled") and act_cfg.get("scaled_before_quant"))

        col_scales = wrap_dualquant_layer(
            layer,
            layer_activations=calib_acts,
            opt_config=legacy_cfg,
            quant_method=weight_fmt.name,
            act_quant_flag=act_quant_flag,
        )
        
        if col_scales is not None and layer_key is not None:
            self.collected_scales[layer_key] = col_scales.detach().cpu()
        return col_scales

    def save_scales(self, model_id, num_iter, out_dir, scale_option="row_column", row_init="max_abs", col_init="l1_norm"):
        if not self.collected_scales:
            return None
        os.makedirs(out_dir, exist_ok=True)
        model_tag = model_id.replace("/", "_")
        path = os.path.join(out_dir, f"DQ_{model_tag}_iter{num_iter}_scale_{scale_option}_row_{row_init}_col_{col_init}_rtn_int4.pt")
        torch.save(self.collected_scales, path)
        return path
            
