"""Collect per-Linear-layer input activations from a calibration dataset.

Thin wrapper around the legacy `awq.get_calib_dataset` and
`awq.collect_activations` until those helpers are migrated.
"""

import _legacy_path  # noqa: F401

import torch

from awq import collect_activations, get_calib_dataset    # legacy


def collect_calib_activations(model, tokenizer, n_samples=512, max_seq_len=512, num_collect=128):
    """Returns a dict {layer_name: tensor (n_collected_samples, seq, in_features)}.

    Layer names are dotted paths like 'model.layers.0.self_attn.q_proj'.
    """
    calibration_data = get_calib_dataset(
        tokenizer=tokenizer, n_samples=n_samples, max_seq_len=max_seq_len
    )
    torch.cuda.empty_cache()
    try:
        mc = model.bfloat16().cuda()
    except Exception:
        mc = model.bfloat16()
    activations = collect_activations(mc, calibration_data, num_samples=num_collect)
    torch.cuda.empty_cache()
    return activations


def calib_acts_for(layer_idx, sublayer_path, activations):
    """Look up the activation tensor for a given (layer_idx, sublayer_path).

    sublayer_path is e.g. 'self_attn.q_proj' or 'mlp.up_proj'.
    """
    key = f"model.layers.{layer_idx}.{sublayer_path}"
    return activations.get(key)
