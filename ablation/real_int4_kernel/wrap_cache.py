"""Disk cache for DualQuant's method.wrap() output (mat_q + beta).

mat_q/beta are a pure function of (model weights, method, block_size,
row/col init, num_iter) -- entirely independent of which GEMM kernel later
consumes them. The BCD optimization to produce them takes ~35-40 minutes for
a 32-layer 8B model; caching it means switching kernel backends (Triton,
nunchaku, ...) or re-running after a bugfix costs seconds, not minutes.
"""
import hashlib
import os

import torch

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrap_cache")


def _cache_path(model_id, layer_key, block_size):
    safe_model = model_id.replace("/", "__")
    h = hashlib.sha1(f"{model_id}|{layer_key}|{block_size}".encode()).hexdigest()[:8]
    return os.path.join(_CACHE_DIR, f"{safe_model}__{layer_key}__bs{block_size}__{h}.pt")


def wrap_cached(method, module, cfg, layer_key, model_id, block_size):
    """Same contract as method.wrap(module, cfg, layer_key=...): mutates
    module.weight in place to mat_q, returns beta. Uses a disk cache keyed
    on (model_id, layer_key, block_size) so repeated runs across different
    kernel backends skip the BCD optimization entirely."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(model_id, layer_key, block_size)
    if os.path.exists(path):
        cached = torch.load(path, map_location="cpu")
        module.weight.data.copy_(cached["mat_q"].to(module.weight.dtype).to(module.weight.device))
        return cached["beta"].to(module.weight.device)

    beta = method.wrap(module, cfg, layer_key=layer_key)
    torch.save(
        {"mat_q": module.weight.detach().cpu().clone(), "beta": beta.detach().cpu().clone()},
        path,
    )
    return beta
