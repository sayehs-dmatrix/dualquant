"""Set the online-Hadamard flags on PermLinear instances after QuaRot.

The logic is the same as spcl/QuaRot's `main.py:33-41` (Apache-2.0): walk
the qlayers, set `online_full_had` on every `down_proj` and `online_partial_had`
on every `o_proj`, pulling `had_K` / `K` / `had_dim` from `get_hadK(...)`.

Call this AFTER:
  1. `apply_quarot(model, ...)` has done the offline R1/R2/R4 weight folds.
  2. `install_perm_linear(...)` has wrapped every Linear with PermLinear.

Without this call, the offline R2 and R4 folds are uncancelled and the
model's forward output drifts from the unrotated baseline.
"""

# Trigger sys.modules stubs + sys.path setup before importing hadamard_utils.
from . import vendor  # noqa: F401

import hadamard_utils  # the vendored one — resolves via quarot/__init__ sys.path


def install_online_hadamards(model, fp32_had: bool = False, verbose: bool = False) -> None:
    """Iterate model named-modules; on every PermLinear wrapping a down_proj
    or o_proj, set the QuaRot online-Hadamard flags."""
    # Imported lazily to avoid a circular import: perm_linear.py imports the
    # `quarot` package at module level (to install sys.modules stubs), so this
    # file gets loaded while perm_linear is still initialising. By the time
    # `install_online_hadamards` is actually CALLED (post-init), the symbol
    # is available.
    from activations.perm_linear import PermLinear
    cfg = model.config
    intermediate_size = int(cfg.intermediate_size)
    num_heads = int(cfg.num_attention_heads)
    # head_dim is not always hidden_size / num_heads. Qwen3, Llama-4, and
    # Gemma-3 set head_dim explicitly on the config — e.g. Qwen3-0.6B has
    # hidden_size=1024, num_heads=16, head_dim=128. The online half must use
    # the same head_dim the offline rotation used in apply_r1 (or H @ H
    # won't cancel and the model output is garbage).
    head_dim = int(getattr(cfg, "head_dim", None) or (int(cfg.hidden_size) // num_heads))

    n_full, n_partial = 0, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, PermLinear):
            continue
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(intermediate_size)
            mod.online_full_had = True
            mod.had_K = had_K
            mod.K = K
            mod.fp32_had = fp32_had
            n_full += 1
        elif "o_proj" in name:
            had_K, K = hadamard_utils.get_hadK(num_heads)
            mod.online_partial_had = True
            mod.had_K = had_K
            mod.K = K
            mod.had_dim = head_dim
            mod.fp32_had = fp32_had
            n_partial += 1

    if verbose:
        print(f"[quarot] installed online Hadamards: "
              f"{n_full} down_proj (full), {n_partial} o_proj (partial)")
