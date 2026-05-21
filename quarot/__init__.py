"""QuaRot package — R1 + R2 + R4 milestone (R3 skipped: no KV-cache quant).

We vendor four files from spcl/QuaRot (Apache-2.0) byte-for-byte under
`vendor/`:
    rotation_utils.py   — rotation + LN-fusion math (R1, R2, R3, R4)
    model_utils.py      — model-shape helpers; includes the RMSN class
    utils.py            — small utility module (DEV, cleanup_memory, ...)
    hadamard_utils.py   — Hadamard matrices + apply_exact_had_to_linear

The reference's `rotation_utils.py` imports `fast_hadamard_transform`,
`hadamard_utils`, and `quant_utils` at the top of the file. The first is a
Tri Dao CUDA extension and not installable in this env (no nvcc); the third
is only consumed by R3 (online-rotation) code we don't run. So before
loading the vendored files this package:

  - registers a `fast_hadamard_transform` module in sys.modules whose
    `hadamard_transform(x, scale)` is a pure-torch Walsh-Hadamard butterfly
    on the last axis. Equivalent to the Tri Dao kernel up to fp round-off.
    Slower but correct and CPU/GPU-portable.
  - registers an inert `quant_utils.ActQuantizer` stub (only QKRotationWrapper
    consumes it; R3 is not enabled here).
  - adds `vendor/` to `sys.path` so the bare `import model_utils` / `import
    utils` / `import hadamard_utils` inside the vendored files resolve to
    our byte-identical copies.

R1 + R2 + R4 are all reachable through the vendored functions; R3 is
deliberately not invoked (no KV-cache quantisation in this codebase).
"""

import os
import sys
import types

import torch

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def _hadamard_transform_torch(x: torch.Tensor, scale=1.0) -> torch.Tensor:
    """Walsh-Hadamard transform along the last axis (Sylvester convention).

    Pure-torch equivalent of `fast_hadamard_transform.hadamard_transform`.
    Requires `x.shape[-1]` to be a power of 2. Returns `scale · H_n · x`
    treating x's last axis as the column vector.
    """
    n = x.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"last dim must be a positive power of 2; got {n}")
    h = 1
    while h < n:
        view_shape = x.shape[:-1] + (n // (2 * h), 2, h)
        x = x.reshape(view_shape)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        new = torch.empty_like(x)
        new[..., 0, :] = a + b
        new[..., 1, :] = a - b
        x = new.reshape(x.shape[:-3] + (n,))
        h *= 2
    if isinstance(scale, torch.Tensor):
        x = x * scale.to(x.device).to(x.dtype)
    elif scale != 1.0:
        x = x * scale
    return x


def _install_stubs():
    # ── fast_hadamard_transform ────────────────────────────────────────────
    # Provides the real Walsh-Hadamard. Used by:
    #   * hadamard_utils.matmul_hadU_cuda             (R4 path)
    #   * hadamard_utils.apply_exact_had_to_linear     (R2 path, output=True)
    if "fast_hadamard_transform" not in sys.modules:
        fht = types.ModuleType("fast_hadamard_transform")
        fht.hadamard_transform = _hadamard_transform_torch
        sys.modules["fast_hadamard_transform"] = fht

    # ── quant_utils ────────────────────────────────────────────────────────
    # Only QKRotationWrapper (R3) reads ActQuantizer; we don't run R3.
    if "quant_utils" not in sys.modules:
        qu = types.ModuleType("quant_utils")

        class _ActQuantizerStub:
            def configure(self, **kw): pass
            def find_params(self, x): pass
            def __call__(self, x): return x
            def free(self): pass

        qu.ActQuantizer = _ActQuantizerStub
        sys.modules["quant_utils"] = qu


_install_stubs()

# Make `import model_utils` / `import utils` / `import hadamard_utils` (bare
# imports inside the vendored files) resolve to OUR vendor copies.
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)


from .llama_quarot import apply_r1  # noqa: E402
from .online_had_install import install_online_hadamards  # noqa: E402

__all__ = ["apply_r1", "install_online_hadamards"]
