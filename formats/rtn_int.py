"""Plain RTN INT4 / INT8 (no MX block scale; per-row scale only).

`cast` is the literature-standard symmetric int quantiser: per-row max-abs
scale, round, clamp, dequantise. `cast_element_only` does only the round +
clamp (caller-provided pre-scaling).

Pass `mse=True` to `make_rtn_int4` / `make_rtn_int8` (or equivalently use
`--weight-clip` on the CLI) to enable MSE-optimal weight clipping: a grid
search over 80 shrinkage factors that minimises the per-row L2.4 quantisation
error. Equivalent to QuaRot's `--w_clip` flag. Applied to weights only —
activations always use plain max-abs (dynamic per token, no search needed).
"""

import torch

from .base import FormatSpec


def _mse_scale(block: torch.Tensor, qmax: float, qmin: float,
               norm: float = 2.4, grid: int = 100, maxshrink: float = 0.80):
    """Per-row MSE-optimal scale via grid search (QuaRot --w_clip equivalent).

    Tries `int(maxshrink * grid)` shrinkage factors and returns the per-row
    scale that minimises the L-`norm` quantisation error.
    """
    xmax = block.abs().amax(dim=1)                           # (rows,)
    best_err   = torch.full((block.shape[0],), float("inf"), device=block.device)
    best_scale = (xmax.clamp(min=1e-8) / qmax)               # (rows,)

    for i in range(int(maxshrink * grid)):
        p       = 1.0 - i / grid
        scale_i = (p * xmax / qmax).clamp(min=1e-8)          # (rows,)
        q_int   = (block / scale_i.unsqueeze(1)).round_().clamp_(qmin, qmax)
        q_dq    = q_int * scale_i.unsqueeze(1)
        err     = (q_dq - block).abs_().pow_(norm).sum(dim=1) # (rows,)
        improved = err < best_err
        best_err[improved]   = err[improved]
        best_scale[improved] = scale_i[improved]

    return best_scale.unsqueeze(1)                            # (rows, 1)


def _rtn_cast(W, qmax, qmin, block_size, mse=False):
    orig_shape = W.shape
    if W.ndim != 2:
        W = W.reshape(-1, orig_shape[-1])
    Wq = W.clone()
    _, cols = W.shape

    # `-1` sentinel = per-row for weights / per-token for activations: one
    # block spans the full input dim. Mimics QuaRot's --w_groupsize -1 and
    # the per-token activation path (resolved dynamically per linear since
    # `cols` differs across q/k/v/o (4096) and down_proj (11008) in Llama).
    if block_size == -1:
        block_size = cols

    for start in range(0, cols, block_size):
        end   = min(start + block_size, cols)
        block = W[:, start:end]
        # one scale per row per block — MSE-optimal or plain max-abs
        if mse:
            s = _mse_scale(block, qmax, qmin)
        else:
            xmax = block.abs().amax(dim=1, keepdim=True)
            s = xmax / qmax
            s[xmax == 0] = 1.0   # zero rows → identity (matches QuaRot zero guard)
        Wq[:, start:end] = (block / s).round_().clamp_(qmin, qmax) * s

    return Wq.reshape(orig_shape)


def _rtn_cast_element_only(W, qmax, qmin):
    return W.clone().round_().clamp_(qmin, qmax)


def make_rtn_int4(block_size=128, mse=False) -> FormatSpec:
    return FormatSpec(
        name="rtn_int4",
        block_size=block_size,
        max_val=7.0,
        cast=lambda W: _rtn_cast(W, 7, -8, block_size, mse=mse),
        cast_element_only=lambda W: _rtn_cast_element_only(W, 7, -8),
    )


def make_rtn_int8(block_size=128, mse=False) -> FormatSpec:
    return FormatSpec(
        name="rtn_int8",
        block_size=block_size,
        max_val=127.0,
        cast=lambda W: _rtn_cast(W, 127, -128, block_size, mse=mse),
        cast_element_only=lambda W: _rtn_cast_element_only(W, 127, -128),
    )


def _rtn_cast_asym(W, block_size):
    """Asymmetric per-group INT4: scale=(max-min)/15, zero-point shifts range to [0,15]."""
    orig_shape = W.shape
    if W.ndim != 2:
        W = W.reshape(-1, orig_shape[-1])
    Wq = W.clone()
    _, cols = W.shape
    if block_size == -1:
        block_size = cols
    for start in range(0, cols, block_size):
        end   = min(start + block_size, cols)
        block = W[:, start:end]
        w_min = block.amin(dim=1, keepdim=True)
        w_max = block.amax(dim=1, keepdim=True)
        scale = (w_max - w_min).clamp(min=1e-8) / 15.0
        zero  = (-w_min / scale).round_().clamp_(0, 15)
        q     = (block / scale + zero).round_().clamp_(0, 15)
        Wq[:, start:end] = (q - zero) * scale
    return Wq.reshape(orig_shape)


def make_rtn_int4_asym(block_size=128) -> FormatSpec:
    return FormatSpec(
        name="rtn_int4_asym",
        block_size=block_size,
        max_val=7.0,
        cast=lambda W: _rtn_cast_asym(W, block_size),
        cast_element_only=lambda W: _rtn_cast_element_only(W, 7, -8),
    )
