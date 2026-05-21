"""Plain RTN INT4 / INT8 (no MX block scale; per-row scale only).

`cast` is the literature-standard symmetric int quantiser: per-row max-abs
scale, round, clamp, dequantise. `cast_element_only` does only the round +
clamp (caller-provided pre-scaling).
"""

from .base import FormatSpec


# def _rtn_cast(W, qmax, qmin):
#     s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
#     return (W / s).round_().clamp_(qmin, qmax) * s

def _rtn_cast(W, qmax, qmin, block_size):
    orig_shape = W.shape
    if W.ndim != 2:
        W = W.reshape(-1, orig_shape[-1])
    Wq = W.clone()
    _, cols = W.shape

    for start in range(0, cols, block_size):
        end = min(start + block_size, cols)
        block = W[:, start:end]
        # one scale per row per block
        s = (block.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax)
        Wq[:, start:end] = (block / s).round_().clamp_(qmin, qmax) * s

    return Wq.reshape(orig_shape)



def _rtn_cast_element_only(W, qmax, qmin):
    return W.clone().round_().clamp_(qmin, qmax)


def make_rtn_int4(block_size=128) -> FormatSpec:
    return FormatSpec(
        name="rtn_int4",
        block_size=block_size,    # used for column chunking only
        max_val=7.0,
        cast=lambda W: _rtn_cast(W, 7, -8, block_size),
        cast_element_only=lambda W: _rtn_cast_element_only(W, 7, -8),
    )


def make_rtn_int8(block_size=128) -> FormatSpec:
    return FormatSpec(
        name="rtn_int8",
        block_size=block_size,
        max_val=127.0,
        cast=lambda W: _rtn_cast(W, 127, -128, block_size),
        cast_element_only=lambda W: _rtn_cast_element_only(W, 127, -128),
    )
