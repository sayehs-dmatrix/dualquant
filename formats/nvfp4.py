"""NVFP4 (block_size = 16, fixed by spec; per-block scale is E4M3)."""

import _legacy_path  # noqa: F401

from blockfmt import Format
from torch_quant import float_to_fp4

from .base import FormatSpec


def make_nvfp4() -> FormatSpec:
    fmt = Format.from_shorthand("NVFP4[E2M1]{16}")
    return FormatSpec(
        name="nvfp4",
        block_size=16,
        max_val=6.0,
        cast=lambda W: fmt.cast(W, -1),
        cast_element_only=lambda W: float_to_fp4(W),
    )
