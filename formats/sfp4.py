"""SFP4 (block_size = 16; SFP-style 4-bit float)."""

import _legacy_path  # noqa: F401

from torch_quant import convert_to_sfp

from .base import FormatSpec


def make_sfp4() -> FormatSpec:
    return FormatSpec(
        name="sfp4",
        block_size=16,
        max_val=6.0,
        cast=lambda W: convert_to_sfp(W),
        cast_element_only=lambda W: convert_to_sfp(W),
    )
