"""MX-spec formats (block_size configurable; default 32):
   mxfp4 (E2M1), mxint4, mxfp8_e4m3, mxfp8_e5m2, mxint8.

`cast` uses dmx.compressor's Format.cast — applies an internal E8M0 block
scale + element quant.
`cast_element_only` does the element quant only (no internal block scale);
the caller is responsible for any per-row/per-col scaling.
"""

import _legacy_path  # noqa: F401

import torch
from dmx.compressor import Format
from torch_quant import float_to_fp4

from .base import FormatSpec


def _int_element_only(W, qmax, qmin):
    return W.clone().round_().clamp_(qmin, qmax)


def make_mxfp4(block_size=32) -> FormatSpec:
    fmt = Format.from_shorthand(f"MXFP4[E2M1]{{{block_size}}}")
    return FormatSpec(
        name="mxfp4",
        block_size=block_size,
        max_val=6.0,
        cast=lambda W: fmt.cast(W, -1),
        cast_element_only=lambda W: float_to_fp4(W),
    )


def make_mxint4(block_size=32) -> FormatSpec:
    fmt = Format.from_shorthand(f"MXINT4{{{block_size}}}")
    return FormatSpec(
        name="mxint4",
        block_size=block_size,
        max_val=7.0,
        cast=lambda W: fmt.cast(W, -1),
        cast_element_only=lambda W: _int_element_only(W, 7, -8),
    )


def make_mxfp8_e4m3(block_size=32) -> FormatSpec:
    fmt = Format.from_shorthand(f"MXFP8[E4M3]{{{block_size}}}")
    return FormatSpec(
        name="mxfp8_e4m3",
        block_size=block_size,
        max_val=torch.finfo(torch.float8_e4m3fn).max,    # 448.0
        cast=lambda W: fmt.cast(W, -1),
        cast_element_only=lambda W: W.to(torch.float8_e4m3fn).to(torch.float32),
    )


def make_mxfp8_e5m2(block_size=32) -> FormatSpec:
    fmt = Format.from_shorthand(f"MXFP8[E5M2]{{{block_size}}}")
    return FormatSpec(
        name="mxfp8_e5m2",
        block_size=block_size,
        max_val=torch.finfo(torch.float8_e5m2).max,      # 57344.0
        cast=lambda W: fmt.cast(W, -1),
        cast_element_only=lambda W: W.to(torch.float8_e5m2).to(torch.float32),
    )


def make_mxint8(block_size=32) -> FormatSpec:
    fmt = Format.from_shorthand(f"MXINT8{{{block_size}}}")
    return FormatSpec(
        name="mxint8",
        block_size=block_size,
        max_val=127.0,
        cast=lambda W: fmt.cast(W, -1),
        cast_element_only=lambda W: _int_element_only(W, 127, -128),
    )
