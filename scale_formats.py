"""Quantisers for per-row / per-block scale tensors.

Selected via `scale_format` config field — required for any method that
stores scales (dualquant, gptq w/ groupwise scale, etc.).

Each function takes a fp32 scale tensor and returns a fp32 tensor that has
been rounded to the target scale dtype.
"""

import _legacy_path  # noqa: F401

import torch
from torch_quant import fake_quantize_float32_to_e4m4, fake_quantize_float32_to_e5m3


_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
_F8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
_HP_MBITS = 23
_HP_EBIAS = 127
_E8M0_BIAS = 127


def quantize_scale_e8m0(scale_fp32: torch.Tensor, ebias=None) -> torch.Tensor:
    """Round to nearest power-of-2 (E8M0 storage). ebias unused, accepted for uniform call sig."""
    s_int32 = scale_fp32.view(torch.int32)
    leading = (s_int32 >> (_HP_MBITS - 1)) & 1
    e_unbiased = (torch.bitwise_right_shift(s_int32, _HP_MBITS) & 0xFF) - _HP_EBIAS + leading
    e_unbiased = torch.clamp(e_unbiased, min=-_E8M0_BIAS, max=_E8M0_BIAS + 1)
    e_biased = (e_unbiased + _E8M0_BIAS).to(torch.uint8)
    e_biased = torch.where(
        torch.isnan(scale_fp32),
        torch.tensor(255, dtype=torch.uint8),
        e_biased,
    )
    out = torch.bitwise_left_shift(e_biased.to(torch.int32), _HP_MBITS).view(torch.float32)
    return torch.clamp(out, min=2 ** -127)


def quantize_scale_e4m3(scale_fp32: torch.Tensor, ebias=None) -> torch.Tensor:
    """Cast to native FP8 E4M3 and back. ebias unused."""
    return (
        torch.clamp(scale_fp32, min=_E4M3_EPS, max=_F8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )


def quantize_scale_e4m4(scale_fp32: torch.Tensor, ebias) -> torch.Tensor:
    """Custom 9-bit float (4 exp + 4 mantissa). Needs ebias."""
    if ebias is None:
        raise ValueError("e4m4 scale quantiser requires `ebias`")
    out = fake_quantize_float32_to_e4m4(mat=scale_fp32, ebias=ebias)
    if out.dim() > scale_fp32.dim():
        out = out.squeeze(-1)
    return out


def quantize_scale_e5m3(scale_fp32: torch.Tensor, ebias) -> torch.Tensor:
    """Custom 9-bit float (5 exp + 3 mantissa). Needs ebias."""
    if ebias is None:
        raise ValueError("e5m3 scale quantiser requires `ebias`")
    out = fake_quantize_float32_to_e5m3(mat=scale_fp32, ebias=ebias)
    if out.dim() > scale_fp32.dim():
        out = out.squeeze(-1)
    return out


def quantize_scale_none(scale_fp32: torch.Tensor, ebias=None) -> torch.Tensor:
    """No scale rounding (RTN, fp32 scale storage)."""
    return scale_fp32


_REGISTRY = {
    "e8m0": quantize_scale_e8m0,
    "e4m3": quantize_scale_e4m3,
    "e4m4": quantize_scale_e4m4,
    "e5m3": quantize_scale_e5m3,
    "none": quantize_scale_none,
}


def quantize_scale(scale_fp32: torch.Tensor, scale_format: str, ebias=None) -> torch.Tensor:
    """Dispatch to the appropriate scale quantiser by name."""
    if scale_format not in _REGISTRY:
        raise ValueError(
            f"unknown scale_format {scale_format!r}; expected one of {list(_REGISTRY)}"
        )
    return _REGISTRY[scale_format](scale_fp32, ebias)
