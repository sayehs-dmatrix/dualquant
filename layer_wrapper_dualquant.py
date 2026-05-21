"""
Dual-quant layer wrapper (cleaned).

Replaces layer_wrapper_with_mseReduction*.py. Same math, deduplicated.

The method is "dualquant" because two scales are learned jointly:
  - alpha (per-row)
  - beta  (per-col)

Inner-quant mode is selected by `dualquant_to_element_tensor` in opt_config:

    True (default)  - inside the dualquant iteration the inner quantiser is
                      element-only (int4/fp4/int8/fp8 cast, NO internal
                      block scale). alpha and beta are the only scales the
                      closed-form math sees, so the alpha/beta updates are
                      mathematically consistent. Final stored alpha is
                      rounded by the chosen scale_format.

    False (legacy)  - inside the dualquant iteration the inner quantiser is
                      the full block format via dmx.compressor (the cast
                      applies its own internal block scale on top of alpha,
                      beta). Kept for parity with the historical baseline.

The per-row scale dtype is selected explicitly by opt_config['scale_format']:
    'e8m0' | 'e4m3' | 'e4m4' | 'e5m3' | 'none'
(No per-format defaults — every config must specify scale_format.)

Reconstruction for the final stored weight always uses the block-format
cast (regardless of inner-quant mode) so the saved tensor matches what the
hardware sees.
"""


# torch_quant.py and the other helpers below are vendored under
# `legacy_vendor/` — _legacy_path adds it to sys.path.
import _legacy_path  # noqa: F401

import torch

from torch_quant import (
    convert_to_sfp,
    fake_quantize_float32_to_e4m4,
    fake_quantize_float32_to_e5m3,
    find_ebias,
    float_to_fp4,
)
from dmx.compressor import Format


# ----- format constants -----------------------------------------------------

E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
F8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max     # 448.0
F8_E5M2_MAX = torch.finfo(torch.float8_e5m2).max      # 57344.0
F4_E2M1_MAX = 6.0
HP_MBITS = 23
HP_EBIAS = 127
E8M0_BIAS = 127
EPS = 1e-12


def _quant_block_size(quant_method, config_block_size):
    """sfp4/nvfp4 are fixed at 16; everything else uses config."""
    if quant_method in ("sfp4", "nvfp4"):
        return 16
    return config_block_size


def _quant_max_val(quant_method):
    """Max representable element value (used for alpha init)."""
    if quant_method in ("sfp4", "nvfp4", "mxfp4"):
        return F4_E2M1_MAX                  # 6.0
    if quant_method == "mxfp8_e4m3":
        return F8_E4M3_MAX                  # 448.0
    if quant_method == "mxfp8_e5m2":
        return F8_E5M2_MAX                  # 57344.0
    if quant_method in ("mxint8", "rtn_int8"):
        return 127.0
    if quant_method in ("mxint4", "rtn_int4"):
        return 7.0
    return F4_E2M1_MAX                      # safe fallback


# ----- scale quantisers (one per supported scale dtype) ---------------------

def quantize_scale_e8m0(scale_fp32):
    """Round scale to nearest power-of-2 (E8M0 storage). No ebias needed."""
    s_int32 = scale_fp32.view(torch.int32)
    leading = (s_int32 >> (HP_MBITS - 1)) & 1
    e_unbiased = (torch.bitwise_right_shift(s_int32, HP_MBITS) & 0xFF) - HP_EBIAS + leading
    e_unbiased = torch.clamp(e_unbiased, min=-E8M0_BIAS, max=E8M0_BIAS + 1)
    e_biased = (e_unbiased + E8M0_BIAS).to(torch.uint8)
    e_biased = torch.where(
        torch.isnan(scale_fp32),
        torch.tensor(255, dtype=torch.uint8),
        e_biased,
    )
    out = torch.bitwise_left_shift(e_biased.to(torch.int32), HP_MBITS).view(torch.float32)
    return torch.clamp(out, min=2 ** -127)


def quantize_scale_e4m3(scale_fp32):
    """Cast to E4M3 and back. No ebias needed (uses native FP8 range)."""
    return (
        torch.clamp(scale_fp32, min=E4M3_EPS, max=F8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )


def quantize_scale_e4m4(scale_fp32, ebias):
    """E4M4 (custom 9-bit float, 4 exp + 4 mantissa). Needs per-block ebias."""
    return fake_quantize_float32_to_e4m4(mat=scale_fp32, ebias=ebias)


def quantize_scale_e5m3(scale_fp32, ebias):
    """E5M3 (custom 9-bit float, 5 exp + 3 mantissa). Needs per-block ebias."""
    return fake_quantize_float32_to_e5m3(mat=scale_fp32, ebias=ebias)


_VALID_SCALE_FORMATS = ("e8m0", "e4m3", "e4m4", "e5m3", "none")


def _quantize_alpha_inv(alpha_inv, scale_format, ebias):
    """Quantise per-row scale alpha_inv (= 1/alpha) to the requested dtype."""
    if scale_format == "e8m0":
        return quantize_scale_e8m0(alpha_inv)
    if scale_format == "e4m3":
        return quantize_scale_e4m3(alpha_inv)
    if scale_format == "e4m4":
        out = quantize_scale_e4m4(alpha_inv, ebias)
        # E4M4/E5M3 helpers return shape (..., 1) when applied to a 1-D scale;
        # squeeze the trailing dim so caller sees a (n_rows,) tensor.
        if out.dim() > alpha_inv.dim():
            out = out.squeeze(-1)
        return out
    if scale_format == "e5m3":
        out = quantize_scale_e5m3(alpha_inv, ebias)
        if out.dim() > alpha_inv.dim():
            out = out.squeeze(-1)
        return out
    if scale_format == "none":
        return alpha_inv
    raise ValueError(
        f"_quantize_alpha_inv: unknown scale_format {scale_format!r}; "
        f"expected one of {_VALID_SCALE_FORMATS}"
    )


# ----- inner element-only quantiser (dualquant_to_element_tensor=True) ------

def _quant_element_int(weight, qmax, qmin):
    """Element-only INT quant. Assumes weight is already pre-scaled by alpha,beta."""
    return weight.clone().round_().clamp_(qmin, qmax)


def _inner_quant_element(W_scaled, quant_method, block_size):
    """Element-only quant: no internal block scale."""
    if quant_method in ("nvfp4", "mxfp4"):
        return float_to_fp4(W_scaled)
    if quant_method in ("mxint4", "rtn_int4"):
        return _quant_element_int(W_scaled, qmax=7, qmin=-8)
    if quant_method in ("mxint8", "rtn_int8"):
        return _quant_element_int(W_scaled, qmax=127, qmin=-128)
    if quant_method == "mxfp8_e4m3":
        return W_scaled.to(torch.float8_e4m3fn).to(torch.float32)
    if quant_method == "mxfp8_e5m2":
        return W_scaled.to(torch.float8_e5m2).to(torch.float32)
    if quant_method == "sfp4":
        return convert_to_sfp(W_scaled)
    raise ValueError(f"_inner_quant_element: unsupported quant_method {quant_method!r}")


# ----- inner block quantiser (dualquant_to_element_tensor=False) ------------

def _inner_quant_block(W_scaled, quant_method, block_size):
    """Block-format quant via dmx.compressor — applies its own internal block scale."""
    if quant_method == "nvfp4":
        return Format.from_shorthand("NVFP4[E2M1]{16}").cast(W_scaled, -1)
    if quant_method == "mxfp4":
        return Format.from_shorthand(f"MXFP4[E2M1]{{{block_size}}}").cast(W_scaled, -1)
    if quant_method == "mxint4":
        return Format.from_shorthand(f"MXINT4{{{block_size}}}").cast(W_scaled, -1)
    if quant_method == "mxfp8_e4m3":
        return Format.from_shorthand(f"MXFP8[E4M3]{{{block_size}}}").cast(W_scaled, -1)
    if quant_method == "mxfp8_e5m2":
        return Format.from_shorthand(f"MXFP8[E5M2]{{{block_size}}}").cast(W_scaled, -1)
    if quant_method == "mxint8":
        return Format.from_shorthand(f"MXINT8{{{block_size}}}").cast(W_scaled, -1)
    if quant_method == "rtn_int8":
        qmax = 127
        s = W_scaled.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        return (W_scaled / s).round_().clamp_(-128, qmax) * s
    if quant_method == "rtn_int4":
        qmax = 7
        s = W_scaled.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        return (W_scaled / s).round_().clamp_(-8, qmax) * s
    if quant_method == "sfp4":
        return convert_to_sfp(W_scaled)
    raise ValueError(f"_inner_quant_block: unsupported quant_method {quant_method!r}")


def _inner_quant(W_scaled, quant_method, block_size, dualquant_to_element_tensor):
    fn = _inner_quant_element if dualquant_to_element_tensor else _inner_quant_block
    return fn(W_scaled, quant_method, block_size)


def _final_block_quant(W_scaled, quant_method, block_size):
    """Used at the end to produce the stored weight. Always block format."""
    return _inner_quant_block(W_scaled, quant_method, block_size)


# ----- splitting helper -----------------------------------------------------

def splitting_weights(weight_matrix, num_split, split_stat):
    split_size = weight_matrix.shape[0] // num_split
    splits_dict = {}
    for split_idx in range(num_split):
        start_idx = split_idx * split_size
        end_idx = start_idx + split_size if split_idx < num_split - 1 else weight_matrix.shape[0]
        split_matrix = weight_matrix[start_idx:end_idx]

        if split_stat == "max":
            col_div = split_matrix.abs().max(dim=0).values
        elif split_stat == "max+1":
            col_div = split_matrix.abs().max(dim=0).values + 1
        elif split_stat == "mean":
            col_div = split_matrix.abs().mean(dim=0)
        elif split_stat == "rms":
            col_div = torch.sqrt(torch.mean(split_matrix ** 2, dim=0))
        elif split_stat == "l1_norm":
            col_div = torch.norm(split_matrix, p=1, dim=0).to(split_matrix.dtype)
        elif split_stat == "l2_norm":
            col_div = torch.norm(split_matrix, p=2, dim=0).to(split_matrix.dtype)
        elif split_stat == "all_ones":
            col_div = torch.ones(split_matrix.shape[1], device=split_matrix.device, dtype=split_matrix.dtype)
        elif split_stat == "constant":
            col_div = 3 * torch.ones(split_matrix.shape[1], device=split_matrix.device, dtype=split_matrix.dtype)
        else:
            raise ValueError(f"splitting_weights: unknown split_stat {split_stat!r}")

        splits_dict[f"split_{split_idx}"] = (split_matrix / col_div, col_div)
    return splits_dict


# ----- init helpers ---------------------------------------------------------

def _init_beta(block_matrix, beta_init):
    """Per-column initial scale (forward direction: W * beta_fwd normalises cols)."""
    n_cols = block_matrix.shape[1]
    if beta_init == "l1_norm":
        beta_stat = block_matrix.abs().sum(dim=0).clamp(min=1e-8)
    elif beta_init == "l2_norm":
        beta_stat = torch.sqrt((block_matrix ** 2).sum(dim=0)).clamp(min=1e-8)
    elif beta_init == "all_one":
        beta_stat = torch.ones(n_cols, device=block_matrix.device, dtype=block_matrix.dtype)
    else:
        raise ValueError(f"_init_beta: unsupported beta_init {beta_init!r}")
    return 1.0 / beta_stat


def _init_alpha(beta_normalized, block_size, max_val, alpha_init):
    """Per-row initial scale, computed on the beta-normalised matrix."""
    n_rows = beta_normalized.shape[0]
    if alpha_init == "max_abs":
        scales = beta_normalized.clone()
        scales = scales.reshape(*scales.shape[:-1], -1, block_size)
        scales = scales.abs().max(dim=-1).values
        scales = scales / torch.tensor(max_val, dtype=torch.float32)
        if scales.dim() == 2:
            scales = scales.squeeze(-1)
        alpha_stat = scales
    elif alpha_init == "all_one":
        alpha_stat = torch.ones(n_rows, device=beta_normalized.device, dtype=beta_normalized.dtype)
    else:
        raise ValueError(f"_init_alpha: unsupported alpha_init {alpha_init!r}")
    return 1.0 / alpha_stat


# ----- dualquant iterators --------------------------------------------------
#
# Math note on alpha_init * alpha_iter:
#   We initialise alpha_fwd = 1/alpha_stat (where alpha_stat is, e.g., per-row
#   max-abs of the beta-normalised blocks). The iteration starts at this value
#   and refines it. The returned `alpha` IS the combined product
#       alpha_final_fwd = alpha_init_fwd * alpha_iter_refinement
#   because the iteration updates alpha multiplicatively from the init point.
#   Same for beta.
#
# Quantisation note:
#   The per-row scale alpha_inv = 1/alpha is rounded inside the loop using
#   the chosen scale_format ("quant-aware iteration"): each step uses an
#   alpha that is already representable in the target scale dtype. To switch
#   to "iterate in fp32, quantise once at the end" semantics, drop the
#   _quantize_alpha_inv() call inside the loop and apply it once at the end.


def dualquant_with_alpha_and_beta(block_matrix, opt_config, quant_method):
    """Returns (alpha_final_fwd, beta_final_fwd, ebias)."""
    num_iter = opt_config["num_iter"]
    block_size = opt_config["block_size"]
    alpha_init = opt_config["row_init"]
    beta_init = opt_config["col_init"]
    scale_format = opt_config["scale_format"]
    dualquant = opt_config.get("dualquant_to_element_tensor", True)

    max_val = torch.max(torch.abs(block_matrix)) / 6.0
    ebias = find_ebias(matrix=max_val)
    qmax = _quant_max_val(quant_method)

    # init: forward scales (W * beta normalises cols; alpha * (...) normalises rows)
    beta = _init_beta(block_matrix, beta_init)
    beta_normalized = block_matrix * beta.unsqueeze(0)
    alpha = _init_alpha(beta_normalized, block_size, qmax, alpha_init)

    W = block_matrix.clone()
    alpha_inv = 1.0 / alpha
    beta_inv = 1.0 / beta

    for _ in range(num_iter):
        # ---- alpha update ----
        W_scaled = alpha.unsqueeze(1) * W * beta.unsqueeze(0)
        M = _inner_quant(W_scaled, quant_method, block_size, dualquant)
        denom = torch.sum((M * beta_inv.unsqueeze(0)) ** 2, dim=1) + EPS
        alpha_inv = torch.sum(W * M * beta_inv.unsqueeze(0), dim=1) / denom
        alpha_inv = _quantize_alpha_inv(alpha_inv, scale_format, ebias)
        alpha = 1.0 / alpha_inv

        # ---- beta update ----
        W_scaled = alpha.unsqueeze(1) * W * beta.unsqueeze(0)
        M = _inner_quant(W_scaled, quant_method, block_size, dualquant)
        denom = torch.sum((M * alpha_inv.unsqueeze(1)) ** 2, dim=0) + EPS
        beta_inv = torch.sum(W * M * alpha_inv.unsqueeze(1), dim=0) / denom
        # NaN/zero guard from element mode (NVFP4 underflow can drive M->0)
        if dualquant and quant_method == "nvfp4":
            beta_inv = beta_inv.clamp(min=E4M3_EPS)
        beta = 1.0 / beta_inv

    return alpha, beta, ebias


def dualquant_with_beta_only(block_matrix, opt_config, quant_method):
    """Returns (beta_final_fwd, ebias). alpha is fixed at 1."""
    num_iter = opt_config["num_iter"]
    block_size = opt_config["block_size"]
    beta_init = opt_config["col_init"]
    dualquant = opt_config.get("dualquant_to_element_tensor", True)

    max_val = torch.max(torch.abs(block_matrix)) / 6.0
    ebias = find_ebias(matrix=max_val)

    beta = _init_beta(block_matrix, beta_init)
    W = block_matrix.clone()
    n_rows = W.shape[0]
    alpha = torch.ones(n_rows, device=W.device, dtype=W.dtype)
    alpha_inv = 1.0 / alpha
    beta_inv = 1.0 / beta

    for _ in range(num_iter):
        W_scaled = alpha.unsqueeze(1) * W * beta.unsqueeze(0)
        M = _inner_quant(W_scaled, quant_method, block_size, dualquant)
        denom = torch.sum((M * alpha_inv.unsqueeze(1)) ** 2, dim=0) + EPS
        beta_inv = torch.sum(W * M * alpha_inv.unsqueeze(1), dim=0) / denom
        if dualquant and quant_method == "nvfp4":
            beta_inv = beta_inv.clamp(min=E4M3_EPS)
        beta = 1.0 / beta_inv

    return beta, ebias


def dualquant_with_alpha_only(block_matrix, opt_config, quant_method):
    """Returns (alpha_final_fwd, ebias). beta is fixed at 1."""
    num_iter = opt_config["num_iter"]
    block_size = opt_config["block_size"]
    alpha_init = opt_config["row_init"]
    scale_format = opt_config["scale_format"]
    dualquant = opt_config.get("dualquant_to_element_tensor", True)

    max_val = torch.max(torch.abs(block_matrix)) / 6.0
    ebias = find_ebias(matrix=max_val)
    qmax = _quant_max_val(quant_method)

    W = block_matrix.clone()
    alpha = _init_alpha(W, block_size, qmax, alpha_init)

    for _ in range(num_iter):
        W_scaled = alpha.unsqueeze(1) * W
        M = _inner_quant(W_scaled, quant_method, block_size, dualquant)
        denom = torch.sum(M ** 2, dim=1) + EPS
        alpha_inv = torch.sum(W * M, dim=1) / denom
        alpha_inv = _quantize_alpha_inv(alpha_inv, scale_format, ebias)
        alpha = 1.0 / alpha_inv

    return alpha, ebias


# ----- per-block reconstruction ---------------------------------------------
#
# Final stored weight is always block-format quantised (matches what hardware
# sees) regardless of which mode the iteration ran in.


def _reconstruct_block_alpha_beta(block_matrix, alpha, beta, quant_method, block_size, scale_format, ebias):
    """W_recon_block = (1/alpha_q) * BlockQuant(alpha*W*beta), col scales applied later."""
    scaled = alpha.unsqueeze(1) * block_matrix * beta.unsqueeze(0)
    Wq = _final_block_quant(scaled, quant_method, block_size)
    alpha_inv_q = _quantize_alpha_inv(1.0 / alpha, scale_format, ebias)
    return alpha_inv_q.unsqueeze(1) * Wq


def _reconstruct_block_beta_only(block_matrix, beta, quant_method, block_size):
    scaled = block_matrix * beta.unsqueeze(0)
    return _final_block_quant(scaled, quant_method, block_size)


def _reconstruct_block_alpha_only(block_matrix, alpha, quant_method, block_size, scale_format, ebias):
    scaled = alpha.unsqueeze(1) * block_matrix
    Wq = _final_block_quant(scaled, quant_method, block_size)
    alpha_inv_q = _quantize_alpha_inv(1.0 / alpha, scale_format, ebias)
    return alpha_inv_q.unsqueeze(1) * Wq


# ----- main wrapper ---------------------------------------------------------

def wrap_dualquant_layer(layer, layer_activations, opt_config, quant_method, act_quant_flag=False):
    """Quantise a Linear layer with dualquant alpha/beta scales.

    opt_config keys consumed:
        num_splits         (int, default 1)
        scale_option       ('row_column' | 'only_column' | 'only_row')
        col_init           ('l1_norm' | 'l2_norm' | 'all_one')
        row_init           ('max_abs' | 'all_one')
        block_size         (int)
        num_iter           (int)
        scale_format       ('e8m0' | 'e4m3' | 'e4m4' | 'e5m3' | 'none')   REQUIRED
        dualquant_to_element_tensor  (bool, default True)

    Mutates layer.weight in-place. Returns the per-column scale tensor when
    act_quant_flag=True, otherwise returns None.
    """
    num_splits = opt_config["num_splits"]
    scale_option = opt_config["scale_option"]
    split_stat = opt_config["col_init"]
    scale_format = opt_config["scale_format"]
    if scale_format not in _VALID_SCALE_FORMATS:
        raise ValueError(
            f"opt_config['scale_format']={scale_format!r}; "
            f"must be one of {_VALID_SCALE_FORMATS}"
        )

    block_size = _quant_block_size(quant_method, opt_config["block_size"])
    opt_config = {**opt_config, "block_size": block_size}

    dtype = layer.weight.dtype
    matrix = layer.weight.clone().to(torch.float32).cpu()
    n_rows, n_cols = matrix.shape

    assert n_cols % block_size == 0, (
        f"n_cols ({n_cols}) must be divisible by block_size ({block_size})"
    )
    assert n_rows % num_splits == 0, (
        f"n_rows ({n_rows}) must be divisible by num_splits ({num_splits})"
    )

    num_blocks = n_cols // block_size
    quantized_split_list = []
    column_scales_list = []

    with torch.no_grad():
        if num_splits > 1:
            splits_dict = splitting_weights(matrix, num_splits, split_stat)
        else:
            ones = torch.ones(n_cols, device=matrix.device, dtype=matrix.dtype)
            splits_dict = {"split_0": (matrix, ones)}

        for _, (split_matrix, _col_div) in splits_dict.items():
            reconstructed_blocks = []
            beta_inv_list = []

            for block_idx in range(num_blocks):
                col_start = block_idx * block_size
                block_matrix = split_matrix[:, col_start:col_start + block_size]
                if scale_option == "row_column":
                    alpha, beta, ebias = dualquant_with_alpha_and_beta(
                        block_matrix, opt_config, quant_method
                    )
                    quantized_block = _reconstruct_block_alpha_beta(
                        block_matrix, alpha, beta, quant_method, block_size, scale_format, ebias
                    )

                elif scale_option == "only_column":
                    beta, _ebias = dualquant_with_beta_only(
                        block_matrix, opt_config, quant_method
                    )
                    quantized_block = _reconstruct_block_beta_only(
                        block_matrix, beta, quant_method, block_size
                    )

                elif scale_option == "only_row":
                    alpha, ebias = dualquant_with_alpha_only(
                        block_matrix, opt_config, quant_method
                    )
                    quantized_block = _reconstruct_block_alpha_only(
                        block_matrix, alpha, quant_method, block_size, scale_format, ebias
                    )
                    beta = torch.ones(block_size, device=block_matrix.device, dtype=block_matrix.dtype)

                else:
                    raise ValueError(f"unknown scale_option {scale_option!r}")

                reconstructed_blocks.append(quantized_block)
                beta_inv_list.append(1.0 / beta)

            quantized_split = torch.cat(reconstructed_blocks, dim=1)        # (rows_per_split, n_cols)
            quantized_split_list.append(quantized_split)
            column_scales_list.append(torch.cat(beta_inv_list, dim=0))      # (n_cols,)

        mat_q = torch.cat(quantized_split_list, dim=0)                      # (n_rows, n_cols)
        column_scales = torch.cat(column_scales_list, dim=0)

        if num_splits > 1:
            reconstruct_mat = [
                quantized_split_list[s] * column_scales_list[s].unsqueeze(0)
                for s in range(num_splits)
            ]
            layer.weight.copy_(torch.cat(reconstruct_mat, dim=0).to(dtype))
            return None
        if act_quant_flag:
            # Column scales returned for runtime activation-side pre-scaling.
            layer.weight.copy_(mat_q.to(dtype))
            return column_scales

        layer.weight.copy_((mat_q * column_scales.unsqueeze(0)).to(dtype))
        return None


# ----- thin compatibility shims for the existing dispatcher -----------------
# These let layer_wrapper.py keep importing the old names without changes.

def wrap_mseReduct_noSinq_Split_layer_element_quantization(
    layer, layer_activations, opt_config, quant_method, act_quant_flag=False
):
    return wrap_dualquant_layer(layer, layer_activations, opt_config, quant_method, act_quant_flag)


def wrap_mseReduct_noSinq_noSplit_layer(
    layer, layer_activations, opt_config, quant_method, act_quant_flag=False
):
    no_split = {**opt_config, "num_splits": 1}
    return wrap_dualquant_layer(layer, layer_activations, no_split, quant_method, act_quant_flag)
