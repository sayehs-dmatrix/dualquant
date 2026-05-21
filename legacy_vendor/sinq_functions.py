import torch
from sinkhorn import *
from awq import *
from torch_quant import convert_to_sfp, convert_to_sbfp12, convert_float32_mat_to_scales_fp4_ebias, convert_float32_mat_to_scales_int4_ebias, fake_quantize_float32_to_e4m4, find_ebias

E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
E8M0_EXPONENT_BIAS = 127
F8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

NF4_CODEBOOK = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
     0.0,
     0.07958029955625534,
     0.16093020141124725,
     0.24611230194568634,
     0.33791524171829224,
     0.44070982933044434,
     0.5626170039176941,
     0.7229568362236023,
     1.0
], dtype=torch.float32)

def quantize_rtn(
    matrix: torch.Tensor,
    min_max=[],
    niter=None,
    mode: str = "uniform",        # "uniform" | "nf4" | "nf3"
    nf_use_shift: bool = True,    # applies to both NF4 and NF3
    nf_shift_kind: str = "mean"   # "minmax" | "mean"
):
    w = matrix
    orig_dtype = w.dtype
    dev = w.device
    w = w.to(torch.float32)

    # breakpoint()

    if mode.lower() in ("nf4", "nf3"):
        cb = NF4_CODEBOOK 

        if not nf_use_shift:
            scales = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
            zeros  = torch.zeros_like(scales)
            norm   = w / scales
        else:
            if nf_shift_kind == "minmax":
                w_max = w.amax(dim=1, keepdim=True)
                w_min = w.amin(dim=1, keepdim=True)
                denom  = (w_max - w_min).clamp_min(1e-4)
                scales = denom / 2.0
                zeros  = - (w_max + w_min) / denom
                norm   = w / scales + zeros
            elif nf_shift_kind == "mean":
                mu     = w.mean(dim=1, keepdim=True)
                x      = w - mu
                scales = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
                zeros  = (-mu / scales)
                norm   = w / scales + zeros
            else:
                raise ValueError("nf_shift_kind must be 'minmax' or 'mean'")

        cb = cb.to(dev, dtype=norm.dtype)
        q  = (norm.unsqueeze(-1) - cb.view(1,1,-1)).abs().argmin(dim=-1).to(torch.int8)
        return q.contiguous(), scales.to(orig_dtype), zeros.to(orig_dtype), orig_dtype

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = min_max[1]
    min_int = min_max[0]
    scales  = (max_val - min_val).clamp(min=1e-4) / max_int
    zeros   = -torch.round(min_val / scales)
    q       = torch.clamp(torch.round(w / scales + zeros), min_int, max_int).to(torch.int8)
    return q.contiguous(), scales.to(orig_dtype), zeros.to(orig_dtype), orig_dtype


def dq8(x):
    if x is None:
        return None
    # Dict form (from saved models)
    if isinstance(x, dict):
        xt = x["x"]; s = x["s"]; m = x["m"]; shape = x.get("shape")
        # Guard: catch accidental stringified tensors
        bad = [t for t in (xt, s, m) if isinstance(t, str)]
        if bad:
            raise ValueError("quantAux meta was serialized incorrectly: found strings in meta['scale'/'zero']. "
                             "Ensure save_weights_safetensors() recursively extracts tensors in meta.")
        if isinstance(shape, list):
            shape = tuple(shape)
        return (xt * s + m).view(shape)

    # Legacy tuple form (live before save)
    xt, s, m, shape = x
    return (xt * s + m).view(shape)


def quantize_dual_scale_shift(matrix, layer_activations, min_max, method, awq_scale=None):
    dtype = matrix.dtype
    dev = matrix.device
    matrix = matrix.float()

    tile = 1
    min_max = [0,15]
    shape = matrix.shape


    matrix, mu1, mu2 = sinkhorn_log(matrix, 16)

    if layer_activations is not None:
        matrix = matrix * awq_scale
        mu1 = mu1 / awq_scale.float()


    # breakpoint()
    if "nf4" in method.lower():
        # breakpoint()
        q, scales, z, _ = quantize_rtn(matrix, min_max, mode="nf4")
    elif "nf3" in method.lower():
        q, scales, z, _ = quantize_rtn(matrix, min_max, mode="nf3")
    #######################
    elif "sfp4" in method.lower():
        # print('sfp4 quantization')
        scales, q, _ = convert_float32_mat_to_scales_fp4_ebias(matrix, ebias=None)
        q = q[:,0, :]
        z = torch.zeros(scales.shape)
        #######################
    elif "sbfp12" in method.lower():
        # breakpoint()
        scales, q, _ = convert_float32_mat_to_scales_int4_ebias(matrix, ebias=None)
        q = q[:,0, :]
        z = torch.zeros(scales.shape)
    ##### Newly added #####
    elif "activation_no_quant" in method.lower():
        # int4_mat = matrix.reshape(*scales.shape[:-1], -1, 16)
        q = torch.clone(matrix)
        scales = torch.clone(matrix)
        scales = scales.reshape(*scales.shape[:-1], -1, 16) 
        scales = torch.abs(scales)
        scales = torch.max(scales, dim=-1)[0] 
        scales /= torch.tensor(7.0).to(torch.float32)
        scales = scales.reshape(*matrix.shape[:-1], -1)
        scales = torch.ones(scales.shape)
        z = torch.zeros(scales.shape)
    else:
        q, scales, z, _ = quantize_rtn(matrix, min_max, mode="uniform")
                
    scales_columns = torch.ones(1,matrix.shape[1]).to(dev).to(mu1.dtype) * mu1
    scales = scales*mu2

    q = q.to(dtype).to(dev)
    scales_W_rows = (scales).to(dtype) 
    scales_W_columns = (scales_columns).to(dtype) 
    z = (z).to(dtype).to(dev)

    return q, scales_W_rows.to(dev), scales_W_columns.to(dev), z


####### No SINKhorn ################
def quantize_dual_scale_shift_no_Sinkhorn(matrix, min_max, method, awq_scale=None):
    dtype = matrix.dtype
    dev = matrix.device
    matrix = matrix.float()

    # normalize the matrix with sinkhorn inspired std.dev. scaling
    # matrix, mu1, mu2 = min_kurt_vectors_vmap(matrix,32) # for kurtosis exp.
    # matrix, mu1, mu2 = sinkhorn_log(matrix, 16)

    # if not ('sinq' in method):
    #     matrix = matrix * mu1 * mu2
    #     mu1 = torch.ones_like(mu1)
    #     mu2 = torch.ones_like(mu2)

    # if 'awq' in method:
    #     matrix = matrix * awq_scale
    #     mu1 = mu1 / awq_scale.float()

    # breakpoint()
    # print('In RTN Function!')

    if "nf4" in method.lower():
        # breakpoint()
        q, scales, z, _ = quantize_rtn(matrix, min_max, mode="nf4")
    elif "nf3" in method.lower():
        q, scales, z, _ = quantize_rtn(matrix, min_max, mode="nf3")
    #######################
    elif "sfp4" in method.lower():
        # print('sfp4 quantization')
        scales, q, _ = convert_float32_mat_to_scales_fp4_ebias(matrix, ebias=None)
        q = q[:,0, :]
        z = torch.zeros(scales.shape)
        #######################
    elif "sbfp12" in method.lower():
        scales, q, _ = convert_float32_mat_to_scales_int4_ebias(matrix, ebias=None)
        q = q[:,0, :]
        z = torch.zeros(scales.shape)
    else:
        q, scales, z, _ = quantize_rtn(matrix, min_max, mode="uniform")
                
    scales2 = torch.ones(1,matrix.shape[1]).to(dev).to(matrix.dtype) 
    scales = scales

    q = q.to(dtype).to(dev)
    s1 = (scales).to(dtype) 
    s2 = (scales2).to(dtype) 
    z = (z).to(dtype).to(dev)

    return q, s1.to(dev), s2.to(dev), z
#######################################################
 
def dequantize(W_q, meta: dict, use_unpack_kernel = False):
        compute_dtype = meta.get("compute_dtype", torch.float16)

        W_r = W_q.to(compute_dtype)

        # 2) Load scales/zeros
        method = meta.get("method", "").lower()
        # breakpoint()
        if "quantaux" in method:
            s = dq8(meta["scale"])
            z = dq8(meta["zero"])
        else:
            s = meta["scale"]
            z = meta["zero"]
        s2 = meta.get("scale2", None)

        # Make sure they're on W_r.device
        dev = W_r.device
        import torch as _torch
        if isinstance(s, _torch.Tensor):  s = s.to(dev)
        if isinstance(z, _torch.Tensor):  z = z.to(dev)
        if isinstance(s2, _torch.Tensor): s2 = s2.to(dev)

        # 3) NFx (NF3 or NF4)
        if ("nf4" in method) or ("nf3" in method):
            is_nf3 = ("nf3" in method)
            cb = (NF4_CODEBOOK).to(W_r.device, dtype=s.dtype)
            max_code = cb.numel() - 1

            if len(s.shape) == 2:
                rows = s.shape[0]
                idx  = W_r[:rows].to(torch.int64).clamp_(0, max_code)   # 0..7 or 0..15
                vals = cb[idx]                                          # [-1,1] levels
                out  = ((vals - z.to(cb.dtype)) * s.to(cb.dtype)).reshape(meta["shape"])
                if s2 is not None:
                    out = out * s2
                W_r = out

            elif len(s.shape) == 3:
                # Blocked/tiling case
                H, W = meta["shape"]
                block = W_r.shape[-1]
                n_h, n_w = H // block, W // block
                total = n_h * block * n_w * block

                idx  = W_r.view(-1)[:total].to(torch.int64).clamp_(0, max_code)
                vals = cb[idx].view(n_h, block, n_w, block)

                # Broadcast s/z/s2 like your uniform path
                try:
                    z_ = z.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3).to(cb.dtype)
                except Exception:
                    z_ = z.to(cb.dtype)
                s1 = s.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3).to(cb.dtype)
                s2_ = (s2.reshape(n_h, n_w, 1, block).permute(0, 2, 1, 3)
                    if s2 is not None else torch.ones_like(s1))

                W_r = ((vals - z_) * s1 * s2_).view(H, W)
            else:
                raise ValueError("invalid scale shape for NF dequant")

            if torch.any(torch.isnan(W_r)):
                raise RuntimeError("NaN detected in NF dequantized weights")
            return W_r

        # 4) Uniform / other paths (unchanged, but robust s2 handling)
        if len(s.shape) == 2:
            s2_eff = 1 if s2 is None else s2
            W_r = W_r[: s.shape[0]]
            W_r = ((W_r - z) * s).reshape(meta["shape"]) * s2_eff

        elif len(s.shape) == 3:
            H, W = meta["shape"]
            block = W_r.shape[-1]
            n_h, n_w = H // block, W // block
            total = n_h * block * n_w * block

            W_r = W_r.view(-1)[: total].view(n_h, block, n_w, block)

            try:
                z_ = z.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3)
            except Exception:
                z_ = z
            s1 = s.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3)
            s2_ = (s2.reshape(n_h, n_w, 1, block).permute(0, 2, 1, 3)
                if s2 is not None else torch.ones_like(s1))

            W_r = ((W_r - z_) * s1 * s2_).view(H, W)
        else:
            raise ValueError("invalid scale shape for dequant")

        if torch.any(torch.isnan(W_r)):
            raise RuntimeError("NaN detected in dequantized weights")

        return W_r.to(compute_dtype)


##################################################################################
def partial_dequantize_extract_col_scales(W_q, meta: dict, use_unpack_kernel = False):
        compute_dtype = meta.get("compute_dtype", torch.float16)
        W_r = W_q.to(compute_dtype)

        # 2) Load scales/zeros
        method = meta.get("method", "").lower()
        s = meta["scale"]
        z = meta["zero"]
        s2 = meta.get("scale2", None)

        # Make sure they're on W_r.device
        dev = W_r.device
        import torch as _torch
        if isinstance(s, _torch.Tensor):  s = s.to(dev)
        if isinstance(z, _torch.Tensor):  z = z.to(dev)
        if isinstance(s2, _torch.Tensor): s2 = s2.to(dev)

        if len(s.shape) == 2:
            s2_eff = 1 if s2 is None else s2
            W_r = W_r[: s.shape[0]]
            W_with_row_scales = ((W_r - z) * s).reshape(meta["shape"]) 
            act_scales = s2_eff

        elif len(s.shape) == 3:
            H, W = meta["shape"]
            block = W_r.shape[-1]
            n_h, n_w = H // block, W // block
            total = n_h * block * n_w * block

            W_r = W_r.view(-1)[: total].view(n_h, block, n_w, block)

            try:
                z_ = z.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3)
            except Exception:
                z_ = z
            s1 = s.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3)
            s2_ = (s2.reshape(n_h, n_w, 1, block).permute(0, 2, 1, 3)
                if s2 is not None else torch.ones_like(s1))

            # W_r = ((W_r - z_) * s1 * s2_).view(H, W)
            W_with_row_scales = ((W_r - z_) * s1).view(H, W)
            act_scales = s2_
            
        else:
            raise ValueError("invalid scale shape for dequant")

        if torch.any(torch.isnan(W_with_row_scales)):
            raise RuntimeError("NaN detected in dequantized weights")

        return W_with_row_scales.to(compute_dtype), act_scales


def _quantize_row_scale(mu_row, quant_method):
    """Quantize Sinkhorn row scales to the hardware scale format of quant_method.

    Sinkhorn produces normalized scales (max_val ~ 1), so ebias is derived from 1.0.
    Mirrors what MSE reduction does to block_scales so both methods are comparable.
    """
    hp_mbits = 23
    hp_exp_bias = 127
    mu = mu_row.float().contiguous()

    if quant_method == 'sfp4':
        ebias = find_ebias(matrix=torch.tensor(1.0))
        return fake_quantize_float32_to_e4m4(mat=mu, ebias=ebias).reshape(mu_row.shape)

    elif quant_method == 'nvfp4':
        q = torch.clamp(mu, min=E4M3_EPS, max=F8_E4M3_MAX).to(torch.float8_e4m3fn)
        return q.to(torch.float32).reshape(mu_row.shape)

    elif quant_method in ('mxfp4', 'mxint4', 'mxfp8_e4m3', 'mxfp8_e5m2', 'mxint8'):
        scale_int32 = mu.view(torch.int32)
        leading_mantissa_bit = (scale_int32 >> (hp_mbits - 1)) & 1
        extracted_pow2_round = (
            (torch.bitwise_right_shift(scale_int32, hp_mbits) & 0b11111111)
            - hp_exp_bias + leading_mantissa_bit
        )
        scale_e8m0_unbiased = torch.clamp(
            extracted_pow2_round,
            min=-E8M0_EXPONENT_BIAS,
            max=E8M0_EXPONENT_BIAS + 1
        )
        scale_e8m0_biased = (scale_e8m0_unbiased + E8M0_EXPONENT_BIAS).to(torch.uint8)
        scale_e8m0_biased = torch.where(
            torch.isnan(mu),
            torch.tensor(255, dtype=torch.uint8),
            scale_e8m0_biased
        )
        scale_fp32 = (
            torch.bitwise_left_shift(scale_e8m0_biased.to(torch.int32), hp_mbits)
        ).view(torch.float32)
        return torch.clamp(scale_fp32, min=2**-127).reshape(mu_row.shape)

    else:  # rtn_int4, rtn_int8 — no dedicated scale format
        return mu_row


######################## Wrap with SINQ / ASINQ Method #########################
def wrap_layer_with_sinq_method(weight, layer_activations, quant_method, block_size=128):
    """
    SINQ  (layer_activations is None):
        Per block b of W:
            W_bal, mu_col, mu_row = sinkhorn_log(W[:, b])
            W_q[:, b]  = pure_qunatize(W_bal) * mu_row   (row scales absorbed)
            col_scales[b] = mu_col                        (returned for activation use)

    ASINQ (layer_activations provided):
        AWQ scale s (in_features,) is computed once on the original W.
        Per block b:
            W_bal, mu_col, mu_row = sinkhorn_log(W[:, b])   # Sinkhorn first
            awq_b = s[b]
            W_bal  = W_bal * awq_b                           # AWQ second
            W_q[:, b] = pure_qunatize(W_bal) * mu_row
            col_scales[b] = mu_col / awq_b  # net: SINQ comp / AWQ comp

    Returns:
        weight_out : (out_features, in_features) same dtype as input — fake-quantised
                     weight with row scales already absorbed.
        col_scales : (in_features,) same dtype — per-channel activation pre-scales.
                     The caller should multiply input activations by col_scales before
                     the linear projection to undo the column rescaling.
    """
    from layer_wrapper_baseline_data_formats_methods import pure_qunatize

    with torch.no_grad():
        orig_dtype   = weight.dtype
        out_features, in_features = weight.shape

        # sfp4 / nvfp4 have a fixed 16-element hardware block; override caller's block_size
        if quant_method in ('sfp4', 'nvfp4'):
            block_size = 16

        assert in_features % block_size == 0, (
            f"in_features ({in_features}) must be divisible by block_size ({block_size})"
        )
        n_blocks = in_features // block_size

        W = weight.clone().float().cpu()

        # --- AWQ scales (ASINQ path) -------------------------------------------
        awq_scale = None
        if layer_activations is not None:
            fake_quant_fn = lambda W_s: pure_qunatize(
                W_s.to('cuda'), block_size, quant_method
            ).cpu()
            awq_scale = compute_awq_scale(
                W, layer_activations.cpu().float(), fake_quant_fn
            ).cpu().float()   # (in_features,)

        # --- Block-wise Sinkhorn + quantisation --------------------------------
        W_out      = torch.zeros_like(W)
        col_scales = torch.ones(in_features, dtype=torch.float32)

        for b in range(n_blocks):
            s, e  = b * block_size, (b + 1) * block_size
            W_b   = W[:, s:e]                              # (out_features, block_size)

            W_bal, mu_col, mu_row = sinkhorn_log(W_b)     # Sinkhorn first
            # mu_col: (block_size,)  — col scales (divide cols of W_b)
            # mu_row: (out_features,1) — row scales (divide rows of W_b)

            if awq_scale is not None:
                awq_b  = awq_scale[s:e]                    # (block_size,)
                W_bal  = W_bal * awq_b.unsqueeze(0)        # AWQ second, on balanced
                mu_col_eff = mu_col / awq_b
            else:
                mu_col_eff = mu_col

            # breakpoint()

            W_q = pure_qunatize(
                W_bal.float().to('cuda'), block_size, quant_method
            ).cpu()

            

            mu_row_q = _quantize_row_scale(mu_row, quant_method)
            W_out[:, s:e] = W_q * mu_row_q                # absorb quantized row scales
            col_scales[s:e] = mu_col_eff

        return W_out.to(orig_dtype), col_scales.to(orig_dtype)


################################################################################
################################################################################
######################## Wrap WithOut SINQ Method #################################
def wrap_layer_rtn4(layer, method ='sinq', group_size = 64):
    with torch.no_grad():
        dtype = layer.weight.dtype
        compute_dtype = dtype
        matrix  = layer.weight.clone().to(torch.float32).cpu()
        W = matrix.float()
        shape = W.shape
        channel_wise = True
        axis = 1
        dtype = torch.bfloat16
        W_f = W.to(dtype=dtype, device='cuda')
        tile = W_f.shape[-1]
        min_max = [0, 15]

        M = W
        block = group_size
        mshape = M.shape
        H,W = M.shape
        assert W%block==0, 'block must divide W'
        n_w = W//block

        M = M.view(H, W//block, block)
        M_batched = M.permute(1,0,2).contiguous().view(n_w, H, block)

        batch_size = M_batched.shape[0]
        Q_list, s1_list, s2_list, z_list = [], [], [], []
        for i in range(batch_size):
            mat_block = M_batched[i]
            Q_i, s1_i, s2_i, z_i = quantize_dual_scale_shift_no_Sinkhorn(mat_block, min_max, method=method)
            
            Q_list.append(Q_i)
            s1_list.append(s1_i)
            s2_list.append(s2_i)
            z_list.append(z_i)

        Q_manual = torch.stack(Q_list)
        s1_manual = torch.stack(s1_list) if s1_list[0] is not None else None
        s2_manual = torch.stack(s2_list) if s2_list[0] is not None else None  
        z_manual = torch.stack(z_list) if z_list[0] is not None else None

        Q = Q_manual
        s1 = s1_manual
        s2 = s2_manual
        z = z_manual

        z = z.permute(1,0,2).reshape(-1,1)

        Q = Q.permute(1,0,2).reshape(-1, block)
        s1 = s1.permute(1,0,2).reshape(-1,1)
        s2 = s2.permute(1,0,2).view(1, -1)
        # breakpoint()
        W_q, scale, zero, scale2, awq_scale = Q, s1, z, s2, None
        meta = {
        "nbits": 4,
        "group_size": group_size,
        "shape": shape,
        "scale": scale,
        "scale2": scale2,
        "awq_scale": awq_scale,
        "zero": zero,
        "axis": axis,
        "packing": None,
        "method": method,
        "compute_dtype": compute_dtype,
        }

        W_q = W_q.to(dtype)
        meta["packing"] = None
        
        Wr = dequantize(W_q, meta, use_unpack_kernel = False)
        mat_q = Wr.clone()
        layer.weight.copy_(mat_q.to(dtype))
################################################################################