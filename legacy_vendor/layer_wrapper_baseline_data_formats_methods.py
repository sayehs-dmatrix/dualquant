import math
import time
import torch
import torch.nn as nn
import transformers

from torch_quant import (
    convert_to_sfp,
    convert_float32_mat_to_scales_fp4_ebias,
    convert_float32_mat_to_scales_e5m3_fp4_ebias,
    float_to_fp4,
)
from sinq_functions import wrap_layer_with_sinq_method
from awq import *

from blockfmt import Format


E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
E8M0_EXPONENT_BIAS = 127
F8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
F8_E5M2_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0
F7_E4M2_MAX = 448.0
F7_E3M3_MAX = 30.0
F6_E3M2_MAX = 28.0
F6_E2M3_MAX = 7.5
F5_E2M2_MAX = 7.0
F4_E2M1_MAX = 6.0


################ Just to test ############################

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYMMETRIC PER-CHANNEL  (standard INT8 baseline, e.g. GPTQ Table 1)
#    q  = clamp( round(w / scale), qmin, qmax )
#    scale = max(|w|) / qmax          (one scale per output channel)
# ─────────────────────────────────────────────────────────────────────────────
def rtn_8bit(weight, bits=8):
    """
    Symmetric quantization with one scale per output channel (row).
    Used as the RTN baseline in GPTQ and most weight-only quant papers.

    Args:
        weight : (out_features, in_features)
        bits   : 8 for INT8, 4 for INT4

    Returns:
        Dequantized weight with same shape and dtype as input.
    """
    assert bits in (4, 8), "Only INT4 and INT8 supported"
    qmax = 2 ** (bits - 1) - 1          # 127 for INT8, 7 for INT4
    qmin = -(qmax + 1)                   # -128 for INT8, -8 for INT4

    # scale: shape (out_features, 1)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax

    w_q   = weight.div(scale).round_().clamp_(qmin, qmax)   # integer grid
    w_deq = w_q.mul(scale)                                   # back to float
    return w_deq


# ─────────────────────────────────────────────────────────────────────────────
# 2. ASYMMETRIC PER-GROUP  (standard INT4 baseline, e.g. AWQ / AutoGPTQ)
#    scale = (max - min) / (2^bits - 1)
#    zero  = round(-min / scale)           (stored as integer, shifts grid)
#    q     = clamp( round(w / scale) + zero, 0, 2^bits - 1 )
#    group_size = 128 is the community default for INT4
# ─────────────────────────────────────────────────────────────────────────────

def rtn_4bit(weight, bits=4, group_size=128):
    """
    Symmetric quantization with one scale per group of weights.

    Args:
        weight     : (out_features, in_features)
        bits       : 4 for INT4, 8 for INT8
        group_size : number of weights sharing a scale; -1 means per-row (one group per row)

    Returns:
        Dequantized weight with same shape and dtype as input.
    """
    assert bits in (4, 8), "Only INT4 and INT8 supported"
    qmax = 2 ** (bits - 1) - 1          # 7 for INT4, 127 for INT8
    qmin = -(qmax + 1)                   # -8 for INT4, -128 for INT8

    orig_shape = weight.shape
    if weight.dim() > 2:
        weight = weight.contiguous().reshape(-1, weight.shape[-1])
    
    # orig_shape  = weight.shape
    in_features = weight.shape[1]
    
    # orig_shape  = weight.shape
    # in_features = weight.shape[1]

    if group_size < 0 or group_size >= in_features:
        group_size = in_features

    assert in_features % group_size == 0, (
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    )

    # reshape: (out_features * n_groups, group_size)
    w = weight.reshape(-1, group_size)

    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    w_q   = w.div(scale).round_().clamp_(qmin, qmax)
    w_deq = w_q.mul(scale)

    return w_deq.reshape(orig_shape)


def pure_qunatize(tensor_to_quan, block_size, quant_method):
    if quant_method == 'rtn_int4':
        quantized_tensor = rtn_4bit(tensor_to_quan, bits=4, group_size=block_size)

    elif quant_method == 'rtn_int8':
        quantized_tensor = rtn_4bit(tensor_to_quan, bits=8, group_size=-1)   # per-channel affine
    
    elif quant_method == 'sfp4':
        quantized_tensor = convert_to_sfp(tensor_to_quan)
        # quantized_tensor = Format.from_shorthand("SFP<FP[1|2|1,1](_N)><FP[0|5|3,15](FN)>{16}").cast(tensor_to_quan, -1)

    elif quant_method == 'nvfp4':
        # tensor_to_quan_10 = tensor_to_quan *10
        # quantized_tensor_q10 = Format.from_shorthand("NVFP4[E2M1]{16}").cast(tensor_to_quan_10 , -1)
        # quantized_tensor = quantized_tensor_q10/10
        quantized_tensor = Format.from_shorthand("NVFP4[E2M1]{16}").cast(tensor_to_quan , -1)
        # print('HERE---------')
        # quantized_tensor_sfp = Format.from_shorthand("SFP<FP[1|2|1,1](_N)><FP[0|5|3,15](FN)>{16}").cast(tensor_to_quan, -1)

        # import torch
        # import matplotlib.pyplot as plt
        # import numpy as np

        # # your three tensors (n x 16)
        # # T1, T2, T3 = ...

        # fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"})
        # titles = ["Tensor 1", "Tensor 2", "Tensor 3"]

        # for ax, T, title in zip(axes, [tensor_to_quan, quantized_tensor, quantized_tensor_sfp], titles):
        #     data = T.abs().float().cpu().numpy()   # magnitude
        #     n, m = data.shape                      # (n, 16)
        #     X, Y = np.meshgrid(np.arange(m), np.arange(n))
        #     ax.plot_surface(X, Y, data, cmap="viridis")
        #     ax.set_title(title)
        #     ax.set_xlabel("Column (0-15)")
        #     ax.set_ylabel("Row")
        #     ax.set_zlabel("Magnitude")

        # plt.tight_layout()
        # plt.savefig("tensor_magnitudes.png", dpi=150)
        # plt.show()
        # breakpoint()
    elif quant_method == 'mxfp4':
        quantized_tensor = Format.from_shorthand(f"MXFP4[E2M1]{{{block_size}}}").cast(tensor_to_quan, -1)

    elif quant_method == 'mxfp8_e4m3':
        quantized_tensor = Format.from_shorthand(f"MXFP8[E4M3]{{{block_size}}}").cast(tensor_to_quan, -1)

    elif quant_method == 'mxfp8_e5m2':
        quantized_tensor = Format.from_shorthand(f"MXFP8[E5M2]{{{block_size}}}").cast(tensor_to_quan, -1)

    elif quant_method == 'mxint4':
        quantized_tensor = Format.from_shorthand(f"MXINT4{{{block_size}}}").cast(tensor_to_quan, -1)

    elif quant_method == 'mxint8':
        quantized_tensor = Format.from_shorthand(f"MXINT8{{{block_size}}}").cast(tensor_to_quan, -1)

    return quantized_tensor
################################################################################
################################################################################
# ----------- Pure quantization without split or SINQ -----------------
def wrap_just_quantize_layer(layer, layer_activations, perm, quant_method, act_quant_flag, block_size): ### before it was wrap_layer
    weight_matrix = layer.weight.clone().to(torch.float32).cpu()
    shape = weight_matrix.shape
    with torch.no_grad():
        dtype = layer.weight.dtype
        mat = layer.weight.clone().to(torch.float32).cpu()

        if layer_activations is not None:
            # awq_scale = compute_awq_scale(weight_matrix.view(shape), layer_activations, min_max=min_max, tile=tile, method=quant_method)
            # awq_scale = awq_scale.unsqueeze(0).to(weight_matrix.device).to(weight_matrix.dtype)
            # awq_scale_1d = awq_scale.flatten()
            # diag_matrix = torch.diag(awq_scale_1d )
            # matrix =  weight_matrix @ diag_matrix
            fake_quant_fn = lambda W_scaled: pure_qunatize(W_scaled.to('cuda'), block_size, quant_method).cpu()
            awq_scale = compute_awq_scale(weight_matrix.view(shape), layer_activations, fake_quant_fn)
            awq_scale = awq_scale.unsqueeze(0).to(weight_matrix.device).to(weight_matrix.dtype)
            awq_scale_1d = awq_scale.flatten()
            diag_matrix = torch.diag(awq_scale_1d)
            matrix = weight_matrix @ diag_matrix

            mat_q_scaled = pure_qunatize(matrix.to('cuda'), block_size, quant_method).cpu()
            mat_q = mat_q_scaled @ torch.linalg.inv(diag_matrix)

        elif layer_activations is None:
            mat_q = pure_qunatize(mat.to('cuda'), block_size, quant_method).cpu()
        layer.weight.copy_(mat_q.to(dtype))

################################################################################
################################################################################
# ----------- AWQ: activation-aware weight quantization ----------------------
# Scale input channels of W by s before quantizing, then fold s back out.
# The scale s is chosen to minimise reconstruction error on calibration acts.
# At inference the weight already encodes the rescaling, so no activation-side
# compensation is needed.
def wrap_awq_layer(layer, layer_activations, quant_method, block_size):
    assert layer_activations is not None, "wrap_awq_layer requires calibration activations"
    with torch.no_grad():
        dtype = layer.weight.dtype
        W     = layer.weight.clone().to(torch.float32).cpu()

        # fake_quant_fn uses the actual target format and block size so the
        # scale grid search minimises the real quantisation error proxy
        def fake_quant_fn(W_scaled):
            return pure_qunatize(W_scaled.to('cuda'), block_size, quant_method).cpu()

        scale = compute_awq_scale(W, layer_activations.cpu().float(),
                                  fake_quant_fn)            # (in_features,)

        # Scale input channels up, quantise, fold scale back element-wise
        W_q = pure_qunatize((W * scale.unsqueeze(0)).to('cuda'),
                            block_size, quant_method).cpu()
        W_q = W_q / scale.unsqueeze(0)

        layer.weight.copy_(W_q.to(dtype))

################################################################################
################################################################################

def wrap_sinq_layer(layer, layer_activations, quant_method, act_quant_flag, block_size):
    with torch.no_grad():
        dtype = layer.weight.dtype
        weight_matrix = layer.weight.clone()
        Wq, weight_reduce_dim_scales = wrap_layer_with_sinq_method(weight_matrix, layer_activations, quant_method, block_size)

        if act_quant_flag:
            layer.weight.copy_(Wq.to(dtype))
            return weight_reduce_dim_scales
        else:
            recosntricted_weight = Wq * weight_reduce_dim_scales
            layer.weight.copy_(recosntricted_weight.to(dtype))
        return None


################################################################################
# GPTQ class — copied from:
#   GPTQ_to_Be_Applied/My_Llama_GPTQ_SFP4_vs_SBFP12/
#   The_Working_GPTQ_for_Llama_20250912/gptq_blockfmt.py
################################################################################

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer: any = None

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=16, actorder=False):

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        ##### blockfmt does not have this ################################
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # GPTQ act_order: permute columns by descending diag(H), so the most
        # important columns (largest input second moment) are quantised first
        # while remaining columns still have full freedom to absorb their
        # quantisation error. Reference: IST-DASLab/gptq + QuaRot fake_quant.
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        diag = torch.arange(self.columns, device=self.dev)
        damp = percdamp * torch.mean(torch.diag(H))
        for attempt in range(6):
            H_try = H.clone()
            H_try[diag, diag] += damp
            try:
                H_try = torch.linalg.cholesky(H_try)
                H_try = torch.cholesky_inverse(H_try)
                H_try = torch.linalg.cholesky(H_try, upper=True)
                H = H_try
                break
            except torch._C._LinAlgError:
                damp *= 10
                if attempt == 5:
                    raise
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if isinstance(self.quantizer, nBits_quantizer):
                # Standard column-by-column GPTQ matching the IST-DASLab original.
                # find_params refreshes the group scale every groupsize columns from the
                # globally-updated W; each column's quantization error is then propagated
                # to all remaining columns in the block before they are quantized.
                for j in range(count):
                    if groupsize != -1 and (i1 + j) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + j):(i1 + j + groupsize)], weight=True
                        )
                    w_col = W1[:, j]
                    d = Hinv1[j, j]
                    q_col = self.quantizer.quantize(w_col.unsqueeze(1)).squeeze(1)
                    Q1[:, j] = q_col
                    err1 = (w_col - q_col) / d
                    W1[:, j:] -= err1.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
                    Err1[:, j] = err1
            else:
                # Batch approach for block quantizers (mxfp4, sfp4, nvfp4, etc.).
                # These formats share one scale across a fixed block of elements,
                # so the whole block must be quantized together.
                for j1 in range(0, count, groupsize):
                    j2 = min(j1 + groupsize, count)
                    w = W1[:, j1:j2]
                    hinv = Hinv1[j1:j2, j1:j2]
                    self.quantizer.find_params(w, weight=True)
                    q = self.quantizer.quantize(w)
                    err = (w - q).matmul(torch.linalg.inv(hinv))
                    Q1[:, j1:j2] = q
                    W1[:, j2:] -= err.matmul(Hinv1[j1:j2, j2:])
                    Err1[:, j1:j2] = err

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        return self.layer.weight.data

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


################################################################################
# Quantizer classes for GPTQ
# sfp4 / sfp4_e5m3 : delegate to functions imported from torch_quant.py
# all other formats : delegate to Format.from_shorthand(...).cast() from blockfmt
################################################################################

class sfp4_quantizer_cls(nn.Module):
    """SFP4 with E4M4 block scale (the vendor sfp4)."""
    def __init__(self, block_size=16):
        super().__init__()
        self.blocksize = 16   # SFP4 always uses 16-element blocks
        self.scales = None

    def configure(self, bits):
        self.bits = bits

    def find_params(self, w, *args, **kwargs):
        self.scales, _, self.ebias = convert_float32_mat_to_scales_fp4_ebias(w, ebias=None)

    def quantize(self, w):
        # self.scales: (rows, 1) when fq_groupsize=16; broadcasts over w: (rows, 16)
        fp4_w = float_to_fp4(w / self.scales)
        return (self.scales * fp4_w).reshape(w.shape)

    def enabled(self):
        return True

    def ready(self):
        return True


class sfp4_e5m3_quantizer_cls(nn.Module):
    """SFP4 with E5M3 block scale."""
    def __init__(self, block_size=16):
        super().__init__()
        self.blocksize = 16   # SFP4 always uses 16-element blocks
        self.scales = None

    def configure(self, bits):
        self.bits = bits

    def find_params(self, w, *args, **kwargs):
        self.scales, _, self.ebias = convert_float32_mat_to_scales_e5m3_fp4_ebias(w, ebias=None)

    def quantize(self, w):
        # self.scales: (rows, 1) when fq_groupsize=16; broadcasts over w: (rows, 16)
        fp4_w = float_to_fp4(w / self.scales)
        return (self.scales * fp4_w).reshape(w.shape)

    def enabled(self):
        return True

    def ready(self):
        return True


class _FormatQuantizer(nn.Module):
    """Wraps Format.from_shorthand(...).cast() for all blockfmt formats.
    find_params is a no-op: Format.cast() computes scale and quantizes in one call."""
    def __init__(self, shorthand):
        super().__init__()
        self.fmt = Format.from_shorthand(shorthand)

    def configure(self, bits):
        pass

    def find_params(self, w, *args, **kwargs):
        pass

    def quantize(self, w):
        return self.fmt.cast(w, -1)

    def enabled(self):
        return True

    def ready(self):
        return True


def nvfp4_quantizer_cls(block_size=16):
    return _FormatQuantizer("NVFP4[E2M1]{16}")


def mxfp4_quantizer_cls(block_size=16):
    return _FormatQuantizer(f"MXFP4[E2M1]{{{block_size}}}")


def mxfp8_e4m3_quantizer_cls(block_size=32):
    return _FormatQuantizer(f"MXFP8[E4M3]{{{block_size}}}")


def mxfp8_e5m2_quantizer_cls(block_size=32):
    return _FormatQuantizer(f"MXFP8[E5M2]{{{block_size}}}")


def mxint4_quantizer_cls(block_size=32):
    return _FormatQuantizer(f"MXINT4{{{block_size}}}")


def mxint8_quantizer_cls(block_size=32):
    return _FormatQuantizer(f"MXINT8{{{block_size}}}")



class nBits_quantizer:
    def __init__(self, bits=4, group_size=-1, sym=True):
        self.configure(bits, group_size, sym)
        self._scale = None
        self._zero = None

    def configure(self, bits=4, group_size=-1, sym=True):
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        if self.sym:
            self.qmin = -(2 ** (self.bits - 1))
            self.qmax = (2 ** (self.bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (2 ** self.bits) - 1

    def find_params(self, Weight_mat, *args, **kwargs):
        """Cache scale/zero from a weight group (rows × group_size).
        Called at the start of each quantizer group in the GPTQ inner loop."""
        w = Weight_mat
        if self.sym:
            self._scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / self.qmax
            self._zero = None
        else:
            w_min = w.amin(dim=1, keepdim=True)
            w_max = w.amax(dim=1, keepdim=True)
            self._scale = (w_max - w_min).clamp(min=1e-8) / (self.qmax - self.qmin)
            self._zero = (self.qmin - w_min / self._scale).round().clamp(self.qmin, self.qmax)

    def quantize(self, w, dim=1):
        """Quantize w using the scale cached by find_params."""
        assert self._scale is not None, "call find_params before quantize"
        scale = self._scale
        if self.sym:
            q = (w / scale).round_().clamp_(self.qmin, self.qmax)
            return q * scale
        else:
            q = (w / scale + self._zero).round_().clamp_(self.qmin, self.qmax)
            return (q - self._zero) * scale

    def forward(self, w):
        return self.quantize(w)

    def enabled(self):
        return True

    def ready(self):
        return True


################################################################################
# wrap_gptq_layer — follows the same steps as:
#   quantize_llama_sfp4_vs_sbfp12.py  (GPTQ object setup + add_batch +
#   fasterquant), but packaged as a single layer wrapper function.
#
# Quantizer mapping:
#   sfp4      -> sfp4_quantizer_cls         (the vendor SFP4, E4M4 block scale from torch_quant)
#   nvfp4     -> nvfp4_quantizer_cls        (NVIDIA FP4,  NVFP4[E2M1]{16} via blockfmt)
#   mxfp4     -> mxfp4_quantizer_cls        (MX FP4,      MXFP4[E2M1]{block} via blockfmt)
#   mxint4    -> mxint4_quantizer_cls       (MX INT4,     MXINT4{block} via blockfmt)
#   mxint8    -> mxint8_quantizer_cls       (MX INT8,     MXINT8{block} via blockfmt)
#   mxfp8_e4m3 -> mxfp8_e4m3_quantizer_cls (MX FP8 E4M3, MXFP8[E4M3]{block} via blockfmt)
#   mxfp8_e5m2 -> mxfp8_e5m2_quantizer_cls (MX FP8 E5M2, MXFP8[E5M2]{block} via blockfmt)
#   rtn_int4  -> nBits_quantizer            (INT4 symmetric per-channel)
#   rtn_int8  -> nBits_quantizer            (INT8 symmetric per-channel)
################################################################################

def wrap_gptq_layer(layer, layer_activations, quant_method, act_quant_flag, block_size,
                    blocksize=128, percdamp=0.01):
    with torch.no_grad():
        dtype = layer.weight.dtype

        # Step 1: create GPTQ object for this layer
        gptq_obj = GPTQ(layer)

        # Step 2: assign quantizer based on quant_method
        if quant_method == 'sfp4':
            gptq_obj.quantizer = sfp4_quantizer_cls(block_size=block_size)
            gptq_obj.quantizer.configure(bits=4)
        elif quant_method == 'nvfp4':
            gptq_obj.quantizer = nvfp4_quantizer_cls(block_size=block_size)
        elif quant_method == 'mxfp4':
            gptq_obj.quantizer = mxfp4_quantizer_cls(block_size=block_size)
        elif quant_method == 'mxint4':
            gptq_obj.quantizer = mxint4_quantizer_cls(block_size=block_size)
        elif quant_method == 'mxint8':
            gptq_obj.quantizer = mxint8_quantizer_cls(block_size=block_size)
        elif quant_method == 'mxfp8_e4m3':
            gptq_obj.quantizer = mxfp8_e4m3_quantizer_cls(block_size=block_size)
        elif quant_method == 'mxfp8_e5m2':
            gptq_obj.quantizer = mxfp8_e5m2_quantizer_cls(block_size=block_size)
        elif quant_method == 'rtn_int4':
            gptq_obj.quantizer = nBits_quantizer(bits=4, group_size=-1, sym=True)
        elif quant_method == 'rtn_int8':
            # symmetric per-channel: standard for INT8 GPTQ
            gptq_obj.quantizer = nBits_quantizer(bits=8, group_size=-1, sym=True)
        else:
            raise ValueError(f"wrap_gptq_layer: unsupported quant_method '{quant_method}'")

        # Step 3: accumulate Hessian from calibration activations
        if layer_activations is not None:
            gptq_obj.add_batch(layer_activations.to(gptq_obj.dev), None)

        # Step 4: run GPTQ — groupsize controls quantizer granularity inside fasterquant:
        #   rtn_int8  : full row (per-channel, one scale per output channel)
        #   rtn_int4  : block_size (128) columns share one scale — standard grouped INT4
        #   sfp4/sfp4_e5m3: always 16 — convert_float32_mat_to_scales_fp4_ebias uses 16-element blocks
        #                    and scale is max(|block|)/6.0 (FP4 max = 6.0). Using any other groupsize
        #                    causes a shape mismatch in quantize (scales (rows,N) vs w (rows, groupsize)).
        #   MX/NVFP4 formats: block_size from config (16 or 32 per MX spec)
        if quant_method == 'rtn_int8':
            fq_groupsize = gptq_obj.columns   # per-channel affine
        elif quant_method == 'rtn_int4':
            fq_groupsize = block_size
        elif quant_method in ('sfp4', 'sfp4_e5m3'):
            fq_groupsize = 16
        else:
            fq_groupsize = block_size

        gptq_obj.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=fq_groupsize)
        gptq_obj.free()
