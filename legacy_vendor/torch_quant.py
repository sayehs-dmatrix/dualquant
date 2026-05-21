import sys
import time
import numpy as np
import torch
from functools import partial

def convert_to_sbfp12(matrix, ebias=None, dev=None):
    # convert float32 matrix to sbfp12 format
    # with ebias
    # if ebias is none, ebias is computed
    # return fake quantized values
    matrix = matrix.to(torch.float32)
    scales_e4m4, mat_int4, ebias = convert_float32_mat_to_scales_int4_ebias(matrix=matrix, ebias=ebias)
    scales_e4m4 = scales_e4m4.reshape(*matrix.shape[:-1], -1, 1)
    res = scales_e4m4 * mat_int4
    return res.reshape(*matrix.shape)

def convert_to_sfp(matrix, ebias=None):
    # convert float32 matrix to sfp4 format
    # with block size of 16
    # fp4 are calculated according to round to nearest
    # if ebias is not given, ebias is computed
    # return fake quantized values
    matrix = matrix.to(torch.float32)
    scales_e4m4, mat_fp4, ebias = convert_float32_mat_to_scales_fp4_ebias(matrix=matrix, ebias=ebias)
    scales_e4m4 = scales_e4m4.reshape(*matrix.shape[:-1], -1, 1)
    res = scales_e4m4 * mat_fp4
    return res.reshape(*matrix.shape)

def convert_to_sfp_e5m3(matrix, ebias=None):
    # same as convert to sfp but scales are e5 m3 format
    matrix = matrix.to(torch.float32)
    scales_e5m3, mat_fp4, ebias = convert_float32_mat_to_scales_e5m3_fp4_ebias(matrix=matrix, ebias=ebias)
    scales_e5m3 = scales_e5m3.reshape(*matrix.shape[:-1], -1, 1)
    res = scales_e5m3 * mat_fp4
    return res.reshape(*matrix.shape)

def convert_to_nvfp4(matrix, ebias=None):
    # convert to nvidia fp4 format
    # scales are e4m3 format
    # return fake quantized values
    matrix = matrix.to(torch.float32)
    scales_e4m3, mat_fp4, ebias = convert_float32_mat_to_e4m3_scales_fp4_ebias(matrix=matrix, ebias=ebias)
    scales_e4m3 = scales_e4m3.reshape(*matrix.shape[:-1], -1, 1)
    res = scales_e4m3 * mat_fp4
    return res.reshape(*matrix.shape)



   
def convert_float32_mat_to_scales_int4_ebias(matrix, ebias=None):
    # given float 32 matrix
    # compute scales for sbfp12
    # and the integer matrix elements
    # and the ebias if none is given
    # return as a tuple of scales, int4's, ebias
    if ebias is None:
        ebias = find_ebias_sbfp(matrix=matrix)
    scales = torch.clone(matrix)
    scales = scales.reshape(*scales.shape[:-1], -1, 16) 
    scales = torch.abs(scales)
    scales = torch.max(scales, dim=-1)[0] 
    scales /= torch.tensor(7.0).to(torch.float32)
    scales = fake_quantize_float32_to_e4m4(mat=scales, ebias=ebias).reshape(*matrix.shape[:-1], -1)
    int4_mat = matrix.reshape(*scales.shape[:-1], -1, 16) / scales[..., None]
    int4_mat = round_away_zero(int4_mat)
    int4_mat = torch.minimum(int4_mat, torch.tensor(7, dtype=torch.int8))
    int4_mat = torch.maximum(int4_mat, torch.tensor(-7, dtype=torch.int8))
    return scales, int4_mat, ebias

def convert_float32_mat_to_scales_e5m3_fp4_ebias(matrix, ebias=None):
    # convert float32 matrix to scales (e5m3), fp4 and ebias
    # for use with sfp4 with e5m3 scales
    # if ebias is not given compute ebias
    if ebias is None:
        max_val = torch.max(torch.abs(matrix)) / 6.0
        ebias = find_ebias_e5(matrix=max_val)
    scales = torch.clone(matrix)
    scales = scales.reshape(*scales.shape[:-1], -1, 16) 
    scales = torch.abs(scales)
    scales = torch.max(scales, dim=-1)[0] 
    scales /= torch.tensor(6.0, dtype=torch.float32)
    #scales /= 7.0 # need to convert scales to e4m4 first
    # then compute int4_mat
    scales = fake_quantize_float32_to_e5m3(mat=scales, ebias=ebias).reshape(*matrix.shape[:-1], -1)
    fp4_mat = matrix.reshape(*scales.shape[:-1], -1, 16) / scales[..., None]
    fp4_mat = float_to_fp4(fp4_mat)
    return scales, fp4_mat, ebias

def convert_float32_mat_to_scales_fp4_ebias(matrix, ebias=None):
    # same as convert_float32_mat_to_scales_e5m3_fp4_ebias but
    # for e4m4 scales for sfp4 format
    if ebias is None:
        max_val = torch.max(torch.abs(matrix)) / 6.0
        ebias = find_ebias(matrix=max_val)
    scales = torch.clone(matrix)
    scales = scales.reshape(*scales.shape[:-1], -1, 16) 
    scales = torch.abs(scales)
    scales = torch.max(scales, dim=-1)[0] 
    scales /= torch.tensor(6.0, dtype=torch.float32)
    #scales /= 7.0 # need to convert scales to e4m4 first
    # then compute int4_mat
    scales = fake_quantize_float32_to_e4m4(mat=scales, ebias=ebias).reshape(*matrix.shape[:-1], -1)
    fp4_mat = matrix.reshape(*scales.shape[:-1], -1, 16) / scales[..., None]
    fp4_mat = float_to_fp4(fp4_mat)
    return scales, fp4_mat, ebias

def convert_float32_mat_to_e4m3_scales_fp4_ebias(matrix, ebias=None):
    # same as convert_float32_mat-to_scales_e5m3_fp4_ebias but
    # for e4m3 scales for nvfp4 format
    if ebias is None:
        max_val = torch.max(torch.abs(matrix)) / 6.0
        ebias = find_ebias(matrix=max_val)
    scales = torch.clone(matrix)
    scales = scales.reshape(*scales.shape[:-1], -1, 16)
    scales = torch.abs(scales)
    scales = torch.max(scales, dim=-1)[0]
    scales /= torch.tensor(6.0, dtype=torch.float32)
    scales = fake_quantize_float32_to_e4m3(mat=scales, ebias=ebias).reshape(*matrix.shape[:-1], -1)
    fp4_mat = matrix.reshape(*scales.shape[:-1], -1, 16) / scales[..., None]
    fp4_mat = float_to_fp4(fp4_mat)
    return scales, fp4_mat, ebias

def find_ebias_sbfp(matrix):
    # find the ebias for the sbfp12 format
    # 1 <= ebias <= 15
    max_val = torch.max(torch.abs(matrix)) / torch.tensor(7.0, dtype=torch.float32)
    mat_exp, mat_man = extract_rounded_4bit_exp_man(max_val)
    mat_exp -= 127
    ebias = torch.minimum(15 - mat_exp, torch.tensor(15))
    ebias = torch.maximum(ebias, torch.tensor(1))
    return ebias

def find_ebias(matrix):
    # find ebias for sfp4 (e4m4), nvfp4, etc.
    matrix = matrix.view(torch.int32)
    exp = matrix & 0x7F800000
    exp >>= 23
    exp -= 127
    return 15 - exp

def find_ebias_e5(matrix):
    # find ebias for sfp4 (e5m3) format
    matrix = matrix.view(torch.int32)
    exp = matrix & 0x7F800000
    exp >>= 23
    exp -= 127
    return 31 - exp


def round_away_zero(mat):
    # round away from zero
    floor = torch.floor(mat)
    ceil = torch.ceil(mat)
    pos = torch.where(mat >= 0.0,
            torch.where(mat - floor >= 0.5,
                floor + 1.0, floor
                ),
            0.0
            )
    neg = torch.where(mat < 0.0,
            torch.where(ceil - mat >= 0.5,
                ceil - 1.0, ceil
                ),
            0.0
            )
    res = pos + neg
    res = res.to(torch.int8)
    return res


def fake_quantize_float32_to_e5m3(mat, ebias):
    # given a matrix and ebias
    # convert to e5m3 format
    # and then return as a float32
    mat_man, mat_exp = convert_float32_to_5exp_3man(mat, ebias)
    mat_man = mat_man.to(torch.int32)
    mat_exp = mat_exp.to(torch.int32)
    mat_exp = torch.bitwise_left_shift(mat_exp, 23)
    mat_man = torch.bitwise_left_shift(mat_man, 20)
    mat_e5m3 = torch.bitwise_or(mat_exp, mat_man)
    mat_e5m3 = mat_e5m3.view(torch.float32)
    return mat_e5m3


def fake_quantize_float32_to_e4m4(mat, ebias):
    # given a float 32 matrix
    # convert to e4m4 format
    # and then return as float32
    mat_man, mat_exp = convert_float32_to_4exp_4man(mat, ebias)
    mat_man = mat_man.to(torch.int32)
    mat_exp = mat_exp.to(torch.int32)
    mat_exp = torch.bitwise_left_shift(mat_exp, 23)
    mat_man = torch.bitwise_left_shift(mat_man, 19)
    mat_e4m4 = torch.bitwise_or(mat_exp, mat_man)
    mat_e4m4 = mat_e4m4.view(torch.float32)
    return mat_e4m4



def convert_float32_to_5exp_3man(mat, ebias):
    # given a float32 matrix
    # convert to e5m3 format
    # and return mantissa, exponent
    dev = mat.device
    mat_exp, mat_man = extract_rounded_5exp_3man(mat)
    mat_exp = mat_exp - 127 + ebias
    mat_man = torch.where(mat_exp < 0, 0, mat_man)
    mat_exp = torch.where(mat_exp < 0, torch.zeros(size=list(mat_exp.shape), dtype=torch.int16, device=dev), mat_exp)
    mat_man = torch.where(mat_exp > 31, 7, mat_man)
    mat_exp = torch.where(mat_exp > 31, 31, mat_exp)
    mat_exp = mat_exp + 127 - ebias
    return mat_man, mat_exp

def convert_float32_to_4exp_4man(mat, ebias):
    # given a float32 matrix
    # convert to e4 m4 and return mantissa, exponent
    dev = mat.device
    mat_exp, mat_man = extract_rounded_4bit_exp_man(mat)
    mat_exp = mat_exp - 127 + ebias
    mat_man = torch.where(mat_exp < 0, 0, mat_man)
    mat_exp = torch.where(mat_exp < 0, torch.zeros(size=list(mat_exp.shape), dtype=torch.int16, device=dev), mat_exp)
    mat_man = torch.where(mat_exp > 15, 15, mat_man)
    mat_exp = torch.where(mat_exp > 15, 15, mat_exp)
    mat_exp = mat_exp + 127 - ebias
    return mat_man, mat_exp

def extract_rounded_5exp_3man(mat):
    # given a float32 matrix
    # extract rounded 5 bit exponent, 3 bit mantissa
    mat = mat.view(torch.int32)
    mat_man = torch.bitwise_right_shift(torch.bitwise_and(mat, torch.tensor(2 ** 23 - 1, dtype=torch.int32)), 18) # 5 bits
    mat_exp = torch.bitwise_right_shift(torch.bitwise_and(mat, 255 << 23), 23)
    mat_exp = mat_exp.to(torch.int16)
    sticky_one = torch.where(torch.bitwise_and(mat, 2 ** 19 - 1) != 0, 1, 0)
    mat_man = torch.bitwise_or(mat_man, sticky_one)
    and_3 = torch.bitwise_and(mat_man, 3)
    and_6 = torch.bitwise_and(mat_man, 6)
    ends_in_3_or_6 = torch.where(torch.logical_or(and_3 == 3, and_6 == 6), 4, 0)
    mat_man += ends_in_3_or_6
    mat_man = torch.bitwise_right_shift(mat_man, 2)
    add_one_to_exp = torch.where(torch.bitwise_and(mat_man, 8) == 8, 1, 0)
    mat_man = torch.where(torch.bitwise_and(mat_man, 8) == 8, 0, mat_man)
    mat_exp += add_one_to_exp
    return mat_exp, mat_man


def extract_rounded_4bit_exp_man(mat):
    # extract rounded 4 bit exponent, 4 bit mantissa
    mat = mat.view(torch.int32)
    mat_man = torch.bitwise_right_shift(torch.bitwise_and(mat, torch.tensor(2 ** 23 - 1, dtype=torch.int32)), 17) # 6 bits
    mat_exp = torch.bitwise_right_shift(torch.bitwise_and(mat, 255 << 23), 23)
    mat_exp = mat_exp.to(torch.int16)
    sticky_one = torch.where(torch.bitwise_and(mat, 2 ** 18 - 1) != 0, 1, 0)
    mat_man = torch.bitwise_or(mat_man, sticky_one)
    and_3 = torch.bitwise_and(mat_man, 3)
    and_6 = torch.bitwise_and(mat_man, 6)
    ends_in_3_or_6 = torch.where(torch.logical_or(and_3 == 3, and_6 == 6), 4, 0)
    mat_man += ends_in_3_or_6
    mat_man = torch.bitwise_right_shift(mat_man, 2)
    add_one_to_exp = torch.where(torch.bitwise_and(mat_man, 16) == 16, 1, 0)
    mat_man = torch.where(torch.bitwise_and(mat_man, 16) == 16, 0, mat_man)
    mat_exp += add_one_to_exp
    return mat_exp, mat_man

def fake_quantize_float32_to_e4m3(mat, ebias):
    # given a float32 matrix
    # convert to e4 m3 and then return the equivalent float32
    mat_man, mat_exp = convert_float32_to_4exp_3man(mat, ebias)
    mat_exp <<= 23
    mat_man <<= 20
    res = mat_exp | mat_man
    res = res.view(torch.float32)
    return res

def convert_float32_to_4exp_3man(mat, ebias):
    # given a float32 return mantissa (rounded to 3bits) and exponent (rounded to 4 bits)
    dev = mat.device
    mat_exp, mat_man = extract_rounded_4bit_exp_3bit_man(mat)
    mat_exp = mat_exp - 127 + ebias
    mat_man = torch.where(mat_exp < 0, 0, mat_man)
    mat_exp = torch.where(mat_exp < 0, torch.zeros(size=list(mat_exp.shape), dtype=torch.int16, device=dev), mat_exp)
    mat_man = torch.where(mat_exp > 15, 7, mat_man)
    mat_exp = torch.where(mat_exp > 15, 15, mat_exp)
    mat_exp = mat_exp + 127 - ebias
    return mat_man, mat_exp

def extract_rounded_4bit_exp_3bit_man(mat):
    # given float32 matrix
    # compute rounded 4 bit exponent and 3 bit mantissa
    mat = mat.view(torch.int32)
    mat_exp = mat & 0x7F800000
    mat_man = mat & 0x007FFFFF
    mat_man >>= 18 # 5 bits
    mat_exp >>= 23
    sticky_one = torch.where((mat & 0x0003FFFF) != 0, 1, 0)
    mat_man |= sticky_one
    and_3 = mat_man & 3
    and_6 = mat_man & 6
    ends_in_3_or_6 = torch.where(torch.logical_or(and_3 == 3, and_6 == 6), 4, 0)
    mat_man += ends_in_3_or_6
    mat_man = torch.bitwise_right_shift(mat_man, 2)
    add_one_to_exp = torch.where(torch.bitwise_and(mat_man, 8) == 8, 1, 0)
    mat_man = torch.where(torch.bitwise_and(mat_man, 8) == 8, 0, mat_man)
    mat_exp += add_one_to_exp
    return mat_exp, mat_man


def convert_to_mxfp_infty(mat, block_size=16, exponent=8):
    # convert to mxfp using 'round to infinity'
    mat = mat.to(torch.float32).to('cuda:0')
    shape = mat.shape
    exps = mat.reshape(*shape[:-1], -1, block_size)
    exps = torch.max(torch.abs(exps), dim=-1)[0] / 6.0
    exps = torch.log2(exps)
    exps = torch.ceil(exps)
    exps = torch.clamp(exps, -127.0, 127.0)
    els = mat.reshape(*shape[:-1], -1, block_size) / 2. ** exps[..., None]
    els = float_to_fp4(els)
    res = (2 ** exps[..., None]) * els
    res = res.reshape(shape)
    return res

  

def float_to_fp4(mat):
    # Due to Nikita
    # round to nearest fp4 value
    dev = mat.device
    mask_done = torch.zeros_like(mat, dtype=torch.bool, device=dev)
    sign = mat < 0
    mat = torch.abs(mat)
    cur_mask = mat > 5.0 # mask 6 
    mat[cur_mask] = 6.0
    mask_done |= cur_mask
    cur_mask = mat >= 3.5 # mask_4
    mat[torch.logical_and(cur_mask, torch.logical_not(mask_done))] = 4.0
    mask_done |= cur_mask
    cur_mask = mat > 2.5  # mask 3
    mat[torch.logical_and(cur_mask, torch.logical_not(mask_done))] = 3.0
    mask_done |= cur_mask
    cur_mask = mat >= 1.75 # mask 2
    mat[torch.logical_and(cur_mask, torch.logical_not(mask_done))] = 2.0
    mask_done |= cur_mask
    cur_mask = mat > 1.25 # mask 1.5
    mat[torch.logical_and(cur_mask, torch.logical_not(mask_done))] = 1.5
    mask_done |= cur_mask
    cur_mask = mat >= 0.75 # mask 1 
    mat[torch.logical_and(cur_mask, torch.logical_not(mask_done))] = 1.0
    mask_done |= cur_mask
    cur_mask = mat > 0.25
    mat[torch.logical_and(cur_mask, torch.logical_not(mask_done))] = 0.5
    mask_done |= cur_mask
    mat[torch.logical_not(mask_done)] = 0.0
    mat[sign] *= -1.0
    return mat


def convert_to_mxfp(mat, block_size=16, exponent=8, bias=None):
    # convert to mxfp using original algorithm
    # from microsoft paper
    mat = mat.to(torch.float32).to('cuda:0')
    shape = mat.shape
    exps = mat.reshape(*shape[:-1], -1, block_size)
    exps = torch.abs(exps) / 4.0
    exps = torch.log2(exps)
    exps = torch.floor(exps)
    exps = torch.max(exps, -1)[0]
    exps = torch.maximum(exps, torch.tensor(-128.0))
    exps = torch.minimum(exps, torch.tensor(127.0))
    els = mat.reshape(*shape[:-1], -1, block_size) / 2. ** exps[..., None]
    els = float_to_fp4(els)
    res = (2 ** exps[..., None]) * els
    res = res.reshape(*shape)
    return res


def convert_to_bfp16(mat, block_size=64):
    # convert to bfp 16 64
    # taken as best as possible from c code on git repository
    dev = mat.device
    mat = mat.to(torch.float32)
    m, n = mat.shape
    s = mat < 0
    mat = torch.abs(mat)
    exp_max = torch.zeros(m, n // block_size, dtype=torch.int32, device=dev)
    sticky = torch.zeros(m, n, dtype=torch.int32, device=dev)
    man = torch.zeros(m, n, dtype=torch.int32, device=dev)
    mat = mat.view(torch.int32)
    exp = mat & 0x7F800000
    exp >>= 23
    frac = mat & 0x007FFFFF
    exp_zero = exp == 0
    exp_255 = exp == 255
    man[exp_255] = 0x00FE0000
    sticky[exp_255] = 0
    exp[exp_255] = 260
    mask_done = exp_255
    man[exp_zero] = 0
    mask_done = torch.logical_or(exp_zero, mask_done)
    man[torch.logical_not(mask_done)] = frac[torch.logical_not(mask_done)] | 0x00800000
    exp_max = exp.reshape(m, -1, block_size)
    exp_max = torch.max(exp_max, -1)[0]
    diffs = exp_max[:, :, None] - exp.reshape(m, -1, block_size)
    diffs = diffs.reshape(m, n)
    zero_mask = diffs == 0
    sticky[zero_mask] = man[zero_mask] & 0x0000FFFF
    tmp_mask = torch.logical_and(diffs < 32, torch.logical_not(zero_mask))
    sticky[tmp_mask] = man[tmp_mask] & ( (1 << diffs[tmp_mask]) - 1)
    man[tmp_mask] >>= diffs[tmp_mask]
    mask = torch.logical_or(diffs < 32, zero_mask)
    sticky[torch.logical_not(mask)] = 0
    man[torch.logical_not(mask)] = 0
    sticky[torch.logical_not(zero_mask)] += man[torch.logical_not(zero_mask)] & 0x0000FFFF

    round_bit = (man >> 16) & 0x00000001
    lsb = (man >> 17) & 0x00000001
    mask = torch.logical_and(round_bit == 1, torch.logical_or(sticky > 0, lsb == 1))
    man_tmp = man + (0x00020000)
    tmp_mask = torch.logical_and(mask, man_tmp < 0x01000000)
    man[tmp_mask] = man_tmp[tmp_mask]
    man >>= 17
    man[s] = -man[s]
    exp_out = torch.zeros(m, n, dtype=torch.int32, device=dev)
    exp_max = exp_max[:, :, None].expand(-1, -1, block_size).reshape(m, n)
    exp_max_zero = exp_max == 0
    res = torch.zeros(m, n, dtype=torch.float32, device=dev)
    res[exp_max_zero] = 0.0
    exp_out[exp_max_zero] = -128
    exp_out[torch.logical_not(exp_max_zero)] = exp_max[torch.logical_not(exp_max_zero)] - 133
    res[torch.logical_not(exp_max_zero)] = (2. ** exp_out[torch.logical_not(exp_max_zero)]) * man[torch.logical_not(exp_max_zero)]
    return res


def get_base_16_ufunc():
    # for testing
    base_16 = partial(int, base=16)
    return np.frompyfunc(base_16, 1, 1)
    
    


def e4m4_to_float32(mat, ebias):
    # convert from e4m4 back to float32
    # for testing
    # read mat as uint32
    mat_exp = torch.bitwise_and(mat, torch.int8( (2 ** 4 - 1) << 4))
    mat_exp = torch.bitwise_right_shift(mat_exp, 4)
    mat_exp = mat_exp.to(torch.int16)
    mat_exp = mat_exp + 127 - ebias
    mat_exp = mat_exp.to(torch.int32)
    mat_exp = torch.bitwise_left_shift(mat_exp, 23)
    mat_man = torch.bitwise_and(mat, torch.int8(2 ** 4 - 1))
    mat_man = mat_man.to(torch.int32)
    mat_man = torch.bitwise_left_shift(mat_man, 19)
    res = torch.bitwise_or(mat_man, mat_exp)
    res = res.view(torch.float32)
    return res.reshape(*mat.shape)

def read_input_floats(input_float_csv):
    # given a csv of float16 hex values
    # return float32 matrix
    hex_float16 = np.loadtxt(input_float_csv, dtype=np.str_, delimiter=',')
    base_16_ufunc = get_base_16_ufunc()
    base_16_float16 = base_16_ufunc(hex_float16)
    base_16_float16 = base_16_float16.astype(np.int16)
    base_16_bytes = base_16_float16.tobytes()
    float16_input = np.frombuffer(base_16_bytes, dtype=np.float16)
    float32_input = float16_input.astype(np.float32)
    float32_input = float32_input.reshape(*hex_float16.shape)
    return float32_input

def read_scales_and_int4s(input_sbfp12_csv):
    # given a csv of hex values for sbfp12
    # return scales and int4 values
    hex_sbfp12 = np.loadtxt(input_sbfp12_csv, dtype=np.str_, delimiter=',')
    base_16_ufunc = get_base_16_ufunc()
    base_16_sbfp12 = base_16_ufunc(hex_sbfp12)
    # row 0 -- mamat scale exp for each col
    # row 1 - scales, row 2,3,4,5,6,7,8,9 int4s
    # scales row 1, 10, 19, 28
    scale_rows = [1, 10, 19, 28]
    scales_int = base_16_sbfp12[scale_rows, :]
    scales_int = scales_int.astype(np.uint8)
    int4_rows = np.delete(base_16_sbfp12, [0] + scale_rows, axis=0)
    int4_rows = int4_rows.astype(np.uint8)
    int4_odd_rows = np.bitwise_and(int4_rows, np.uint8(15 << 4))
    int4_odd_rows = np.bitwise_right_shift(int4_odd_rows, 4)
    int4_even_rows = np.bitwise_and(int4_rows, np.uint8(15))
    interleaved_int4 = np.empty( (int4_rows.shape[0] * 2, int4_rows.shape[1]), dtype=np.uint8)
    interleaved_int4[0::2, :] = int4_even_rows
    interleaved_int4[1::2, :] = int4_odd_rows
    return scales_int, interleaved_int4




def test():
    rng = np.random.default_rng()
    matrix = rng.standard_normal((2 ** 6, 2 ** 10), dtype=np.float32) 
    matrix = torch.from_numpy(matrix)
    start = time.time()
    mat_appr = convert_to_bfp16(matrix).to('cpu')
    end = time.time()
    print(f"bfp16: {end - start}")
    err = matrix - mat_appr
    err = torch.linalg.norm(err.flatten(), ord=1) 
    print(err)



if __name__ == '__main__':
    test()