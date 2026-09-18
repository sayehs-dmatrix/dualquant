"""Minimal build for the DualQuant rebuttal: only QuantizedGEMM (the
standalone W4A4 GEMM), skipping FluxModel/SanaModel/Block-Sparse-Attention
(flash-attention kernels) entirely to cut build time drastically.
"""
import os

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            ext.extra_compile_args.setdefault("cxx", [])
            ext.extra_compile_args.setdefault("nvcc", [])
        super().build_extensions()


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    INCLUDE_DIRS = [
        ROOT_DIR + "/src",
        ROOT_DIR + "/nunchaku/csrc",
        ROOT_DIR + "/third_party/cutlass/include",
        ROOT_DIR + "/third_party/json/include",
        ROOT_DIR + "/third_party/mio/include",
        ROOT_DIR + "/third_party/spdlog/include",
    ]

    GCC_FLAGS = ["-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-DNUNCHAKU_BF16_ONLY=1", "-fvisibility=hidden", "-std=c++20", "-O2"]
    NVCC_FLAGS = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-DNUNCHAKU_BF16_ONLY=1",
        "-gencode", "arch=compute_89,code=sm_89",
        "-std=c++20",
        "-Xcudafe", "--diag_suppress=20208",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--threads=4",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-O2",
    ]

    nunchaku_min_ext = CUDAExtension(
        name="nunchaku_min",
        sources=[
            "nunchaku/csrc/pybind_minimal.cpp",
            "nunchaku/csrc/fused_beta_mul.cu",
            "src/interop/torch.cpp",
            "src/activation.cpp",
            "src/layernorm.cpp",
            "src/Linear.cpp",
            "src/Serialization.cpp",
            "src/kernels/activation_kernels.cu",
            "src/kernels/layernorm_kernels.cu",
            "src/kernels/misc_kernels.cu",
            "src/kernels/zgemm/gemm_w4a4.cu",
                "src/kernels/zgemm/gemm_w4a4_launch_bf16.cu",
            "src/kernels/zgemm/gemm_w8a8.cu",
            "src/kernels/dwconv.cu",
            "src/kernels/gemm_batched.cu",
            "src/kernels/gemm_f16.cu",
            "src/kernels/awq/gemv_awq.cu",
        ],
        extra_compile_args={"cxx": GCC_FLAGS, "nvcc": NVCC_FLAGS},
        include_dirs=INCLUDE_DIRS,
    )

    setuptools.setup(
        name="nunchaku_min",
        version="0.0.1",
        ext_modules=[nunchaku_min_ext],
        cmdclass={"build_ext": CustomBuildExtension},
    )
