# DualScale kernel benchmark (vendored)

Byte-for-byte copy of
`Quantization_Repo_July2025/MSE_Reduction_Two_approache_All_DataFormats_20260410/
__Baselines_with_the_same_fils_as_MSE/DualScale_Kernel_Benchmark/`
from the the monorepo workspace this repo was extracted from.

Vendored because `validate_w4a4.py`, `validate_w4a4_sweep.py`,
`validate_real_kernel.py` and `full_model_w4a4.py` do
`from fused_dual_scale_kernel import ...`, which would otherwise reach outside
the repository. Same rationale as `legacy_vendor/`: keep the codebase
self-contained.
