"""Add the in-repo `legacy_vendor/` to sys.path for transitional imports.

Usage: `import _legacy_path  # noqa: F401` at the top of any file that
needs to import unmigrated modules (torch_quant, eval_utils, datautils, awq,
sinkhorn, sinq_functions, layer_wrapper_baseline_data_formats_methods,
smoothquant.smooth_with_scale_dict).

Files in `legacy_vendor/` are byte-for-byte copies of the corresponding
files in the original `MSE_Reduction_Two_approache_All_DataFormats_20260410/
__Baselines_with_the_same_fils_as_MSE/` workspace, kept here so the
codebase is self-contained.
"""
import os
import sys

_LEGACY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "legacy_vendor")
)

if _LEGACY_DIR not in sys.path:
    sys.path.insert(0, _LEGACY_DIR)
