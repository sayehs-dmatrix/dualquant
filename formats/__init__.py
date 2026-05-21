"""Format registry. `make_format(name, **fmt_cfg)` returns a FormatSpec."""

from .base import FormatSpec
from .mx import (
    make_mxfp4,
    make_mxfp8_e4m3,
    make_mxfp8_e5m2,
    make_mxint4,
    make_mxint8,
)
from .nvfp4 import make_nvfp4
from .rtn_int import make_rtn_int4, make_rtn_int8
from .sfp4 import make_sfp4

_FACTORIES = {
    "mxfp4": make_mxfp4,
    "mxint4": make_mxint4,
    "mxfp8_e4m3": make_mxfp8_e4m3,
    "mxfp8_e5m2": make_mxfp8_e5m2,
    "mxint8": make_mxint8,
    "nvfp4": make_nvfp4,
    "sfp4": make_sfp4,
    "rtn_int4": make_rtn_int4,
    "rtn_int8": make_rtn_int8,
}


def make_format(name: str, **fmt_cfg) -> FormatSpec:
    if name not in _FACTORIES:
        raise ValueError(f"unknown format {name!r}; expected one of {list(_FACTORIES)}")
    factory = _FACTORIES[name]
    # nvfp4 / sfp4 take no kwargs (block_size is fixed); the rest accept block_size
    if name in ("nvfp4", "sfp4"):
        if fmt_cfg.get("block_size") not in (None, 16):
            raise ValueError(f"{name} block_size is fixed at 16 (got {fmt_cfg.get('block_size')})")
        return factory()
    return factory(**fmt_cfg)


__all__ = ["FormatSpec", "make_format"]
