from .smoothquant import apply_smoothquant
from .quarot import apply_quarot


def apply_preprocess(name, model, preprocess_cfg):
    if name in ("none", None):
        return model
    if name == "smoothquant":
        return apply_smoothquant(model, preprocess_cfg)
    if name == "quarot":
        return apply_quarot(model, preprocess_cfg)
    raise ValueError(f"unknown preprocess {name!r}")


__all__ = ["apply_preprocess", "apply_smoothquant", "apply_quarot"]
