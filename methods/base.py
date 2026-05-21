"""Base class for quantisation methods.

Each method subclasses QuantMethod and implements `wrap(layer, cfg, calib_acts)`.

`cfg` is a small dict assembled by main.py from CLI flags + an optional
per-method JSON. Layout:

    cfg = {
        "method_cfg":           {...},           # from --method-cfg JSON; method-specific hyperparams
        "weight_fmt":           FormatSpec,
        "weight_scale_format":  "e8m0",          # 'e8m0'|'e4m3'|'e4m4'|'e5m3'|'none'
        "act_quant": {
            "enabled":              False,
            "fmt":                  FormatSpec or None,
            "scale_format":         "none",
            "scaled_before_quant":  False,
            "quantize_bmm":         False,
        },
    }
"""


class QuantMethod:
    name = ""
    needs_calib_acts = False    # True for GPTQ, AWQ; main.py will collect them

    def wrap(self, layer, cfg, calib_acts=None, layer_key=None):
        """Mutate layer.weight in place. Return per-column activation scale
        tensor when cfg['act_quant']['scaled_before_quant'] is True, else None.

        layer_key (e.g. "layer0.self_attn.q_proj") is passed by main.py for
        methods that want to key something by per-layer identity (currently
        only Dualquant for scale-saving). Most methods ignore it.
        """
        raise NotImplementedError
