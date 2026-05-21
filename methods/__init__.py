"""Method registry. `make_method(name)` returns a QuantMethod instance."""

from .awq import AWQ
from .base import QuantMethod
from .dualquant import Dualquant
from .gptq import GPTQMethod
from .gptq_seq import GPTQSequential
from .rtn import RTN
from .sinq import SINQ

_REGISTRY = {
    RTN.name: RTN,
    Dualquant.name: Dualquant,
    AWQ.name: AWQ,
    GPTQMethod.name: GPTQMethod,
    GPTQSequential.name: GPTQSequential,
    SINQ.name: SINQ,
}


def make_method(name: str) -> QuantMethod:
    if name not in _REGISTRY:
        raise ValueError(f"unknown method {name!r}; expected one of {list(_REGISTRY)}")
    return _REGISTRY[name]()


__all__ = ["QuantMethod", "make_method"]
