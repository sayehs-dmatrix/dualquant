"""Block-format quantization backend (indirection layer).

This project depends on an external library providing a ``Format`` class with
``Format.from_shorthand(...)`` and ``.cast(tensor)``. That library is not
publicly redistributable, so it is resolved at import time from an environment
variable rather than being named in the source:

    export BLOCKFMT_BACKEND=<python.module.path>

The module named there must expose ``Format``.
"""

import importlib
import os

_BACKEND = os.environ.get("BLOCKFMT_BACKEND")

if not _BACKEND:
    raise ImportError(
        "Set BLOCKFMT_BACKEND to the import path of a module providing a "
        "`Format` class with .from_shorthand() and .cast(). See README.md."
    )

Format = importlib.import_module(_BACKEND).Format

__all__ = ["Format"]
