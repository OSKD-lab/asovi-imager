"""Deprecated shim — module renamed to :mod:`asvimg.preprocess`.

Keeps ``from asvimg.pipeline01 import Pipeline01Runner`` and the
``asovi-pipeline01`` entry point working while emitting a warning.
"""

from __future__ import annotations

import warnings

from .preprocess import (  # noqa: F401
    PreprocessRunner,
    build_arg_parser,
    main,
    run_from_args,
)

warnings.warn(
    "asvimg.pipeline01 is renamed to asvimg.preprocess; "
    "update imports to 'from asvimg.preprocess import PreprocessRunner'.",
    DeprecationWarning,
    stacklevel=2,
)

Pipeline01Runner = PreprocessRunner  # deprecated alias


__all__ = [
    "Pipeline01Runner",
    "PreprocessRunner",
    "build_arg_parser",
    "main",
    "run_from_args",
]
