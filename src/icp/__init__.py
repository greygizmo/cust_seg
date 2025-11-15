"""Top-level package for ICP scoring logic and utilities.

This module provides convenient imports for commonly used functionality
while keeping the main implementation in submodules under ``src/icp``.
"""

from . import scoring, optimization, divisions, schema, validation, data_access  # noqa: F401

__all__ = [
    "scoring",
    "optimization",
    "divisions",
    "schema",
    "validation",
    "data_access",
]

