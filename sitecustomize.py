"""Ad-hoc path bootstrapper for local runs.

Python imports ``sitecustomize`` automatically if present. This small module
adds the repository root and ``src/`` directory to ``sys.path`` so that:

- ``import icp`` works without installing the package, and
- scripts like ``goe_icp_scoring.py`` can be run directly from the repo root.

The test suite also adds these paths explicitly in ``tests/conftest.py``, so
this file is *not strictly required* for tests to pass. It can be removed if
you prefer to manage ``PYTHONPATH`` or install the package into a virtualenv.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SRC.exists():
    src_str = str(_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
