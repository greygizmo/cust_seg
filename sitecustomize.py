"""Test-time path bootstrapper.

Ensures the ``src`` directory is importable without installing the package.
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
