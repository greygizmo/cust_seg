"""
Lightweight data validation helpers for ICP scoring pipelines.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> Tuple[bool, list[str]]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


def ensure_non_negative(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[bool, list[str]]:
    bad = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if (s < 0).any():
                bad.append(c)
    return (len(bad) == 0, bad)


def log_validation(summary: str, details: list[str] | None = None, root: Path | None = None) -> Path:
    """Append a validation log entry under reports/logs/."""
    root = root or Path.cwd()
    out_dir = root / "reports" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"validation_{datetime.now().strftime('%Y%m%d')}.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {summary}\n")
        if details:
            for line in details:
                f.write(f"    - {line}\n")
    return path

