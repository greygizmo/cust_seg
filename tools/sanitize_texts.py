"""
Sanitize Streamlit UI file and documentation by removing non-ASCII artifacts
introduced by encoding issues (e.g., stray sequences like 'dY…').

Rules:
- For .py and .md files in targets, remove any non-ASCII characters.
- Collapse multiple spaces.
- Apply targeted fixes for common broken tokens.

Run:
  python tools/sanitize_texts.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path


TARGETS = [
    # Streamlit app
    Path("apps") / "streamlit" / "app.py",
    # Project documentation
    Path("docs"),
]


def clean_text(s: str) -> str:
    """Lightweight cleanup of historical encoding artefacts in text files.

    This is intentionally conservative: it avoids stripping all non-ASCII
    characters and instead focuses on a few known bad sequences.
    """
    # Targeted token cleanup (keep spacing intact)
    cleaned = s.replace("dYZ_", "").replace("dY", "")
    return cleaned


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".py", ".md"}:
            yield p


def main():
    changed = []
    for target in TARGETS:
        for f in iter_files(target):
            try:
                orig = f.read_text(encoding="utf-8", errors="ignore")
                cleaned = clean_text(orig)
                if cleaned != orig:
                    f.write_text(cleaned, encoding="utf-8")
                    changed.append(str(f))
            except Exception:
                pass
    if changed:
        print("Sanitized files:")
        for c in changed:
            print(" -", c)
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
