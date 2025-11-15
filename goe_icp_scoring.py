"""Compatibility module that exposes the public API for the scoring pipeline.

Historically this script contained the end-to-end implementation. It now
delegates to the modern CLI in ``icp.cli.score_accounts`` while preserving
the original import surface for tests and legacy consumers.
"""

from icp.cli.score_accounts import (  # type: ignore
    main,
    engineer_features,
    apply_industry_enrichment,
)

__all__ = ["main", "engineer_features", "apply_industry_enrichment"]


if __name__ == "__main__":  # pragma: no cover
    main()
