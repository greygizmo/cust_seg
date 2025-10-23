"""Compatibility module that exposes the public API for the scoring pipeline."""

from src.goe_icp_scoring import (  # type: ignore
    main,
    engineer_features,
    apply_industry_enrichment,
)

__all__ = ["main", "engineer_features", "apply_industry_enrichment"]


if __name__ == "__main__":  # pragma: no cover
    main()
