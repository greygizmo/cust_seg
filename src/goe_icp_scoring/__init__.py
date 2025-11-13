"""Compatibility package exposing the scoring pipeline API."""

from icp.cli.score_accounts import (
    main,
    engineer_features,
    apply_industry_enrichment,
)

__all__ = ["main", "engineer_features", "apply_industry_enrichment"]
