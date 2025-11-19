"""Division configuration and helpers for multi-ICP scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping
import json


@dataclass(frozen=True)
class AdoptionConfig:
    """Configuration describing how to derive adoption signals for a division."""

    asset_column: str | None = None
    profit_column: str | None = None
    asset_goals: tuple[str, ...] = ()
    profit_goals: tuple[str, ...] = ()
    fallback_printer_columns: tuple[str, ...] = ()
    fallback_revenue_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationshipConfig:
    """Configuration describing relationship/engagement signals for a division."""

    profit_column: str | None = None
    profit_goals: tuple[str, ...] = ()
    revenue_fallback_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DivisionConfig:
    """Runtime configuration for an ICP scoring division."""

    key: str
    label: str
    super_division: str
    industry_weights_file: Path
    component_weights: Mapping[str, float]
    size_revenue_column: str | None
    size_revenue_fallback: str | None
    adoption: AdoptionConfig
    relationship: RelationshipConfig
    performance_columns: tuple[str, ...]
    neutral_vertical_score: float = 0.30

    def weight_dict(self) -> Dict[str, float]:
        return dict(self.component_weights)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


_BASE_CONFIGS: Dict[str, DivisionConfig] = {
    "hardware": DivisionConfig(
        key="hardware",
        label="Hardware",
        super_division="Hardware",
        industry_weights_file=_root() / "artifacts" / "weights" / "hardware_industry_weights.json",
        component_weights={"vertical": 0.30, "adoption": 0.50, "relationship": 0.20},
        size_revenue_column=None,
        size_revenue_fallback=None,
        adoption=AdoptionConfig(
            asset_column="adoption_assets",
            profit_column="adoption_profit",
            asset_goals=("Printers", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"),
            profit_goals=("Printers", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"),
            fallback_printer_columns=("Big Box Count", "Small Box Count"),
            fallback_revenue_columns=("Total Hardware Revenue", "Total Consumable Revenue"),
        ),
        relationship=RelationshipConfig(
            profit_column="relationship_profit",
            profit_goals=("CAD", "CPE", "Specialty Software"),
            revenue_fallback_columns=(
                "Total Maintenance Revenue",
            ),
        ),
        performance_columns=("Printers", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"),
        neutral_vertical_score=0.30,
    ),
    "cre": DivisionConfig(
        key="cre",
        label="CRE",
        super_division="Software",
        industry_weights_file=_root() / "artifacts" / "weights" / "cre_industry_weights.json",
        component_weights={"vertical": 0.25, "adoption": 0.45, "relationship": 0.30},
        size_revenue_column=None,
        size_revenue_fallback=None,
        adoption=AdoptionConfig(
            asset_column="cre_adoption_assets",
            profit_column="cre_adoption_profit",
            # CRE adoption considers CAD and Specialty Software divisions
            asset_goals=("CAD", "Specialty Software"),
            profit_goals=("CAD", "Specialty Software"),
            fallback_revenue_columns=("CAD", "Specialty Software"),
        ),
        relationship=RelationshipConfig(
            profit_column="cre_relationship_profit",
            profit_goals=("Specialty Software", "Training/Services"),
            revenue_fallback_columns=(
                "Total Software License Revenue",
                "Total SaaS Revenue",
                "Total Maintenance Revenue",
            ),
        ),
        # CRE industry performance considers CAD, Specialty Software, and
        # Training/Services restricted to CRE-specific rollups (set upstream as CRE_Training)
        performance_columns=("CAD", "Specialty Software", "CRE_Training"),
        neutral_vertical_score=0.35,
    ),
    "cpe": DivisionConfig(
        key="cpe",
        label="CPE",
        super_division="Software",
        industry_weights_file=_root() / "artifacts" / "weights" / "cpe_industry_weights.json",
        component_weights={"vertical": 0.25, "adoption": 0.45, "relationship": 0.30},
        size_revenue_column=None,
        size_revenue_fallback=None,
        adoption=AdoptionConfig(
            asset_column="cpe_adoption_assets",
            profit_column="cpe_adoption_profit",
            asset_goals=("CPE",),
            profit_goals=("CPE",),
            fallback_revenue_columns=("CPE",),
        ),
        relationship=RelationshipConfig(
            profit_column="cpe_relationship_profit",
            profit_goals=("CPE",),
            revenue_fallback_columns=("CPE",),
        ),
        performance_columns=("CPE",),
        neutral_vertical_score=0.35,
    ),
}


def available_divisions() -> tuple[str, ...]:
    """Return the canonical list of division keys supported by the runtime."""

    return tuple(sorted(_BASE_CONFIGS))


def _load_division_override(key: str) -> Mapping[str, Any]:
    path = _root() / "artifacts" / "divisions" / f"{key}.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, Mapping):
            return data
    except Exception:
        return {}
    return {}


def _deep_update(target: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = value
    return target


def get_division_config(key: str | None) -> DivisionConfig:
    """Resolve :class:`DivisionConfig` for the requested key."""

    if key is None:
        key = "hardware"
    norm = str(key).lower()
    if norm not in _BASE_CONFIGS:
        raise KeyError(f"Unknown division '{key}'. Available: {', '.join(available_divisions())}")

    base = _BASE_CONFIGS[norm]
    override = _load_division_override(norm)
    if not override:
        return base

    # Convert dataclass to nested dict for merging then reconstruct.
    data: Dict[str, Any] = {
        "key": base.key,
        "label": base.label,
        "super_division": base.super_division,
        "industry_weights_file": str(base.industry_weights_file),
        "component_weights": dict(base.component_weights),
        "size_revenue_column": base.size_revenue_column,
        "size_revenue_fallback": base.size_revenue_fallback,
        "adoption": {
            "asset_column": base.adoption.asset_column,
            "profit_column": base.adoption.profit_column,
            "asset_goals": list(base.adoption.asset_goals),
            "profit_goals": list(base.adoption.profit_goals),
            "fallback_printer_columns": list(base.adoption.fallback_printer_columns),
            "fallback_revenue_columns": list(base.adoption.fallback_revenue_columns),
        },
        "relationship": {
            "profit_column": base.relationship.profit_column,
            "profit_goals": list(base.relationship.profit_goals),
            "revenue_fallback_columns": list(base.relationship.revenue_fallback_columns),
        },
        "performance_columns": list(base.performance_columns),
        "neutral_vertical_score": base.neutral_vertical_score,
    }

    _deep_update(data, override)

    # Reconstruct dataclasses (cast lists back to tuples and path to Path).
    adoption_cfg = AdoptionConfig(
        asset_column=data["adoption"].get("asset_column"),
        profit_column=data["adoption"].get("profit_column"),
        asset_goals=tuple(data["adoption"].get("asset_goals", ())),
        profit_goals=tuple(data["adoption"].get("profit_goals", ())),
        fallback_printer_columns=tuple(data["adoption"].get("fallback_printer_columns", ())),
        fallback_revenue_columns=tuple(data["adoption"].get("fallback_revenue_columns", ())),
    )
    relationship_cfg = RelationshipConfig(
        profit_column=data["relationship"].get("profit_column"),
        profit_goals=tuple(data["relationship"].get("profit_goals", ())),
        revenue_fallback_columns=tuple(data["relationship"].get("revenue_fallback_columns", ())),
    )

    industry_path = Path(data["industry_weights_file"])

    return DivisionConfig(
        key=base.key,
        label=data.get("label", base.label),
        super_division=data.get("super_division", base.super_division),
        industry_weights_file=industry_path,
        component_weights=data.get("component_weights", base.component_weights),
        size_revenue_column=data.get("size_revenue_column"),
        size_revenue_fallback=data.get("size_revenue_fallback"),
        adoption=adoption_cfg,
        relationship=relationship_cfg,
        performance_columns=tuple(data.get("performance_columns", base.performance_columns)),
        neutral_vertical_score=float(data.get("neutral_vertical_score", base.neutral_vertical_score)),
    )
