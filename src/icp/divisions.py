"""
Division configuration loader for multi-division ICP scoring.

This module provides utilities to load and validate division configurations
that define how each division (Hardware, CRE, etc.) calculates its ICP scores.
"""
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import sys

ROOT = Path(__file__).resolve().parents[2]


class DivisionConfig:
    """Configuration for a single division's ICP scoring."""
    
    def __init__(self, config_dict: dict):
        self.name = config_dict.get("name")
        self.display_name = config_dict.get("display_name", self.name)
        self.source_goal = config_dict.get("source_goal")  # For CRE, maps "CAD" from source
        
        # Weights for the four ICP components
        self.weights = config_dict.get("weights", {})
        # Ensure size is always zero-weighted
        self.weights["size"] = 0.0
        
        # Goals that contribute to adoption score
        self.adoption_goals = config_dict.get("adoption_goals", [])
        
        # Goals that contribute to relationship score
        self.relationship_goals = config_dict.get("relationship_goals", [])
        
        # Flag indicating if relationship should reuse hardware adoption features
        self.relationship_uses_hardware_adoption = config_dict.get(
            "relationship_uses_hardware_adoption", False
        )
        
        # Industry weights file name
        self.industry_weights_file = config_dict.get("industry_weights_file", "industry_weights.json")
        
        # Validate required fields
        if not self.name:
            raise ValueError("Division config must have 'name' field")
        if not self.adoption_goals:
            raise ValueError(f"Division {self.name} must have at least one adoption_goal")
        if not self.relationship_goals and not self.relationship_uses_hardware_adoption:
            raise ValueError(
                f"Division {self.name} must have relationship_goals or "
                "relationship_uses_hardware_adoption=True"
            )
    
    def get_industry_weights_path(self) -> Path:
        """Return the full path to the industry weights file for this division."""
        return ROOT / "artifacts" / "weights" / self.industry_weights_file
    
    def __repr__(self):
        return f"DivisionConfig(name={self.name}, weights={self.weights})"


def load_division_config(division_name: str) -> DivisionConfig:
    """
    Load division configuration from JSON file.
    
    Args:
        division_name: Name of the division (e.g., "hardware", "cre")
        
    Returns:
        DivisionConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = ROOT / "artifacts" / "divisions" / f"{division_name.lower()}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Division config not found: {config_path}\n"
            f"Expected divisions: hardware, cre"
        )
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return DivisionConfig(config_dict)


def list_available_divisions() -> List[str]:
    """
    List all available division configuration files.
    
    Returns:
        List of division names (lowercase, without .json extension)
    """
    divisions_dir = ROOT / "artifacts" / "divisions"
    if not divisions_dir.exists():
        return []
    
    division_files = list(divisions_dir.glob("*.json"))
    return [f.stem.lower() for f in division_files]


def validate_division_config(config: DivisionConfig) -> Tuple[bool, List[str]]:
    """
    Validate a division configuration.
    
    Args:
        config: DivisionConfig to validate
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check weights sum to approximately 1.0 (allowing small rounding)
    weight_sum = sum(config.weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(
            f"Weights must sum to 1.0 (got {weight_sum:.3f}): {config.weights}"
        )
    
    # Check size weight is zero
    if config.weights.get("size", 0.0) != 0.0:
        errors.append("Size weight must be 0.0 for all divisions")
    
    # Check adoption goals exist
    if not config.adoption_goals:
        errors.append("Must have at least one adoption_goal")
    
    # Check relationship logic exists
    if not config.relationship_goals and not config.relationship_uses_hardware_adoption:
        errors.append(
            "Must have relationship_goals or relationship_uses_hardware_adoption=True"
        )
    
    return len(errors) == 0, errors
