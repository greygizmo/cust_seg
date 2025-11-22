"""Configuration management for ICP scoring."""
import tomllib
from pathlib import Path
from dataclasses import dataclass, field

# Default paths
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.toml"

@dataclass
class ALSConfig:
    factors_rollup: int = 64
    factors_goal: int = 32
    w_rollup_vec: float = 1.0
    w_goal_vec: float = 1.0
    alpha: float = 40.0
    reg: float = 0.05
    iterations: int = 20
    use_bm25: bool = True

@dataclass
class SimilarityConfig:
    k_neighbors: int = 25
    use_text: bool = True
    use_als: bool = True
    max_dense_accounts: int = 5000
    row_block_size: int = 512
    w_numeric: float = 0.45
    w_categorical: float = 0.20
    w_text: float = 0.25
    w_als: float = 0.10
    numeric_include: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    log1p_cols: list[str] = field(default_factory=list)
    logit_cols: list[str] = field(default_factory=list)

@dataclass
class AppConfig:
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    # Add other sections as needed (e.g., database, paths)

def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Loads configuration from a TOML file."""
    if not path.exists():
        return AppConfig()

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
        
        sim_data = data.get("similarity", {})
        als_data = data.get("als", {})
        
        return AppConfig(
            similarity=SimilarityConfig(**sim_data),
            als=ALSConfig(**als_data)
        )
    except Exception as e:
        print(f"[WARN] Failed to load config from {path}: {e}. Using defaults.")
        return AppConfig()

# Global instance
config = load_config()
