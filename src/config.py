"""
config.py
---------
Load and expose the project configuration and column mapping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk upward from this file until we find base_config.yaml."""
    here = Path(__file__).resolve().parent
    for candidate in [here.parent, here.parent.parent]:
        cfg = candidate / "config" / "base_config.yaml"
        if cfg.exists():
            return candidate
    # Fallback: current working directory
    return Path(os.getcwd())


PROJECT_ROOT: Path = _find_project_root()
CONFIG_DIR: Path = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_base_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the parsed base_config.yaml as a nested dict."""
    if path is None:
        path = CONFIG_DIR / "base_config.yaml"
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_column_mapping(path: str | Path | None = None) -> dict[str, Any]:
    """Return the column mapping from column_mapping.yaml."""
    if path is None:
        path = CONFIG_DIR / "column_mapping.yaml"
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return raw["mapping"]


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

class Config:
    """Thin wrapper around base_config.yaml providing attribute-style access."""

    def __init__(self, path: str | Path | None = None):
        self._data = load_base_config(path)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ---- frequently accessed sub-sections ---------------------------------

    @property
    def periods(self) -> dict:
        return self._data["periods"]

    @property
    def rolling_windows(self) -> dict:
        return self._data["rolling_windows"]

    @property
    def data(self) -> dict:
        return self._data["data"]

    @property
    def covariance_transform(self) -> dict:
        return self._data["covariance_transform"]

    @property
    def model(self) -> dict:
        return self._data["model"]

    @property
    def training(self) -> dict:
        return self._data["training"]

    @property
    def generation(self) -> dict:
        return self._data["generation"]

    @property
    def portfolio(self) -> dict:
        return self._data["portfolio"]

    @property
    def validation(self) -> dict:
        return self._data["validation"]

    @property
    def benchmarks(self) -> dict:
        return self._data["benchmarks"]

    @property
    def ablations(self) -> dict:
        return self._data["ablations"]

    @property
    def random_seed(self) -> int:
        return int(self._data["project"]["random_seed"])

    @property
    def lookback_days(self) -> int:
        return int(self._data["rolling_windows"]["lookback_days"])

    @property
    def horizon_days(self) -> int:
        return int(self._data["rolling_windows"]["horizon_days"])

    @property
    def group_size(self) -> int:
        return int(self._data["industry"]["group_size"])

    @property
    def market_cap_top_n(self) -> int:
        return int(self._data["data"]["market_cap_top_n"])

    @property
    def ridge_epsilon(self) -> float:
        return float(self._data["covariance_transform"]["ridge_epsilon"])


# Module-level singleton (lazy)
_config_singleton: Config | None = None


def get_config() -> Config:
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = Config()
    return _config_singleton


def get_canonical_column(field: str) -> str:
    """
    Return the raw CRSP column name for a canonical field name.
    Raises KeyError if the field is not in column_mapping.yaml.
    """
    mapping = load_column_mapping()
    return mapping[field]["raw_name"]


def is_field_available(field: str) -> bool:
    """Return True if a canonical field is marked available in the column mapping."""
    mapping = load_column_mapping()
    entry = mapping.get(field, {})
    return bool(entry.get("available", False))
