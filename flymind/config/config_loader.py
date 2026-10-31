"""
Configuration loader for FlyMind.
Loads and validates simulation, bio-perception, and safety configuration YAML files.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_DIR = Path(__file__).parent.resolve()


def load_yaml(file_name: str) -> Dict[str, Any]:
    """Loads a YAML file from the config directory."""
    path = CONFIG_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_simulation_config() -> Dict[str, Any]:
    """Returns the simulation physical and sensor configuration."""
    return load_yaml("simulation_params.yaml")


def load_bio_perception_config() -> Dict[str, Any]:
    """Returns the bio-inspired perception and neural model configuration."""
    return load_yaml("bio_perception.yaml")


def load_safety_config() -> Dict[str, Any]:
    """Returns the safety supervisor limits and geofence parameters."""
    return load_yaml("safety_limits.yaml")


def load_full_config() -> Dict[str, Any]:
    """Loads all configuration namespaces into a single unified dictionary."""
    return {
        "simulation": load_simulation_config(),
        "bio_perception": load_bio_perception_config(),
        "safety": load_safety_config(),
    }
