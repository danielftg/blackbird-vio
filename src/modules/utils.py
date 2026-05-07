"""
utils.py - Generic functionality 
"""
from pathlib import Path
import yaml




def load_yaml(path: Path) -> dict:
    """Load a YAML file with explicit UTF-8 encoding (Windows-safe)."""
    with open(path, "r" , encoding="utf-8") as f:
        return yaml.safe_load(f)
