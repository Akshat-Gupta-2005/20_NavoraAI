import yaml
import os
from pathlib import Path
from typing import Optional

# Repo root, resolved from this file so the CWD does not matter.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
