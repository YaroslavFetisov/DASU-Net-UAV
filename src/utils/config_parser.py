"""
YAML configuration loader with merge support.

Usage
-----
    cfg = load_config("configs/base.yaml")
    cfg = load_config("configs/base.yaml", "configs/exp_swin.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    base_path: str | Path,
    override_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load a YAML config, optionally merging an experiment override.

    Parameters
    ----------
    base_path     : Path to the base config (e.g. configs/base.yaml).
    override_path : Path to experiment config that overrides base values.

    Returns
    -------
    Merged configuration dictionary.
    """
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if override_path is not None:
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        cfg = _deep_merge(cfg, override)

    return cfg


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """
    Access nested config value using dot notation.

    Example: get(cfg, "model.encoder_type") -> "swin"
    """
    keys = dotted_key.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val
