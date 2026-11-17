"""The ``falcon:`` section of config.yaml, and the shipped falcon-soccer preset.

config.yaml (and every preset) may carry a ``falcon:`` section. The desktop
window and the sidecar flatten it into the run's settings with
:func:`gui_settings`; the keys are documented in
:mod:`modules.falcon.pipeline_hooks`.
"""
from __future__ import annotations

import os

import yaml

PRESET_NAME = "falcon-soccer"
PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "falcon-soccer.yaml")

#: ``falcon:`` section key -> run setting.
SECTION_KEYS = {
    "marks": "falcon_marks",
    "mark_path": "falcon_mark_path",
    "parent_tap_points": "parent_tap_points",
    "mode": "falcon_mode",
    "pad_before": "falcon_pad_before",
    "pad_after": "falcon_pad_after",
    "merge_gap": "falcon_merge_gap",
    "camera_compensation": "camera_compensation",
    "analysis_proxy": "analysis_proxy",
}


def gui_settings(config_data: dict | None) -> dict:
    """Flat run settings from a config mapping's ``falcon:`` section."""
    section = (config_data or {}).get("falcon") or {}
    if not isinstance(section, dict):
        return {}
    out = {}
    for key, setting in SECTION_KEYS.items():
        if key in section and section[key] is not None:
            out[setting] = section[key]
    return out


def load_shipped() -> dict:
    with open(PRESET_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def install(config_file: str | None = None, *, overwrite: bool = False) -> str:
    """Copy the shipped preset into the user's presets folder. Returns its path.

    Refuses to replace an existing ``falcon-soccer`` preset unless asked: it
    may hold the parent's own tweaks.
    """
    from modules.system.presets import preset_path, save_preset

    path = preset_path(PRESET_NAME, config_file)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists")
    return save_preset(PRESET_NAME, load_shipped(), config_file)
