"""Central configuration loader (PRD §31).

Reads config.yaml, applies environment overrides, and exposes a typed object.
Thresholds are versioned: `threshold_version` is stamped on every incident.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, raw: dict):
        self.raw = raw
        self.system = raw.get("system", {})
        self.pipeline = raw.get("pipeline", {})
        self.thresholds = raw.get("thresholds", {})
        self.recording = raw.get("recording", {})
        self.vlm_queue = raw.get("vlm_queue", {})
        self.retention = raw.get("retention", {})
        self.defaults = raw.get("defaults", {})
        self.cameras: dict[str, dict] = raw.get("cameras", {}) or {}

    @property
    def threshold_version(self) -> str:
        return self.system.get("threshold_version", "t1.0")

    @property
    def model_versions(self) -> dict:
        return self.system.get("model_versions", {})

    def camera_config(self, camera_id: str) -> dict[str, Any]:
        """Defaults deep-merged with per-camera overrides."""
        base = {
            "fps": self.defaults.get("fps", 10),
            "detection_threshold": self.thresholds.get("person_conf", 0.5),
            "anomaly_threshold": self.thresholds.get("anomaly_candidate", 0.55),
            "enabled_event_types": list(self.defaults.get("enabled_event_types", [])),
            "recording_pre_seconds": self.recording.get("pre_seconds", 5),
            "recording_post_seconds": self.recording.get("post_seconds", 5),
        }
        return _deep_merge(base, self.cameras.get(camera_id, {}))


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("VSD_CONFIG", ROOT / "config.yaml"))
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    # Env overrides for deploy flexibility
    if os.environ.get("VSD_THRESHOLD_VERSION"):
        raw.setdefault("system", {})["threshold_version"] = os.environ["VSD_THRESHOLD_VERSION"]
    if os.environ.get("DATABASE_URL"):
        raw["database_url"] = os.environ["DATABASE_URL"]
    if os.environ.get("REDIS_URL"):
        raw["redis_url"] = os.environ["REDIS_URL"]
    return Config(raw)


def database_url(cfg: Config) -> str:
    return cfg.raw.get("database_url", os.environ.get("DATABASE_URL", f"sqlite:///{ROOT}/data/meta.db"))


def redis_url(cfg: Config | None = None) -> str | None:
    if cfg is not None and cfg.raw.get("redis_url"):
        return cfg.raw["redis_url"]
    return os.environ.get("REDIS_URL")
