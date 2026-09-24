"""Candidate event generator (PRD §7).

Combines object detection + temporal anomaly + zone-agnostic rules. Critical
PRD principle: a missed person/weapon detection must NOT block the anomaly
path — high anomaly alone still fires a candidate.
"""
from __future__ import annotations

from .config import Config
from .schemas import CandidateEvent


def classify_candidate(scene_hint: str, person_count: int, weapon_seen: bool) -> str:
    h = (scene_hint or "").lower()
    if any(k in h for k in ("stab", "knife")):
        return "stabbing"
    if any(k in h for k in ("shoot", "gun")):
        return "shooting"
    if any(k in h for k in ("snatch", "chain")):
        return "chain_snatching"
    if any(k in h for k in ("fight",)):
        return "fighting"
    if weapon_seen:
        return "stabbing"
    if person_count >= 2:
        return "fighting"
    return "assault"


class CandidateGenerator:
    def __init__(self, cfg: Config):
        t = cfg.thresholds
        self.anomaly_thr = float(t.get("anomaly_candidate", 0.55))
        self.high_thr = float(t.get("anomaly_high", 0.75))
        self.boost_thr = float(t.get("anomaly_person_boost", 0.45))
        self.persistence = int(cfg.pipeline.get("persistence_windows", 1))
        self._streak: dict[str, int] = {}

    def consider(self, camera_id: str, ws: float, we: float, anomaly: float,
                 person_count: int, weapon_seen: bool, scene_hint: str = "",
                 correlation_id: str | None = None) -> CandidateEvent | None:
        """person_count = max simultaneous persons in the window (not a sum)."""
        persons, weapon = person_count, weapon_seen

        signals: list[str] = []
        if anomaly >= self.anomaly_thr:
            signals.append("anomaly")
        if weapon:
            signals.append("weapon")
        if persons >= 2 and anomaly >= self.boost_thr:
            signals.append("crowd_motion")
        if persons >= 1 and anomaly >= self.high_thr:
            signals.append("person_high_motion")

        if not signals:  # anomaly-only path lives above; nothing fired
            self._streak[camera_id] = 0
            return None

        self._streak[camera_id] = self._streak.get(camera_id, 0) + 1
        if self._streak[camera_id] < self.persistence:
            return None

        priority = anomaly + (0.15 if weapon else 0.0) + (0.05 if persons >= 2 else 0.0)
        return CandidateEvent(
            camera_id=camera_id, window_start=round(ws, 2), window_end=round(we, 2),
            anomaly_score=round(anomaly, 3), person_count=persons, weapon_seen=weapon,
            signals=signals, candidate_event=classify_candidate(scene_hint, persons, weapon),
            priority=round(min(1.2, priority), 3),
            correlation_id=correlation_id or f"corr_{camera_id}_{ws:.0f}",
        )

    def reset(self, camera_id: str):
        self._streak[camera_id] = 0
