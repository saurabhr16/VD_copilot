"""Incident manager (PRD §11): merges overlapping candidate windows into one
incident so a single real-world event never produces repeated alerts.
"""
from __future__ import annotations

from .schemas import CandidateEvent, IncidentRecord, VerificationMessage, utcnow


class IncidentManager:
    def __init__(self, cooldown_seconds: float = 30.0):
        self.cooldown = cooldown_seconds
        self._open: dict[str, IncidentRecord] = {}  # camera_id -> latest incident

    def create_or_merge(self, cand: CandidateEvent, vlm: VerificationMessage,
                        clip_id: str | None, model_version: str,
                        threshold_version: str) -> tuple[IncidentRecord, bool, bool]:
        """Returns (incident, is_new, escalated_to_confirmed)."""
        prev = self._open.get(cand.camera_id)
        if prev is not None and self._same_event(prev, cand):
            prev_status = prev.status
            prev.window_start = min(prev.window_start, cand.window_start)
            prev.window_end = max(prev.window_end, cand.window_end)
            prev.candidate_ids.append(cand.candidate_id)
            prev.anomaly_scores.append(cand.anomaly_score)
            prev.vlm_descriptions.append(vlm.description)
            # Escalate status: confirmed wins, then uncertain, keep rejected lowest.
            rank = {"rejected": 0, "insufficient_evidence": 1, "uncertain": 2, "confirmed": 3}
            if rank.get(vlm.verdict, 0) > rank.get(prev.status, 0):
                prev.status = vlm.verdict
                prev.event_type = vlm.event_type or prev.event_type
                if clip_id:
                    prev.clip_id = clip_id
            prev.updated_at = utcnow()
            escalated = prev_status != "confirmed" and prev.status == "confirmed"
            return prev, False, escalated

        inc = IncidentRecord(
            camera_id=cand.camera_id, status=vlm.verdict,
            event_type=vlm.event_type or cand.candidate_event,
            window_start=cand.window_start, window_end=cand.window_end,
            clip_id=clip_id, candidate_ids=[cand.candidate_id],
            anomaly_scores=[cand.anomaly_score], vlm_descriptions=[vlm.description],
            model_version=model_version, threshold_version=threshold_version,
        )
        vlm.incident_id = inc.incident_id
        self._open[cand.camera_id] = inc
        return inc, True, False

    def _same_event(self, prev: IncidentRecord, cand: CandidateEvent) -> bool:
        overlap = cand.window_start <= prev.window_end + 1e-6 and prev.window_start <= cand.window_end + 1e-6
        gap = cand.window_start - prev.window_end
        within_cooldown = 0 <= gap <= self.cooldown
        same_type = (cand.candidate_event == prev.event_type or
                     {cand.candidate_event, prev.event_type or ""} <= {"fighting", "assault"})
        return (overlap or within_cooldown) and same_type

    def reset(self):
        self._open = {}
