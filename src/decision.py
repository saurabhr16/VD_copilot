"""Decision engine (PRD §10): configurable rules turn VLM *evidence* into a
verdict. VLM confidence alone never decides (PRD G4). Supports the explicit
`uncertain` / `insufficient_evidence` states (PRD §9).
"""
from __future__ import annotations

from .schemas import CandidateEvent, VerificationMessage


class DecisionEngine:
    def __init__(self, high: float = 0.75, mid: float = 0.55):
        self.high = high
        self.mid = mid

    def decide(self, cand: CandidateEvent, vlm: VerificationMessage,
               clip_seconds: float) -> VerificationMessage:
        e = vlm.evidence
        a = cand.anomaly_score
        vlm.clip_id = vlm.clip_id  # set by caller before/after; kept for clarity

        # Insufficient evidence: clip too short to judge.
        if clip_seconds < 1.0:
            vlm.verdict = "insufficient_evidence"
            vlm.rationale = "rule: clip < 1s -> insufficient_evidence"
            return vlm

        # CONFIRMED paths
        if e.weapon_visible and a >= 0.45 and e.temporal_evidence:
            vlm.verdict, vlm.rationale = "confirmed", "rule: weapon + anomaly>=0.45 + temporal"
            return vlm
        if e.physical_contact and a >= 0.5 and e.multiple_people:
            vlm.verdict, vlm.rationale = "confirmed", "rule: contact + anomaly>=0.5 + multi-person"
            return vlm
        if a >= self.high and (e.physical_contact or e.weapon_visible):
            vlm.verdict, vlm.rationale = "confirmed", f"rule: anomaly>={self.high} + contact/weapon"
            return vlm

        # UNCERTAIN: conflicting or borderline signals — needs a human (PRD §9).
        if vlm.uncertain or (self.mid <= a < self.high and not (e.physical_contact or e.weapon_visible)):
            vlm.verdict = "uncertain"
            vlm.rationale = "rule: borderline anomaly without contact/weapon -> human review"
            return vlm
        if e.weapon_visible != cand.weapon_seen and a >= self.mid:
            vlm.verdict = "uncertain"
            vlm.rationale = "rule: detector/VLM disagree on weapon -> human review"
            return vlm

        # REJECTED: nothing corroborates the candidate.
        vlm.verdict = "rejected"
        vlm.rationale = "rule: no corroborating evidence -> rejected"
        return vlm
