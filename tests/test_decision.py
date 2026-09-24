import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision import DecisionEngine
from src.schemas import CandidateEvent, VerificationMessage, VLMEvidence

ENG = DecisionEngine()


def _cand(a=0.8):
    return CandidateEvent(camera_id="c", window_start=0, window_end=4,
                          anomaly_score=a, person_count=2, weapon_seen=False)


def _vlm(contact=False, weapon=False, multi=True, temporal=True, uncertain=False, a=0.8):
    return VerificationMessage(correlation_id="x", camera_id="c", clip_id="clip",
                               verdict="uncertain", event_type="fighting", description="d",
                               evidence=VLMEvidence(physical_contact=contact,
                                                    weapon_visible=weapon,
                                                    multiple_people=multi,
                                                    temporal_evidence=temporal),
                               uncertain=uncertain, model_version="m")


def test_confirmed_on_contact():
    v = ENG.decide(_cand(0.8), _vlm(contact=True), 8.0)
    assert v.verdict == "confirmed"


def test_confirmed_on_weapon():
    c = _cand(0.5)
    c.weapon_seen = True
    v = ENG.decide(c, _vlm(weapon=True), 8.0)
    assert v.verdict == "confirmed"


def test_uncertain_borderline():
    v = ENG.decide(_cand(0.6), _vlm(uncertain=True), 8.0)
    assert v.verdict == "uncertain"


def test_rejected_no_evidence():
    v = ENG.decide(_cand(0.4), _vlm(multi=False, temporal=False), 8.0)
    assert v.verdict == "rejected"


def test_insufficient_short_clip():
    v = ENG.decide(_cand(0.9), _vlm(contact=True), 0.5)
    assert v.verdict == "insufficient_evidence"
