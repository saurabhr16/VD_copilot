import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.incidents import IncidentManager
from src.schemas import CandidateEvent, VerificationMessage


def _cand(ws, we, ev="fighting"):
    return CandidateEvent(camera_id="cam02", window_start=ws, window_end=we,
                          anomaly_score=0.8, candidate_event=ev)


def _vlm(ev="fighting", verdict="confirmed"):
    return VerificationMessage(correlation_id="x", camera_id="cam02",
                               clip_id="clip", verdict=verdict,
                               event_type=ev, description="d", model_version="m")


def test_overlapping_windows_merge():
    m = IncidentManager(cooldown_seconds=30)
    a, is_new_a, _ = m.create_or_merge(_cand(0, 4), _vlm(), "clip1", "mv", "t1")
    b, is_new_b, _ = m.create_or_merge(_cand(2, 6), _vlm(), "clip2", "mv", "t1")
    assert is_new_a and not is_new_b and a.incident_id == b.incident_id
    assert (a.window_start, a.window_end) == (0, 6)
    assert len(a.candidate_ids) == 2


def test_distant_events_stay_separate():
    m = IncidentManager(cooldown_seconds=30)
    a, _, _ = m.create_or_merge(_cand(0, 4), _vlm(), "clip1", "mv", "t1")
    b, is_new, _ = m.create_or_merge(_cand(100, 104), _vlm(), "clip2", "mv", "t1")
    assert is_new and a.incident_id != b.incident_id
