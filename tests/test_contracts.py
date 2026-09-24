"""PRD §16 contract compliance: example payloads must validate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import (AnomalyMessage, DetectionMessage, VerificationMessage,
                         VLMEvidence)


def test_detection_contract():
    m = DetectionMessage(camera_id="cam_03", timestamp="2026-09-17T10:22:31Z",
                         frame_id="frame_123", model_version="detector_v1",
                         correlation_id="corr_1",
                         detections=[{"track_id": "track_21", "label": "person",
                                      "bbox": [1, 2, 3, 4], "confidence": 0.91},
                                     {"label": "knife", "bbox": [5, 6, 7, 8],
                                      "confidence": 0.78}])
    assert m.schema_version == "v1" and len(m.detections) == 2


def test_anomaly_contract():
    m = AnomalyMessage(correlation_id="corr_1", camera_id="cam_03",
                       window_start="2026-09-17T10:22:28Z",
                       window_end="2026-09-17T10:22:36Z",
                       model_version="anomaly_v1", anomaly_score=0.87,
                       candidate_event="assault")
    assert 0 <= m.anomaly_score <= 1


def test_verification_contract():
    m = VerificationMessage(correlation_id="corr_1", camera_id="cam_03",
                            clip_id="clip_1", verdict="confirmed",
                            event_type="assault", description="fight",
                            evidence=VLMEvidence(physical_contact=True),
                            model_version="vlm_v1")
    assert m.verdict in ("confirmed", "rejected", "uncertain", "insufficient_evidence")
