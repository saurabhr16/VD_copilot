"""PRD §16 data contracts + internal event models.

These pydantic models are the plug-and-play contract (PRD §35): a service may
be replaced (mock <-> real model) as long as it preserves these schemas.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "v1"

Verdict = Literal["confirmed", "rejected", "uncertain", "insufficient_evidence"]
OperatorDecision = Literal["confirmed", "false_positive", "uncertain", "escalated"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- detection
class DetectionItem(BaseModel):
    track_id: str | None = None
    label: str  # person | knife | gun | bag | vehicle | ...
    bbox: list[float]  # [x1, y1, x2, y2] in pixels
    confidence: float
    occluded: bool = False
    suspect: bool = False


class DetectionMessage(BaseModel):
    schema_version: str = SCHEMA_VERSION
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    correlation_id: str = Field(default_factory=lambda: new_id("corr"))
    source_service: str = "detection"
    camera_id: str
    timestamp: str
    frame_id: str
    model_version: str
    detections: list[DetectionItem] = []


# ------------------------------------------------------------------ anomaly
class AnomalyMessage(BaseModel):
    schema_version: str = SCHEMA_VERSION
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    correlation_id: str
    source_service: str = "temporal-analysis"
    camera_id: str
    window_start: str
    window_end: str
    model_version: str
    anomaly_score: float
    candidate_event: str | None = None


# ---------------------------------------------------------------- candidate
class CandidateEvent(BaseModel):
    candidate_id: str = Field(default_factory=lambda: new_id("cand"))
    correlation_id: str = Field(default_factory=lambda: new_id("corr"))
    camera_id: str
    window_start: float  # seconds into the video
    window_end: float
    anomaly_score: float
    person_count: int = 0
    weapon_seen: bool = False
    signals: list[str] = []  # which signals fired, e.g. ["anomaly", "weapon", "crowd"]
    candidate_event: str = "assault"
    priority: float = 0.0  # higher = more urgent in the VLM queue
    created_at: str = Field(default_factory=utcnow)


# ------------------------------------------------------------- verification
class VLMEvidence(BaseModel):
    physical_contact: bool = False
    weapon_visible: bool = False
    victim_present: bool = False
    multiple_people: bool = False
    temporal_evidence: bool = False


class VerificationMessage(BaseModel):
    schema_version: str = SCHEMA_VERSION
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    correlation_id: str
    source_service: str = "vlm-verification"
    incident_id: str | None = None
    camera_id: str
    clip_id: str
    verdict: Verdict
    event_type: str | None = None
    description: str = ""
    evidence: VLMEvidence = Field(default_factory=VLMEvidence)
    uncertain: bool = False
    model_version: str
    timestamp: str = Field(default_factory=utcnow)
    rationale: str = ""  # decision-engine rule that produced the verdict


# ----------------------------------------------------------------- incident
class IncidentRecord(BaseModel):
    incident_id: str = Field(default_factory=lambda: new_id("incident"))
    camera_id: str
    status: Verdict = "uncertain"
    event_type: str | None = None
    window_start: float = 0.0  # merged span across all member candidates
    window_end: float = 0.0
    clip_id: str | None = None
    candidate_ids: list[str] = []
    anomaly_scores: list[float] = []
    vlm_descriptions: list[str] = []
    operator_decision: OperatorDecision | None = None
    model_version: str = ""
    threshold_version: str = ""
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)


# ------------------------------------------------------------------ search
class SearchHit(BaseModel):
    clip_id: str
    incident_id: str | None = None
    camera_id: str
    score: float
    event_type: str | None = None
    description: str = ""
    window_start: float = 0.0
    window_end: float = 0.0
