"""VLM verification — mock adapter (PRD §10).

Produces the same structured evidence a real VLM returns (violence_visible,
physical_contact, weapon_visible, ...). The mock inspects clip motion +
detections + scene hint; on the A6000 this is replaced by Qwen2VLAdapter
served via vLLM. The final verdict always comes from the DecisionEngine,
never from raw VLM confidence (PRD G4).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from .schemas import CandidateEvent, VerificationMessage, VLMEvidence


class NarrativeGenerator:
    """Story text generator for camera feeds and incidents.

    Real model providers can be plugged in through environment variables while the
    default behavior remains a deterministic text summary for offline demo use.
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("VLM_NARRATIVE_PROVIDER", "mock")).lower()
        self.model = model or os.getenv("VLM_NARRATIVE_MODEL", "llama3.1")

    def generate(self, camera_id: str, scene_hint: str, detections: list[dict] | None = None,
                 incident: str | None = None) -> str:
        labels = [d.get("label", "object") for d in (detections or [])]
        if not labels:
            labels = ["person"] if "person" in (scene_hint or "").lower() else ["object"]
        details = ", ".join(sorted(set(labels))) if labels else "moving object"
        if incident:
            return f"{camera_id}: {incident} Detected objects: {details}."
        if "fight" in (scene_hint or "").lower() or "assault" in (scene_hint or "").lower():
            return f"{camera_id} shows a possible altercation. The feed includes {details} and elevated motion consistent with a fight or hostile interaction."
        if "knife" in (scene_hint or "").lower() or "weapon" in (scene_hint or "").lower():
            return f"{camera_id} shows a potential weapon threat. The scene includes {details}, with a suspicious object and aggressive movement." 
        if "run" in (scene_hint or "").lower() or "chase" in (scene_hint or "").lower():
            return f"{camera_id} shows rapid motion across the frame. The tracked objects include {details}, which may indicate a chase or sudden sprint."
        return f"{camera_id} remains generally calm. The camera tracks {details} with low-risk motion and no obvious threat signal."

    def _ollama_generate(self, prompt: str) -> str | None:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", self.model)
        payload = {"model": model, "prompt": prompt, "stream": False}
        req = urllib.request.Request(
            f"{base.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("response") or "").strip()
        except (urllib.error.URLError, ValueError, TimeoutError):
            return None

    def _openai_generate(self, prompt: str) -> str | None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        payload = {
            "model": os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if choices:
                    return (choices[0].get("message", {}).get("content") or "").strip()
        except (urllib.error.URLError, ValueError, TimeoutError):
            return None

    def narrative_for_camera(self, camera_id: str, scene_hint: str, detections: list[dict] | None = None) -> str:
        if self.provider == "ollama":
            prompt = (
                f"Write a short surveillance incident summary for camera {camera_id}. "
                f"Scene hint: {scene_hint or 'general monitoring'}. "
                f"Detected objects: {', '.join(sorted({d.get('label', 'object') for d in (detections or [])})) or 'unknown'}. "
                "Keep it concise and factual."
            )
            story = self._ollama_generate(prompt)
            if story:
                return story
        elif self.provider == "openai":
            prompt = (
                f"Write a short surveillance incident summary for camera {camera_id}. "
                f"Scene hint: {scene_hint or 'general monitoring'}. "
                f"Detected objects: {', '.join(sorted({d.get('label', 'object') for d in (detections or [])})) or 'unknown'}. "
                "Keep it concise and factual."
            )
            story = self._openai_generate(prompt)
            if story:
                return story
        return self.generate(camera_id, scene_hint, detections)


class BaseVLMVerifier:
    name = "base"

    def verify(self, clip_path: str, candidate: CandidateEvent,
               weapon_seen: bool, person_count: int,
               scene_hint: str = "") -> VerificationMessage:
        raise NotImplementedError


def clip_motion(clip_path: str, max_frames: int = 24) -> float:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return 0.0
    diffs, prev = [], None
    n = 0
    while n < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean() / 255.0))
        prev = g
        n += 1
    cap.release()
    return float(np.mean(diffs)) if diffs else 0.0


class MockVLMVerifier(BaseVLMVerifier):
    name = "mock-vlm-v1"

    def __init__(self, model_version: str = "mock-vlm-v1", latency_ms: int = 250):
        self.model_version = model_version
        self.latency_ms = latency_ms  # simulates GPU inference time

    def verify(self, clip_path: str, candidate: CandidateEvent,
               weapon_seen: bool, person_count: int,
               scene_hint: str = "") -> VerificationMessage:
        time.sleep(self.latency_ms / 1000.0)
        motion = clip_motion(clip_path)
        h = (scene_hint or "").lower()
        violent_hint = any(k in h for k in ("fight", "assault", "attack", "stab", "shoot", "weapon", "knife", "gun"))

        contact = (person_count >= 2 and (motion > 0.004 or candidate.anomaly_score >= 0.6)) or \
                  ("fight" in h and person_count >= 1)
        weapon_vis = bool(weapon_seen)
        victim = bool(contact or weapon_vis)
        multiple = person_count >= 2
        temporal = candidate.anomaly_score >= 0.55 and motion > 0.0025
        violence_visible = bool((contact and candidate.anomaly_score >= 0.5) or
                                (weapon_vis and candidate.anomaly_score >= 0.45) or
                                (violent_hint and candidate.anomaly_score >= 0.6))

        if weapon_vis and "shoot" in h:
            etype = "shooting"
        elif weapon_vis:
            etype = "stabbing"
        elif contact:
            etype = "fighting"
        elif violent_hint:
            etype = candidate.candidate_event
        else:
            etype = candidate.candidate_event

        parts = []
        parts.append(f"{person_count} person(s) visible" if person_count else "no clear person visible")
        parts.append("physical contact between individuals" if contact else "no clear physical contact")
        parts.append("a weapon-like object is visible" if weapon_vis else "no weapon visible")
        parts.append(f"sustained elevated motion (clip-motion={motion:.3f}, window-anomaly={candidate.anomaly_score:.2f})"
                     if temporal else "motion is unremarkable")
        desc = "Verification evidence: " + "; ".join(parts) + "."
        uncertain = (0.5 <= candidate.anomaly_score < 0.65) and not (contact or weapon_vis)

        return VerificationMessage(
            correlation_id=candidate.correlation_id, camera_id=candidate.camera_id,
            clip_id="", verdict="uncertain", event_type=etype, description=desc,
            evidence=VLMEvidence(physical_contact=bool(contact), weapon_visible=bool(weapon_vis),
                                 victim_present=bool(victim), multiple_people=bool(multiple),
                                 temporal_evidence=bool(temporal)),
            uncertain=bool(uncertain), model_version=self.model_version,
        )


class Qwen2VLAdapter(BaseVLMVerifier):
    """Placeholder for Qwen2-VL via vLLM on the A6000 (PRD §10, §35)."""

    name = "qwen2-vl-v1"

    def __init__(self, *a, **k):
        raise RuntimeError("Qwen2VLAdapter needs a GPU host. See docs/SWAP_TO_REAL_MODELS.md.")

    def verify(self, clip_path, candidate, weapon_seen, person_count, scene_hint=""):
        raise NotImplementedError
