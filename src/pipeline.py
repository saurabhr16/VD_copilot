"""End-to-end pipeline orchestrator (PRD §32, file-input variant).

Per camera:
  ingest -> detect + anomaly (sliding windows) -> candidates
         -> VLM queue (priority, bounded) -> decision -> incident dedup
         -> clip record + metadata + embeddings -> alert

Every stage publishes its PRD §34 message to the broker for traceability,
even though the prototype runs in-process. Real deployment moves each stage
behind a broker consumer without changing stage code.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .broker import Broker
from .candidate import CandidateGenerator
from .clips import ClipRecorder
from .config import Config
from .decision import DecisionEngine
from .embeddings import MockEmbeddingModel
from .incidents import IncidentManager
from .ingestion import RollingBuffer, VideoIngestion
from .anomaly_mock import MockAnomalyScorer
from .detection_mock import MockDetector
from .metrics import Metrics
from .schemas import (AnomalyMessage, CandidateEvent, DetectionMessage,
                      IncidentRecord, VerificationMessage, new_id, utcnow)
from .store import ClipRow, IncidentRow, MetadataDB, VectorStore


class Pipeline:
    def __init__(self, cfg: Config, db: MetadataDB, broker: Broker,
                 vectors: VectorStore, metrics: Metrics, clips_dir: str | Path):
        self.cfg = cfg
        self.db, self.broker, self.vectors, self.metrics = db, broker, vectors, metrics
        mv = cfg.model_versions
        self.det = MockDetector(person_conf=float(cfg.thresholds.get("person_conf", 0.5)),
                                weapon_conf=float(cfg.thresholds.get("weapon_conf", 0.4)))
        self.anom = MockAnomalyScorer(smoothing_alpha=float(cfg.pipeline.get("smoothing_alpha", 0.4)))
        self.gen = CandidateGenerator(cfg)
        self.dec = DecisionEngine(high=float(cfg.thresholds.get("anomaly_high", 0.75)),
                                  mid=float(cfg.thresholds.get("anomaly_candidate", 0.55)))
        self.incmgr = IncidentManager(cooldown_seconds=float(cfg.pipeline.get("cooldown_seconds", 30)))
        ccfg = cfg.recording
        self.clips = ClipRecorder(clips_dir, pre_s=float(ccfg.get("pre_seconds", 5)),
                                  post_s=float(ccfg.get("post_seconds", 5)))
        from .vlm_mock import MockVLMVerifier  # local import: keeps module light
        self.vlm = MockVLMVerifier(model_version=mv.get("vlm", "mock-vlm-v1"))
        self.emb = MockEmbeddingModel()
        self.model_version = f"{mv.get('detector')}+{mv.get('anomaly')}+{mv.get('vlm')}"

    # ------------------------------------------------------------------ main
    def process_video(self, camera_id: str, source: str, scene_hint: str = "") -> list[IncidentRecord]:
        t_start = time.perf_counter()
        p = self.cfg.pipeline
        win, stride = float(p.get("window_seconds", 4)), float(p.get("stride_seconds", 2))
        det_fps = float(p.get("detection_fps", 5))

        self.det.reset(scene_hint)
        self.anom.reset(scene_hint)
        self.gen.reset(camera_id)
        ing = VideoIngestion(source, target_fps=det_fps)
        buf = RollingBuffer(max_seconds=win + float(self.cfg.recording.get("pre_seconds", 5)))

        sampled: list[tuple[float, object, list]] = []  # (t, image, detections)
        for fr in ing.iter_frames():
            with self.metrics.timer("stage.detect"):
                dets = self.det.detect(fr.image)
            buf.append(fr)
            sampled.append((fr.t, fr.image, dets))
            self.metrics.inc("frames_processed")
            self.metrics.inc("persons_detected", sum(1 for d in dets if d.label == "person"))
            if any(d.label in ("knife", "gun") for d in dets):
                self.metrics.inc("weapons_detected")
            if len(sampled) % 10 == 0:  # trace a subset to the broker (full rate in prod)
                self.broker.publish("detections", DetectionMessage(
                    camera_id=camera_id, timestamp=utcnow(), frame_id=f"f_{fr.t:.1f}s",
                    model_version=self.cfg.model_versions.get("detector", ""),
                    detections=dets, correlation_id=new_id("corr")).model_dump())

        # Sliding windows over sampled frames.
        candidates: list[CandidateEvent] = []
        t = 0.0
        dur = sampled[-1][0] + 0.2 if sampled else 0.0
        while t < dur:
            ws, we = t, min(t + win, dur + 0.2)
            wframes = [(tt, im, dd) for tt, im, dd in sampled if ws <= tt < we]
            if not wframes:
                t += stride
                continue
            with self.metrics.timer("stage.anomaly"):
                score = self.anom.score_window([im for _, im, _ in wframes],
                                               [dd for _, _, dd in wframes])
            per_frame_persons = [sum(1 for d in dd if d.label == "person")
                                 for _, _, dd in wframes]
            max_persons = max(per_frame_persons) if per_frame_persons else 0
            any_weapon = any(d.label in ("knife", "gun")
                             for _, _, dd in wframes for d in dd)
            corr = new_id("corr")
            self.broker.publish("anomalies", AnomalyMessage(
                correlation_id=corr, camera_id=camera_id,
                window_start=f"{ws:.2f}s", window_end=f"{we:.2f}s",
                model_version=self.cfg.model_versions.get("anomaly", ""),
                anomaly_score=score).model_dump())
            band = ("anomaly_band.high" if score >= self.dec.high else
                    "anomaly_band.mid" if score >= self.dec.mid else "anomaly_band.low")
            self.metrics.inc(band)
            with self.metrics.timer("stage.candidate"):
                cand = self.gen.consider(camera_id, ws, we, score, max_persons,
                                          any_weapon, scene_hint, corr)
            if cand is not None:
                candidates.append(cand)
                self.broker.publish("candidates", cand.model_dump())
                self.metrics.inc("candidates")
            t += stride

        # VLM verification through the bounded priority queue (PRD §26).
        maxc = int(self.cfg.vlm_queue.get("max_concurrent", 2))
        ttl = int(self.cfg.vlm_queue.get("ttl_seconds", 120))
        maxd = int(self.cfg.vlm_queue.get("max_depth", 50))
        for c in candidates:
            if not self.broker.vlm_enqueue({**c.model_dump(), "scene_hint": scene_hint,
                                             "source": source}, max_depth=maxd):
                self.metrics.inc("vlm_dropped_full")

        jobs = []
        while (job := self.broker.vlm_dequeue(ttl_seconds=ttl)) is not None:
            jobs.append(job)
        self.metrics.inc("vlm_jobs", len(jobs))

        incidents: list[IncidentRecord] = []
        if jobs:
            with ThreadPoolExecutor(max_workers=max(1, maxc)) as ex:
                verifs = list(ex.map(lambda j: self._verify_job(camera_id, j), jobs))
            verifs.sort(key=lambda v: v[1].window_start)
            for vlm_msg, cand, clip in verifs:
                # Persist clip row + embedding + incident row.
                with self.db.session() as s:
                    if clip is not None:
                        clip_id, clip_path, cs, ce = clip
                        if s.get(ClipRow, clip_id) is None:
                            s.add(ClipRow(clip_id=clip_id, camera_id=camera_id,
                                          path=clip_path, window_start=cs, window_end=ce,
                                          description=vlm_msg.description,
                                          event_type=vlm_msg.event_type or ""))
                    inc, is_new, escalated = self.incmgr.create_or_merge(
                        cand, vlm_msg, clip[0] if clip else None,
                        self.model_version, self.cfg.threshold_version)
                    row = s.get(IncidentRow, inc.incident_id)
                    data = inc.model_dump()
                    if row is None:
                        s.add(IncidentRow(**data))
                    else:
                        for k, v in data.items():
                            setattr(row, k, v)
                    s.commit()
                if clip is not None and vlm_msg.incident_id:
                    with self.db.session() as s:
                        r = s.get(ClipRow, clip[0])
                        if r is not None:
                            r.incident_id = vlm_msg.incident_id
                            s.commit()
                self.broker.publish("incidents", {**inc.model_dump(),
                                                  "message_id": new_id("msg"),
                                                  "correlation_id": cand.correlation_id})
                if (is_new and inc.status == "confirmed") or escalated:
                    self.broker.publish("alerts", {"incident_id": inc.incident_id,
                                                   "camera_id": camera_id,
                                                   "event_type": inc.event_type,
                                                   "clip_id": clip[0] if clip else None,
                                                   "correlation_id": cand.correlation_id,
                                                   "message_id": new_id("msg")})
                    self.metrics.inc("alerts_confirmed")
                incidents.append(inc)
        # Unique incidents (dedup may merge several candidates into one).
        uniq = {i.incident_id: i for i in incidents}.values()
        self.metrics.observe("pipeline.video_total",
                             (time.perf_counter() - t_start) * 1000)
        self.metrics.inc("videos_processed")
        self.metrics.inc("frames_decoded_ingest", ing.decoded)
        return sorted(uniq, key=lambda i: i.window_start)

    # ---------------------------------------------------------------- verify
    def _verify_job(self, camera_id: str, job: dict):
        cand = CandidateEvent(**{k: v for k, v in job.items()
                                       if k in CandidateEvent.model_fields})
        scene_hint = job.get("scene_hint", "")
        source = job.get("source", "")
        with self.metrics.timer("stage.clip"):
            try:
                clip = self.clips.record(source, cand.window_start, cand.window_end)
            except Exception:
                clip = None
                self.metrics.inc("clip_failures")
        clip_seconds = (clip[3] - clip[2]) if clip else 0.0
        with self.metrics.timer("stage.vlm"):
            vlm = self.vlm.verify(clip[1] if clip else source, cand,
                                  cand.weapon_seen, cand.person_count, scene_hint)
        vlm.clip_id = clip[0] if clip else ""
        with self.metrics.timer("stage.decision"):
            vlm = self.dec.decide(cand, vlm, clip_seconds)
        self.metrics.inc(f"verdict.{vlm.verdict}")
        # Event-dense embedding for confirmed/uncertain clips (PRD §14).
        if clip is not None and vlm.verdict in ("confirmed", "uncertain"):
            try:
                import cv2 as _cv2
                cap = _cv2.VideoCapture(clip[1])
                frs = []
                while len(frs) < 12:
                    ok, f = cap.read()
                    if not ok:
                        break
                    frs.append(f)
                cap.release()
                vec = self.emb.embed_clip(frs, f"{scene_hint} {vlm.event_type}")
                self.vectors.add(clip[0], vec, {"camera_id": camera_id,
                                                "event_type": vlm.event_type or "",
                                                "description": vlm.description,
                                                "window_start": cand.window_start,
                                                "window_end": cand.window_end})
            except Exception:
                self.metrics.inc("embed_failures")
        return vlm, cand, clip
