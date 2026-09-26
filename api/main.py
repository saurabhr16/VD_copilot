"""Operator API + dashboard (PRD §33 services 13/14/16, file-input variant).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Then open the live preview URL.
"""
from __future__ import annotations

import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.broker import Broker  # noqa: E402
from src.config import database_url, load_config, redis_url  # noqa: E402
from src.detector import create_detector  # noqa: E402
from src.embeddings import MockEmbeddingModel  # noqa: E402
from src.ingestion import VideoIngestion, probe  # noqa: E402
from src.metrics import Metrics  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.schemas import SearchHit  # noqa: E402
from src.store import (CameraRow, ClipRow, FeedbackRow, IncidentRow,  # noqa: E402
                       MetadataDB, VectorStore)
from src.vlm_mock import NarrativeGenerator  # noqa: E402

cfg = load_config()
db = MetadataDB(database_url(cfg))
broker = Broker(redis_url(cfg))
vectors = VectorStore(ROOT / "data")
metrics = Metrics()
pipe = Pipeline(cfg, db, broker, vectors, metrics, ROOT / "data" / "clips")
embedder = MockEmbeddingModel()

UPLOADS = ROOT / "data" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Violence-Detection Surveillance Prototype", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

jobs: dict[str, dict] = {}
analysis_cache: dict[str, tuple[tuple, dict]] = {}
detector_cache: dict[tuple, object] = {}
pool = ThreadPoolExecutor(max_workers=1)  # sequential pipeline runs (SQLite-safe)


# ------------------------------------------------------------------ helpers
def row_to_incident(r: IncidentRow) -> dict:
    return {c.name: getattr(r, c.name) for c in r.__table__.columns}


def _scene_hint_from_name(name: str) -> str:
    n = name.lower()
    hints = []
    for k in ("fight", "assault", "attack", "weapon", "knife", "gun", "stab",
              "shoot", "run", "chase", "snatch", "normal", "calm", "walk"):
        if k in n:
            hints.append(k)
    return " ".join(hints)


# -------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (ROOT / "api" / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "broker": broker.mode, "broker_ok": broker.ping(),
            "db": database_url(cfg).split("://")[0], "vectors": len(vectors),
            "threshold_version": cfg.threshold_version,
            "models": cfg.model_versions,
            "detector_runtime": pipe.det.name,
            "time": time.time()}


@app.get("/api/metrics")
def get_metrics():
    snap = metrics.snapshot()
    snap["broker_depths"] = broker.depths()
    snap["vlm_queue_depth"] = broker.vlm_depth()
    snap["vlm_dropped_full"] = broker.dropped.get("vlm_jobs", 0)
    return snap


@app.get("/api/broker")
def broker_inspect(stream: str = "incidents", count: int = 20):
    return {"stream": stream, "depths": broker.depths(),
            "messages": broker.read(stream, count=count)}


@app.get("/api/config")
def get_config():
    return {"threshold_version": cfg.threshold_version, "pipeline": cfg.pipeline,
            "thresholds": cfg.thresholds, "recording": cfg.recording,
            "vlm_queue": cfg.vlm_queue, "models": cfg.model_versions}


# ------------------------------------------------------------------- cameras
@app.get("/api/cameras")
def list_cameras():
    with db.session() as s:
        rows = s.query(CameraRow).order_by(CameraRow.camera_id).all()
        return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]


@app.get("/api/cameras/{camera_id}/video")
def serve_camera_video(camera_id: str):
    with db.session() as s:
        r = s.get(CameraRow, camera_id)
        if r is None or not r.source_path or not Path(r.source_path).exists():
            raise HTTPException(404, "camera video not found")
        mime = "video/mp4"
        if r.source_path.lower().endswith(".webm"):
            mime = "video/webm"
        elif r.source_path.lower().endswith(".avi"):
            mime = "video/x-msvideo"
        return FileResponse(r.source_path, media_type=mime, filename=f"{camera_id}{Path(r.source_path).suffix}")


@app.get("/api/cameras/{camera_id}/analysis")
def camera_analysis(camera_id: str):
    with db.session() as s:
        row = s.get(CameraRow, camera_id)
        if row is None:
            raise HTTPException(404, "camera not found")
        source = row.source_path or ""
        if not source or not Path(source).exists():
            raise HTTPException(404, "camera video not found")
        scene_hint = (row.extra or {}).get("scene_hint", "")
        source_stat = Path(source).stat()
        cache_key = (source, source_stat.st_mtime_ns, source_stat.st_size, scene_hint,
                     cfg.system.get("yolo_imgsz", 320), id(VideoIngestion), id(create_detector))
    cached = analysis_cache.get(camera_id)
    if cached and cached[0] == cache_key:
        return cached[1]

    detector_key = (cfg.system.get("yolo_weights", "yolov8n.pt"),
                    cfg.system.get("yolo_device", "auto"),
                    int(cfg.system.get("yolo_imgsz", 320)),
                    float(cfg.thresholds.get("person_conf", 0.5)),
                    float(cfg.thresholds.get("weapon_conf", 0.4)), id(create_detector))
    detector = detector_cache.get(detector_key)
    if detector is None:
        detector = create_detector(weights=detector_key[0], device=detector_key[1],
                                   imgsz=detector_key[2], person_conf=detector_key[3],
                                   weapon_conf=detector_key[4])
        detector_cache[detector_key] = detector
    detector.reset(scene_hint)
    detections = []
    frame_results = []
    try:
        preview_fps = float(cfg.system.get("preview_fps", 5))
        preview_seconds = float(cfg.system.get("preview_seconds", 12))
        ing = VideoIngestion(source, target_fps=preview_fps)
        frame_detections_by_label: dict[str, dict] = {}
        for i, frame in enumerate(ing.iter_frames()):
            frame_timestamp = float(getattr(frame, "t", i / preview_fps if preview_fps else i))
            if frame_timestamp > preview_seconds:
                break
            frame_detections = []
            for d in detector.detect(frame.image):
                payload = {
                    "track_id": d.track_id,
                    "label": d.label,
                    "bbox": [round(float(v), 2) for v in d.bbox],
                    "confidence": round(float(d.confidence), 3),
                    "occluded": bool(d.occluded),
                    "suspect": bool(d.suspect),
                }
                frame_detections.append(payload)
                key = (d.track_id or d.label, d.label)
                if key not in frame_detections_by_label:
                    frame_detections_by_label[key] = payload
                else:
                    prev = frame_detections_by_label[key]
                    if payload["confidence"] > prev["confidence"]:
                        frame_detections_by_label[key] = payload
            frame_results.append({"frame_index": i, "timestamp": round(frame_timestamp, 3),
                                  "detections": frame_detections})
        detections = list(frame_detections_by_label.values())
    except Exception:
        detections = []

    story = NarrativeGenerator().narrative_for_camera(camera_id, scene_hint, detections)
    response = {
        "camera_id": camera_id,
        "scene_hint": scene_hint,
        "story": story,
        "vlm_summary": story,
        "detections": detections,
        "frames": frame_results,
        "width": row.width,
        "height": row.height,
        "source": source,
    }
    analysis_cache[camera_id] = (cache_key, response)
    return response


@app.post("/api/cameras/upload")
async def upload_camera(camera_id: str = Form(...),
                        file: UploadFile = File(...),
                        scene_hint: str = Form("")):
    camera_id = "".join(c for c in camera_id.strip() if c.isalnum() or c in "-_") or "cam01"
    dest = UPLOADS / f"{camera_id}_{Path(file.filename or 'video.mp4').name}"
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    try:
        info = probe(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"unreadable video file: {e}")
    hint = scene_hint.strip() or _scene_hint_from_name(file.filename or "")
    with db.session() as s:
        row = s.get(CameraRow, camera_id)
        data = dict(source_path=str(dest), fps=info.fps, frames=info.frames,
                    duration_s=round(info.duration_s, 2), width=info.width,
                    height=info.height, status="uploaded",
                    extra={"scene_hint": hint, "filename": file.filename})
        if row is None:
            s.add(CameraRow(camera_id=camera_id, **data))
        else:
            for k, v in data.items():
                setattr(row, k, v)
        s.commit()
    analysis_cache.pop(camera_id, None)
    return {"camera_id": camera_id, "scene_hint": hint, "duration_s": round(info.duration_s, 2),
            "fps": info.fps, "frames": info.frames}


@app.post("/api/cameras/demo")
def load_demo_videos():
    """Generate (if needed) + register the 4 synthetic demo cameras."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_demo_videos import ensure_demo_videos
    videos = ensure_demo_videos(ROOT / "demo_videos")
    out = []
    for camera_id, path, hint in videos:
        info = probe(str(path))
        with db.session() as s:
            row = s.get(CameraRow, camera_id)
            data = dict(source_path=str(path), fps=info.fps, frames=info.frames,
                        duration_s=round(info.duration_s, 2), width=info.width,
                        height=info.height, status="uploaded",
                        extra={"scene_hint": hint, "demo": True})
            if row is None:
                s.add(CameraRow(camera_id=camera_id, **data))
            else:
                for k, v in data.items():
                    setattr(row, k, v)
            s.commit()
        analysis_cache.pop(camera_id, None)
        out.append({"camera_id": camera_id, "scene_hint": hint,
                    "duration_s": round(info.duration_s, 2)})
    return {"cameras": out}


# ------------------------------------------------------------------ pipeline
class RunRequest(BaseModel):
    camera_ids: list[str] | None = None


def _run_job(job_id: str, camera_ids: list[str]):
    jobs[job_id].update(status="running", started=time.time())
    results = []
    try:
        with db.session() as s:
            rows = s.query(CameraRow).filter(CameraRow.camera_id.in_(camera_ids)).all()
            specs = [(r.camera_id, r.source_path, (r.extra or {}).get("scene_hint", "")) for r in rows]
        for i, (cid, src, hint) in enumerate(specs):
            jobs[job_id].update(current=cid, progress=f"{i}/{len(specs)}")
            with db.session() as s:
                r = s.get(CameraRow, cid)
                if r is not None:
                    r.status = "processing"
                    s.commit()
            try:
                incs = pipe.process_video(cid, src, hint)
                status = "done"
            except Exception as e:  # per-camera isolation (PRD §27)
                incs, status = [], f"error: {e}"
                metrics.inc("camera_errors")
            with db.session() as s:
                r = s.get(CameraRow, cid)
                if r is not None:
                    r.status = status
                    s.commit()
            results.append({"camera_id": cid, "status": status,
                            "incidents": [r.model_dump() for r in incs]})
        jobs[job_id].update(status="done", progress=f"{len(specs)}/{len(specs)}",
                            results=results, ended=time.time())
    except Exception as e:
        jobs[job_id].update(status="error", error=str(e), ended=time.time())


@app.post("/api/pipeline/run")
def run_pipeline(req: RunRequest):
    with db.session() as s:
        if req.camera_ids:
            cams = [r.camera_id for r in s.query(CameraRow)
                    .filter(CameraRow.camera_id.in_(req.camera_ids)).all()]
        else:
            cams = [r.camera_id for r in s.query(CameraRow).all()]
    if not cams:
        raise HTTPException(400, "no cameras registered — upload videos or load the demo set first")
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"job_id": job_id, "status": "queued", "cameras": cams,
                    "created": time.time()}
    pool.submit(_run_job, job_id, cams)
    return {"job_id": job_id, "cameras": cams}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[job_id]


# ----------------------------------------------------------------- incidents
@app.get("/api/incidents")
def list_incidents(status: str | None = None, camera_id: str | None = None):
    with db.session() as s:
        q = s.query(IncidentRow).order_by(IncidentRow.created_at.desc())
        if status:
            q = q.filter(IncidentRow.status == status)
        if camera_id:
            q = q.filter(IncidentRow.camera_id == camera_id)
        return [row_to_incident(r) for r in q.limit(200).all()]


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: str):
    with db.session() as s:
        r = s.get(IncidentRow, incident_id)
        if r is None:
            raise HTTPException(404, "unknown incident")
        d = row_to_incident(r)
        if d.get("clip_id"):
            clip = s.get(ClipRow, d["clip_id"])
            d["clip_path"] = clip.path if clip else None
        return d


class FeedbackRequest(BaseModel):
    decision: str  # confirmed | false_positive | uncertain | escalated


@app.post("/api/incidents/{incident_id}/feedback")
def submit_feedback(incident_id: str, req: FeedbackRequest):
    if req.decision not in ("confirmed", "false_positive", "uncertain", "escalated"):
        raise HTTPException(400, "bad decision value")
    from src.schemas import utcnow as _utcnow
    with db.session() as s:
        r = s.get(IncidentRow, incident_id)
        if r is None:
            raise HTTPException(404, "unknown incident")
        r.operator_decision = req.decision
        s.add(FeedbackRow(incident_id=incident_id, operator_decision=req.decision,
                           timestamp=_utcnow(), model_prediction=r.status,
                           model_version=r.model_version, camera_id=r.camera_id,
                           event_type=r.event_type or ""))
        s.commit()
    metrics.inc("feedback")
    return {"incident_id": incident_id, "operator_decision": req.decision}


# --------------------------------------------------------------------- clips
@app.get("/api/clips")
def list_clips():
    with db.session() as s:
        rows = s.query(ClipRow).order_by(ClipRow.clip_id).all()
        return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]


@app.get("/api/clips/{clip_id}/video")
def serve_clip(clip_id: str):
    with db.session() as s:
        r = s.get(ClipRow, clip_id)
        if r is None or not Path(r.path).exists():
            raise HTTPException(404, "clip not found")
        return FileResponse(r.path, media_type="video/mp4", filename=f"{clip_id}.mp4")


# -------------------------------------------------------------------- search
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    camera_id: str | None = None
    event_type: str | None = None


@app.post("/api/search")
def semantic_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    t = time.perf_counter()
    qv = embedder.embed_text(req.query)
    filters = {k: v for k, v in
               (("camera_id", req.camera_id), ("event_type", req.event_type)) if v}
    raw = vectors.search(qv, top_k=req.top_k, filters=filters or None)
    metrics.observe("search.query", (time.perf_counter() - t) * 1000)
    with db.session() as s:
        out = []
        for h in raw:
            clip = s.get(ClipRow, h["clip_id"])
            out.append(SearchHit(clip_id=h["clip_id"],
                                 incident_id=clip.incident_id if clip else None,
                                 camera_id=h.get("camera_id", ""),
                                 score=round(h["score"], 3),
                                 event_type=h.get("event_type") or None,
                                 description=h.get("description", ""),
                                 window_start=float(h.get("window_start", 0)),
                                 window_end=float(h.get("window_end", 0))).model_dump())
    return {"query": req.query, "hits": out}
