"""Metadata store + vector store (PRD §15).

Metadata: SQLAlchemy models. SQLite by default; PostgreSQL when DATABASE_URL
is set (same code — this is the plug-and-play contract, PRD §35).
Vector:   file-backed numpy store with the same add/search interface the
Qdrant adapter implements (see docs/SWAP_TO_REAL_MODELS.md).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sqlalchemy import JSON, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parent.parent


class Base(DeclarativeBase):
    pass


class CameraRow(Base):
    __tablename__ = "cameras"
    camera_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, default="")
    fps: Mapped[float] = mapped_column(Float, default=10.0)
    frames: Mapped[int] = mapped_column(default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(default=0)
    height: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class IncidentRow(Base):
    __tablename__ = "incidents"
    incident_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    event_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    window_start: Mapped[float] = mapped_column(Float, default=0.0)
    window_end: Mapped[float] = mapped_column(Float, default=0.0)
    clip_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    candidate_ids: Mapped[list] = mapped_column(JSON, default=list)
    anomaly_scores: Mapped[list] = mapped_column(JSON, default=list)
    vlm_descriptions: Mapped[list] = mapped_column(JSON, default=list)
    operator_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), default="")
    threshold_version: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[str] = mapped_column(String(64), default="")
    updated_at: Mapped[str] = mapped_column(String(64), default="")


class FeedbackRow(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(String(64), index=True)
    operator_decision: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[str] = mapped_column(String(64))
    model_prediction: Mapped[str] = mapped_column(String(64), default="")
    model_version: Mapped[str] = mapped_column(String(64), default="")
    camera_id: Mapped[str] = mapped_column(String(64), default="")
    event_type: Mapped[str] = mapped_column(String(64), default="")


class ClipRow(Base):
    __tablename__ = "clips"
    clip_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    incident_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    path: Mapped[str] = mapped_column(Text, default="")
    window_start: Mapped[float] = mapped_column(Float, default=0.0)
    window_end: Mapped[float] = mapped_column(Float, default=0.0)
    description: Mapped[str] = mapped_column(Text, default="")
    event_type: Mapped[str] = mapped_column(String(64), default="")


def make_engine(db_url: str):
    if db_url.startswith("sqlite"):
        path = db_url.split("sqlite:///")[-1]
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        return create_engine(db_url, connect_args={"check_same_thread": False})
    return create_engine(db_url, pool_pre_ping=True)


class MetadataDB:
    def __init__(self, db_url: str):
        self.engine = make_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session: sessionmaker[Session] = sessionmaker(bind=self.engine)

    def session(self) -> Session:
        return self.Session()


# ---------------------------------------------------------------- vector store
class VectorStore:
    """Tiny persistent vector store: <data>/vectors.npz + vectors_meta.json.

    Interface mirrors the Qdrant adapter (add/search) so swapping backends
    only changes this class, never the callers (PRD §35).
    """

    DIM = 128

    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vec_path = self.dir / "vectors.npz"
        self._meta_path = self.dir / "vectors_meta.json"
        self.ids: list[str] = []
        self.vecs: np.ndarray = np.zeros((0, self.DIM), dtype=np.float32)
        self.meta: dict[str, dict] = {}
        self._load()

    def _load(self):
        try:
            if self._vec_path.exists():
                z = np.load(self._vec_path, allow_pickle=True)
                self.ids = list(z["ids"])
                self.vecs = z["vecs"].astype(np.float32)
            if self._meta_path.exists():
                self.meta = json.loads(self._meta_path.read_text())
        except Exception:
            self.ids, self.vecs, self.meta = [], np.zeros((0, self.DIM), dtype=np.float32), {}

    def _save(self):
        np.savez_compressed(self._vec_path, ids=np.array(self.ids), vecs=self.vecs)
        self._meta_path.write_text(json.dumps(self.meta))

    def add(self, clip_id: str, vector: np.ndarray, meta: dict):
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        assert v.shape[0] == self.DIM, f"expected dim {self.DIM}, got {v.shape}"
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        if clip_id in self.ids:
            self.vecs[self.ids.index(clip_id)] = v
        else:
            self.ids.append(clip_id)
            self.vecs = np.vstack([self.vecs, v]) if len(self.vecs) else v.reshape(1, -1)
        self.meta[clip_id] = meta
        self._save()

    def search(self, query_vec: np.ndarray, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        if len(self.ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        sims = self.vecs @ q
        order = np.argsort(-sims)
        hits = []
        for idx in order:
            cid = self.ids[int(idx)]
            m = self.meta.get(cid, {})
            if filters:
                if filters.get("camera_id") and m.get("camera_id") != filters["camera_id"]:
                    continue
                if filters.get("event_type") and m.get("event_type") != filters["event_type"]:
                    continue
            hits.append({"clip_id": cid, "score": float(sims[int(idx)]), **m})
            if len(hits) >= top_k:
                break
        return hits

    def __len__(self):
        return len(self.ids)
