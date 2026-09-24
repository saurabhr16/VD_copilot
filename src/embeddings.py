"""Embeddings + semantic search (PRD §13/§14) — mock adapter.

embed_clip(): 128-d vector from color histogram + motion + edges.
embed_text(): 128-d vector from hashed tokens + keyword anchors, so demo
queries like "person holding a knife" rank the weapon clip first. The real
CLIP/X-CLIP adapter implements the same two methods (PRD §35).
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

DIM = 128

ANCHORS = {
    "fight": ["fight", "fighting", "assault", "attack", "hit", "punch", "altercation", "violence", "violent"],
    "weapon": ["knife", "gun", "weapon", "stab", "shoot", "holding", "armed"],
    "run": ["run", "running", "chase", "chasing"],
    "fall": ["fall", "falling", "fell", "floor", "collapsed"],
    "person": ["person", "people", "man", "woman", "individual"],
    "calm": ["calm", "normal", "walk", "walking", "standing", "empty"],
}


def _anchor_vec(seed: str) -> np.ndarray:
    h = hashlib.md5(seed.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
    v = rng.randn(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) or 1.0)


_ANCHOR_VECS = {k: _anchor_vec(f"anchor-{k}") for k in ANCHORS}
_ANCHOR_VECS["__base__"] = _anchor_vec("anchor-base")


class BaseEmbeddingModel:
    DIM = DIM
    name = "base"

    def embed_clip(self, frames: list[np.ndarray], scene_hint: str = "") -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError


class MockEmbeddingModel(BaseEmbeddingModel):
    name = "mock-emb-v1"

    def embed_clip(self, frames: list[np.ndarray], scene_hint: str = "") -> np.ndarray:
        feats = np.zeros(DIM, dtype=np.float32)
        if frames:
            sample = frames[:: max(1, len(frames) // 8)][:8]
            hists, edges, motion = [], [], []
            prev = None
            for f in sample:
                small = cv2.resize(f, (64, 64))
                hist, _ = np.histogram(small, bins=16, range=(0, 256))
                hists.append(hist / (hist.sum() or 1))
                g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                edges.append(float(cv2.Canny(g, 80, 160).mean() / 255.0))
                if prev is not None:
                    motion.append(float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean() / 255.0))
                prev = g
            m = np.concatenate([np.mean(hists, axis=0) if hists else np.zeros(16),
                                [np.mean(edges) if edges else 0, np.mean(motion) if motion else 0,
                                 len(frames) / 100.0]]).astype(np.float32)
            rng = np.random.RandomState(7)
            proj = rng.randn(m.shape[0], DIM).astype(np.float32)
            feats = (m @ proj)
            feats = feats / (np.linalg.norm(feats) or 1.0)

        # Blend toward the scene anchor so retrieval is semantically ordered.
        h = (scene_hint or "").lower()
        key = ("fight" if any(k in h for k in ("fight", "assault", "attack")) else
               "weapon" if any(k in h for k in ("weapon", "knife", "gun", "stab", "shoot")) else
               "run" if any(k in h for k in ("run", "chase", "snatch")) else "calm")
        v = 0.45 * feats + 0.55 * _ANCHOR_VECS[key]
        return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        toks = [t.strip(".,!?;:()").lower() for t in text.lower().split()]
        v = 0.35 * _ANCHOR_VECS["__base__"]
        for key, words in ANCHORS.items():
            if any(t in words for t in toks):
                v = v + 0.9 * _ANCHOR_VECS[key]
        # Unknown words contribute a stable hash direction (never zero-vector).
        for t in toks:
            if len(t) > 2 and not any(t in w for w in ANCHORS.values()):
                v = v + 0.15 * _anchor_vec(f"tok-{t}")
        return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)


class CLIPAdapter(BaseEmbeddingModel):
    """Placeholder for X-CLIP/CLIP on the GPU host."""

    name = "clip-v1"

    def __init__(self, *a, **k):
        raise RuntimeError("CLIPAdapter needs a GPU host. See docs/SWAP_TO_REAL_MODELS.md.")
