"""End-to-end: short synthetic videos -> incidents + clips + search hits."""
import os
import sys
from pathlib import Path

os.environ["VSD_DEMO_SECONDS"] = "10"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_demo_videos  # noqa: E402

make_demo_videos.SECONDS = 10.0

from src.broker import Broker  # noqa: E402
from src.config import load_config  # noqa: E402
from src.embeddings import MockEmbeddingModel  # noqa: E402
from src.metrics import Metrics  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.store import MetadataDB, VectorStore  # noqa: E402


def _pipe(tmp_path):
    cfg = load_config()
    db = MetadataDB("sqlite:///:memory:")
    br = Broker(None)
    vs = VectorStore(tmp_path / "vec")
    return Pipeline(cfg, db, br, vs, Metrics(), tmp_path / "clips"), vs


def test_fight_video_produces_confirmed_incident(tmp_path):
    pipe, _ = _pipe(tmp_path)
    vids = make_demo_videos.ensure_demo_videos(tmp_path / "vids")
    fight = next(p for c, p, h in vids if c == "cam02")
    incs = pipe.process_video("cam02", str(fight), "fight assault")
    assert len(incs) >= 1, "fight video should yield >=1 incident"
    assert any(i.status == "confirmed" for i in incs), [i.status for i in incs]


def test_normal_video_quiet(tmp_path):
    pipe, _ = _pipe(tmp_path)
    vids = make_demo_videos.ensure_demo_videos(tmp_path / "vids")
    normal = next(p for c, p, h in vids if c == "cam01")
    incs = pipe.process_video("cam01", str(normal), "normal walk calm")
    confirmed = [i for i in incs if i.status == "confirmed"]
    assert not confirmed, "normal video should not confirm incidents"


def test_search_ranks_weapon_clip(tmp_path):
    pipe, vs = _pipe(tmp_path)
    vids = make_demo_videos.ensure_demo_videos(tmp_path / "vids")
    for cid, p, h in vids:
        pipe.process_video(cid, str(p), h)
    assert len(vs) >= 1
    qv = MockEmbeddingModel().embed_text("person holding a knife")
    hits = vs.search(qv, top_k=3)
    assert hits, "expected indexed clips"
    assert hits[0]["camera_id"] == "cam03", [h["camera_id"] for h in hits]
