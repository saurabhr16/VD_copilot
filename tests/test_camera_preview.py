import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app, db
from src.store import CameraRow


def test_camera_preview_route_serves_source_video(tmp_path):
    source = tmp_path / "cam_preview.mp4"
    source.write_bytes(b"fake-video-bytes")

    with db.session() as s:
        row = s.get(CameraRow, "cam_preview_test")
        if row is None:
            s.add(
                CameraRow(
                    camera_id="cam_preview_test",
                    source_path=str(source),
                    fps=30,
                    frames=1,
                    duration_s=1.0,
                    width=640,
                    height=360,
                    status="uploaded",
                    extra={"scene_hint": "preview"},
                )
            )
        else:
            row.source_path = str(source)
        s.commit()

    client = TestClient(app)
    resp = client.get("/api/cameras/cam_preview_test/video")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("video/")


def test_camera_analysis_includes_boxes_and_story(tmp_path):
    source = tmp_path / "cam_story.mp4"
    source.write_bytes(b"fake-video-bytes")

    with db.session() as s:
        row = s.get(CameraRow, "cam_story_test")
        if row is None:
            s.add(
                CameraRow(
                    camera_id="cam_story_test",
                    source_path=str(source),
                    fps=10,
                    frames=10,
                    duration_s=1.0,
                    width=640,
                    height=360,
                    status="uploaded",
                    extra={"scene_hint": "fight assault"},
                )
            )
        else:
            row.source_path = str(source)
            row.extra = {"scene_hint": "fight assault"}
        s.commit()

    client = TestClient(app)
    resp = client.get("/api/cameras/cam_story_test/analysis")

    assert resp.status_code == 200
    body = resp.json()
    assert body["camera_id"] == "cam_story_test"
    assert body["story"]
    assert body["detections"]
    assert any(obj["label"] in {"person", "knife", "gun"} for obj in body["detections"])
