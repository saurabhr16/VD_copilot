# Violence & Anomaly Detection — Surveillance Prototype

Multi-camera violence/anomaly detection pipeline from the PRD, runnable **today on CPU**
with mock/heuristic models, and swappable to real models (YOLO + VadCLIP + Qwen2-VL)
on the 1× RTX A6000 host. File-upload input in this prototype (PRD §32 "Video-4 /
Web UI" variant); RTSP plugs into the same `VideoIngestion` class.

```
upload 4 videos → detect + anomaly → candidates → VLM queue → decision
      → incident dedup → clips + metadata + embeddings → alerts + search
```

## Quickstart

```bash
cd violence-detection
pip install -r requirements.txt
python scripts/make_demo_videos.py          # 4 synthetic cameras (~20 s each)
uvicorn api.main:app --host 0.0.0.0 --port 8000
# open the live preview URL → Cameras → "Load 4 demo videos" → "Run detection"
```

Or one shot: `bash run.sh` (installs deps, builds videos, runs tests, starts server).

## What you get

- **Operator dashboard** (`/`): upload videos, run pipeline, review incidents with
  playable clips, confirm/reject/escalate (human-in-the-loop, PRD §29), semantic
  search, telemetry + broker inspector.
- **Pipeline** (`src/pipeline.py`): sliding 4 s/2 s windows, multi-signal candidates
  (anomaly alone can fire — detection is never a hard gate, PRD §7), bounded
  priority VLM queue (PRD §26), rule-based decision engine with `uncertain` /
  `insufficient_evidence` states (PRD §9–10), cooldown dedup (PRD §11).
- **Contracts**: all messages validate against PRD §16 pydantic schemas
  (`src/schemas.py`); every broker message carries `message_id/correlation_id` (§34).
- **Storage**: SQLite (default) or PostgreSQL via `DATABASE_URL`; numpy vector store
  (default) with the Qdrant-compatible `add/search` interface; H264 pre/post clips.
- **Tests**: `pytest` — contracts, decision rules, dedup, and end-to-end (fight →
  confirmed, normal → quiet, "knife" query → weapon clip first).

## Repo map

| Path | PRD section |
|---|---|
| `src/schemas.py`, `src/config.py` | §16 contracts, §31 config |
| `src/ingestion.py`, `src/clips.py` | §6.1 ingest, §12 recording |
| `src/detection_mock.py`, `src/anomaly_mock.py` | §6.2, §8 |
| `src/candidate.py`, `src/decision.py`, `src/incidents.py` | §7, §10, §11 |
| `src/vlm_mock.py`, `src/embeddings.py` | §10, §13–14 |
| `src/broker.py`, `src/store.py`, `src/metrics.py` | §34, §15, §28 |
| `src/pipeline.py`, `api/` | §32–33 |
| `docs/DESIGN_DECISIONS.md` | every TBD resolved |
| `docs/SWAP_TO_REAL_MODELS.md` | mock → A6000 adapters |

## Expected demo outcome

| Camera | Content | Expected |
|---|---|---|
| cam01 | person strolling | no confirmed incident |
| cam02 | two people fighting | **confirmed** `fighting` |
| cam03 | person + knife | **confirmed** `stabbing` |
| cam04 | person sprinting | uncertain or quiet (tune `anomaly_candidate`) |

Search checks: "person holding a knife" → cam03 first · "two people fighting" →
cam02 first · "calm hallway" → low scores everywhere.

## Production path (A6000)

1. `docs/SWAP_TO_REAL_MODELS.md` — drop in YOLO / VadCLIP / Qwen2-VL.
2. `docker-compose.yml` — Redis, PostgreSQL, Qdrant.
3. Phase-0 benchmark first (PRD §40): re-tune thresholds, bump `threshold_version`,
   run the 1→2→4→8→N camera ladder (§25) before claiming capacity.
