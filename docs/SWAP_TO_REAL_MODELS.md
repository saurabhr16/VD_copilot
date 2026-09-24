# Swapping mock → real models (PRD §35 plug-and-play contract)

A stage is swappable if it keeps: input schema, output schema, version string,
error semantics, and correlation IDs. Each mock module already contains the
real adapter stub.

## 1. Detector: MockDetector → UltralyticsYOLOAdapter

```python
# src/pipeline.py
from src.detection_mock import UltralyticsYOLOAdapter
self.det = UltralyticsYOLOAdapter(weights="yolov8m.pt", device="cuda:0")
```

Needs: `pip install ultralytics torch --index-url https://download.pytorch.org/whl/cu121`.
Fine-tune on your weapon classes; keep labels `person/knife/gun/...`.

## 2. Anomaly: MockAnomalyScorer → VadCLIPAdapter

Implement `score_window(frames, detections) -> float` in `src/anomaly_mock.py`
with VadCLIP/LAVAD/MGFN. Keep the EMA + window/stride in the pipeline so
model swaps don't change event semantics.

## 3. VLM: MockVLMVerifier → Qwen2VLAdapter

Serve Qwen2-VL-7B-AWQ with vLLM (`--max-model-len 4096 --gpu-memory-utilization 0.5`),
implement `verify()` to send clip frames + candidate context and parse the
structured JSON (see PRD §10). Keep the DecisionEngine — the VLM returns
*evidence*, the engine returns the *verdict*.

## 4. Embeddings: MockEmbeddingModel → CLIPAdapter

Implement `embed_clip` / `embed_text` with X-CLIP/CLIP (512-d → project to
128-d or change `VectorStore.DIM`). Re-index existing clips after swap.

## 5. Infra: fakeredis/SQLite/numpy → Redis/Postgres/Qdrant

```bash
export REDIS_URL=redis://localhost:6379/0
export DATABASE_URL=postgresql+psycopg://vsd:vsd@localhost:5432/vsd
```

`Broker`, `MetadataDB`, and `VectorStore` read these env vars; no code change.
For Qdrant, replace `VectorStore` internals with `qdrant-client` calls keeping
`add()` / `search()` signatures.

## Checklist before going live

- [ ] Re-tune thresholds on the Phase-0 benchmark (§40), bump `threshold_version`
- [ ] Re-run `pytest` + the §25 throughput ladder (1→2→4→8→N cameras)
- [ ] Confirm VLM concurrency ≤ 2 and queue behavior under burst (§26)
- [ ] Wire RTSP sources into `VideoIngestion` (source URL instead of file path)
