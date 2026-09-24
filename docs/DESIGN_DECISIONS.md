# Design decisions — every PRD TBD resolved (prototype values)

All values live in `config.yaml` and are stamped on incidents via
`threshold_version`. Validate/change them with the Phase-0 benchmark (PRD §40).

## Temporal analysis (§8)

| Parameter | Value | Rationale |
|---|---|---|
| Window duration | 4 s | covers a punch/grab sequence; short enough for <5 s candidate alert |
| Sampling FPS | 5 | motion cues survive; ~1/5th the full-rate cost |
| Stride | 2 s (50% overlap) | overlapping windows → no boundary misses |
| Smoothing | EMA α=0.4 | kills single-window spikes, keeps <2-window reaction |
| Min event duration | 2 s | shorter events kept but flagged `short` |
| Persistence | 1 window (prototype) / 2 (prod) | responsive demo; prod needs 2 to cut flicker |
| Cooldown | 30 s per camera+type | dedups one incident into one alert (§11) |

## Candidate generation (§7)

Fire when ANY holds: `anomaly ≥ 0.55` · `weapon seen` · `≥2 persons + anomaly ≥ 0.45`.
The anomaly path never requires a detection (PRD key principle #1).

## Decision engine (§10)

| Signals | Verdict |
|---|---|
| weapon + anomaly ≥ 0.45 + temporal | confirmed |
| contact + anomaly ≥ 0.5 + multi-person | confirmed |
| anomaly ≥ 0.75 + (contact or weapon) | confirmed |
| borderline / detector↔VLM disagreement | **uncertain** → human |
| clip < 1 s | insufficient_evidence |
| else | rejected |

## VLM queue / backpressure (§26)

Bounded priority queue: depth 50, job TTL 120 s, 2 concurrent verifications,
expired jobs → dead-letter stream `vlm_dlq`, full queue drops lowest-priority
first and counts `vlm_dropped_full` (never silent).

## Recording (§12)

Pre 5 s + post 5 s, H264 + faststart MP4, 30-day retention default.

## Embeddings (§14)

Event-dense only: confirmed/uncertain clips embedded at 12 frames; normal
footage not embedded in the prototype (low-rate background indexing is a
Phase-3 production task).

## Latency budget (§23, prototype targets)

detect <80 ms/frame · anomaly <150 ms/window · VLM mock 250 ms (real Qwen2-VL
2–8 s on A6000) · candidate alert <5 s · verified alert <15 s.

## VRAM budget, 1× A6000 48 GB (§24, planning)

YOLO 3 GB · VadCLIP 5 GB · Qwen2-VL-7B-AWQ (vLLM) 18 GB · CLIP 2 GB · decode/
buffers 4 GB → ~32 GB used, ~16 GB headroom. Measure, don't trust (§25).

## Infrastructure picks (§41)

Redis Streams (broker) · PostgreSQL (metadata) · Qdrant (vectors) · FastAPI ·
Docker Compose on WSL2. Prototype runs the same code against fakeredis /
SQLite / numpy by default — set `REDIS_URL` / `DATABASE_URL` to switch.
