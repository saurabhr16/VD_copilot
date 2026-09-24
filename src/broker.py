"""Message broker abstraction (PRD §34).

Production: Redis Streams (REDIS_URL set, e.g. redis://redis:6379/0).
Prototype default: in-process fakeredis — identical API, zero infra.

Streams: detections | anomalies | candidates | vlm_jobs | incidents | alerts
Every message carries message_id / correlation_id for end-to-end tracing.
"""
from __future__ import annotations

import json
import time

STREAMS = ["detections", "anomalies", "candidates", "vlm_jobs", "incidents", "alerts"]


class Broker:
    def __init__(self, url: str | None = None):
        self.url = url
        if url:
            import redis

            self._r = redis.Redis.from_url(url, decode_responses=True)
        else:
            import fakeredis

            self._r = fakeredis.FakeStrictRedis(decode_responses=True)
        self.dropped: dict[str, int] = {s: 0 for s in STREAMS}

    @property
    def mode(self) -> str:
        return "redis" if self.url else "fakeredis(edge)"

    def publish(self, stream: str, message: dict, maxlen: int = 5000) -> str:
        body = dict(message)
        body.setdefault("broker_ts", time.time())
        return self._r.xadd(stream, {"json": json.dumps(body)}, maxlen=maxlen, approximate=True)

    def read(self, stream: str, count: int = 50, last_id: str = "0-0") -> list[dict]:
        entries = self._r.xrange(stream, min=last_id if last_id != "0-0" else "-", count=count)
        out = []
        for eid, fields in entries:
            try:
                payload = json.loads(fields.get("json", "{}"))
            except Exception:
                payload = {"raw": fields}
            payload["_entry_id"] = eid
            out.append(payload)
        return out

    def depth(self, stream: str) -> int:
        try:
            return int(self._r.xlen(stream))
        except Exception:
            return 0

    def depths(self) -> dict[str, int]:
        return {s: self.depth(s) for s in STREAMS}

    # --- priority VLM queue (sorted set; score = priority) -----------------
    def vlm_enqueue(self, job: dict, max_depth: int = 50) -> bool:
        """Returns False when the queue is full (caller applies drop policy)."""
        if self._r.zcard("vlm_queue") >= max_depth:
            self.dropped["vlm_jobs"] += 1
            return False
        job = dict(job)
        job.setdefault("enqueued_ts", time.time())
        self._r.zadd("vlm_queue", {json.dumps(job): float(job.get("priority", 0.0))})
        return True

    def vlm_dequeue(self, ttl_seconds: int = 120) -> dict | None:
        """Pop highest-priority non-expired job; expired jobs go to DLQ."""
        now = time.time()
        items = self._r.zrevrange("vlm_queue", 0, 0, withscores=False)
        while items:
            raw = items[0]
            self._r.zrem("vlm_queue", raw)
            try:
                job = json.loads(raw)
            except Exception:
                items = self._r.zrevrange("vlm_queue", 0, 0, withscores=False)
                continue
            if now - float(job.get("enqueued_ts", now)) > ttl_seconds:
                self._r.xadd("vlm_dlq", {"json": json.dumps(job)})
                items = self._r.zrevrange("vlm_queue", 0, 0, withscores=False)
                continue
            return job
        return None

    def vlm_depth(self) -> int:
        try:
            return int(self._r.zcard("vlm_queue"))
        except Exception:
            return 0

    def ping(self) -> bool:
        try:
            self._r.ping()
            return True
        except Exception:
            return False
