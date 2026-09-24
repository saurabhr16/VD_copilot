"""Telemetry collector (PRD §28): per-stage counters, latencies, resources."""
from __future__ import annotations

import shutil
import subprocess
import time
from collections import defaultdict

import psutil


class Metrics:
    def __init__(self):
        self.counters: dict[str, int] = defaultdict(int)
        self.lat_ms: dict[str, list[float]] = defaultdict(list)
        self.t0 = time.time()

    def inc(self, name: str, n: int = 1):
        self.counters[name] += n

    def observe(self, name: str, ms: float):
        buf = self.lat_ms[name]
        buf.append(ms)
        if len(buf) > 500:
            del buf[: len(buf) - 500]

    def timer(self, name: str):
        return _Timer(self, name)

    def snapshot(self) -> dict:
        lat = {}
        for k, v in self.lat_ms.items():
            if not v:
                continue
            s = sorted(v)
            lat[k] = {
                "count": len(s),
                "p50_ms": round(s[len(s) // 2], 1),
                "p95_ms": round(s[int(len(s) * 0.95)], 1),
                "max_ms": round(s[-1], 1),
            }
        disk = shutil.disk_usage("/")
        return {
            "uptime_s": round(time.time() - self.t0, 1),
            "counters": dict(self.counters),
            "latency": lat,
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_mb": round(psutil.virtual_memory().used / 1e6, 1),
            "ram_pct": psutil.virtual_memory().percent,
            "disk_free_gb": round(disk.free / 1e9, 2),
            "gpu": gpu_info(),
        }


class _Timer:
    def __init__(self, m: Metrics, name: str):
        self.m, self.name = m, name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.m.observe(self.name, (time.perf_counter() - self.t) * 1000)


def gpu_info() -> dict:
    """nvidia-smi when present (A6000 host), else a CPU-mode marker."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            u, mu, mt, temp = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
            return {"present": True, "util_pct": float(u), "mem_used_mb": float(mu),
                    "mem_total_mb": float(mt), "temp_c": float(temp)}
    except Exception:
        pass
    return {"present": False, "mode": "cpu-prototype"}
