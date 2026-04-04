"""Synthetic timing loop for CI (no GPU)."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class InferenceEvent:
    event: str
    tensor_parallel: int
    batch_size: int
    precision: str
    tokens_per_s: float
    wall_ms: float
    meta: dict[str, Any]


def run_synthetic_sweep(cfg: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tps = cfg.get("tensor_parallel", [1])
    bs = cfg.get("batch_size", [1])
    prec = cfg.get("precision", ["bf16"])
    warmup = int(cfg.get("warmup_iterations", 1))
    timed = int(cfg.get("timed_iterations", 5))

    lines = []
    for tp in tps:
        for b in bs:
            for p in prec:
                for _ in range(warmup):
                    time.sleep(0.001)
                start = time.perf_counter()
                for _ in range(timed):
                    time.sleep(0.0005 * random.random())
                wall = (time.perf_counter() - start) * 1000
                tok = 64.0 * b * timed / (wall / 1000.0)
                ev = InferenceEvent(
                    event="synthetic",
                    tensor_parallel=tp,
                    batch_size=b,
                    precision=p,
                    tokens_per_s=round(tok, 2),
                    wall_ms=round(wall, 3),
                    meta={"timed_iterations": timed},
                )
                lines.append(json.dumps(asdict(ev)))

    out.write_text("\n".join(lines) + "\n")
