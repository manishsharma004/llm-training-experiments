"""Linear warmup, then cosine decay to a floor."""

from __future__ import annotations

import math


def learning_rate(step: int, total_steps: int, peak: float, floor: float, warmup_steps: int) -> float:
    """`step` is 1-indexed, matching the training loop."""

    if total_steps < 1:
        return peak
    step = max(1, min(step, total_steps))
    if warmup_steps > 0 and step <= warmup_steps:
        return peak * step / warmup_steps
    if total_steps == warmup_steps:
        return peak
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * cosine
