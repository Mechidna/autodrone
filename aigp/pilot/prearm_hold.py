from __future__ import annotations

import math
import time
from typing import Callable


def hold_before_competition_arm(
    duration_s: float,
    *,
    recorder=None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[..., None] = print,
) -> float:
    """Keep the competition vehicle unarmed while sensor threads record."""
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("pre-arm sensor hold duration must be non-negative and finite")
    if duration_s == 0.0:
        return 0.0

    recorder_enabled = bool(getattr(recorder, "enabled", False))
    if recorder_enabled:
        recorder.record_event(
            "competition_prearm_sensor_hold_started",
            configured_duration_s=duration_s,
        )

    print_fn(
        "Competition pre-arm sensor hold: "
        f"recording without arming or control for {duration_s:.2f}s...",
        flush=True,
    )
    started = monotonic_fn()
    sleep_fn(duration_s)
    elapsed_s = max(0.0, monotonic_fn() - started)

    if recorder_enabled:
        recorder.record_event(
            "competition_prearm_sensor_hold_completed",
            configured_duration_s=duration_s,
            elapsed_s=elapsed_s,
        )
    print_fn(
        f"Competition pre-arm sensor hold complete after {elapsed_s:.2f}s.",
        flush=True,
    )
    return elapsed_s
