"""Label-blind, deterministic hysteresis routing of physical PPG channels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ROUTING_SPEC = {"version": "v5-hysteresis-1", "min_sqi": .30, "advantage": .05,
                "consecutive_frames": 2, "top_k": 3, "dwell_normalization_frames": 10}

@dataclass
class HysteresisRouter:
    channel_ids: tuple[int, ...]
    min_sqi: float = .30
    advantage: float = .05
    consecutive_frames: int = 2
    top_k: int = 3

    def __post_init__(self):
        self.channel_ids = tuple(int(c) for c in self.channel_ids)
        if len(set(self.channel_ids)) != len(self.channel_ids) or any(c <= 0 or c == 2 for c in self.channel_ids):
            raise ValueError("Physical candidate IDs must be unique and exclude ambient PPG2")
        if not 1 <= self.top_k <= len(self.channel_ids) or self.consecutive_frames < 1:
            raise ValueError("Invalid routing configuration")
        self.reset()

    def reset(self):
        self.selected = np.zeros(self.top_k, np.int16)
        self.pending = {}
        self.dwell = {}
        self.previous = set()

    def update(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(self.channel_ids),):
            raise ValueError("One SQI per physical candidate is required")
        q = {c: float(s) if np.isfinite(s) else -np.inf for c, s in zip(self.channel_ids, scores)}
        eligible = sorted((c for c in self.channel_ids if q[c] >= self.min_sqi), key=lambda c: (-q[c], c))
        # Failures are not delayed by the two-frame challenger confirmation.
        for slot, c in enumerate(self.selected):
            if not c or q[int(c)] < self.min_sqi:
                self.selected[slot] = 0
                self.pending.pop(slot, None)
        in_use = set(int(c) for c in self.selected if c)
        for slot in np.flatnonzero(self.selected == 0):
            candidate = next((c for c in eligible if c not in in_use), None)
            if candidate is not None:
                self.selected[slot] = candidate
                in_use.add(candidate)
        # Pair best challengers with weakest occupied slots, with deterministic ties.
        challengers = [c for c in eligible if c not in in_use]
        slots = sorted((i for i, c in enumerate(self.selected) if c),
                       key=lambda i: (q[int(self.selected[i])], -int(self.selected[i])))
        next_pending = {}
        for slot, candidate in zip(slots, challengers):
            incumbent = int(self.selected[slot])
            if q[candidate] - q[incumbent] + 1e-12 < self.advantage:
                continue
            prior = self.pending.get(slot)
            count = prior[2] + 1 if prior and prior[:2] == (incumbent, candidate) else 1
            if count >= self.consecutive_frames:
                self.selected[slot] = candidate
            else:
                next_pending[slot] = (incumbent, candidate, count)
        self.pending = next_pending
        current = set(int(c) for c in self.selected if c)
        changed = (1. - len(self.previous & current) / max(len(self.previous), len(current), 1)) if self.previous else float(bool(current) and bool(self.dwell))
        self.dwell = {c: self.dwell.get(c, 0) + 1 for c in current}
        dwell = np.mean([min(self.dwell[c] / 10., 1.) for c in current]) if current else 0.
        self.previous = current
        quality = np.array([q[int(c)] if c else 0. for c in self.selected], np.float32)
        return self.selected.copy(), quality, np.array([changed, dwell], np.float32)


def route_sequence(scores, channel_ids, frame_start_ms=None, **kwargs):
    """Reset only at an actual time discontinuity, never at model crop boundaries."""
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError("scores must be [frame, physical channel]")
    router = HysteresisRouter(tuple(channel_ids), **kwargs)
    ids = np.zeros((len(scores), router.top_k), np.int16)
    quality = np.zeros_like(ids, np.float32)
    context = np.zeros((len(scores), 2), np.float32)
    starts = None if frame_start_ms is None else np.asarray(frame_start_ms)
    if starts is not None and starts.shape != (len(scores),):
        raise ValueError("Frame time length mismatch")
    for i, row in enumerate(scores):
        if starts is not None and i and starts[i] - starts[i - 1] != 30000:
            router.reset()
        ids[i], quality[i], context[i] = router.update(row)
    return ids, quality, context
