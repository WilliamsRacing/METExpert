"""Raw events are targets only: no function here changes signal inputs."""
from __future__ import annotations

import numpy as np

from .storage import fingerprint, read_jsonl
from . import LABEL_VERSION


def hand(value):
    value = str(value or "").strip().lower()
    if value in {"left", "l"} or "左" in value:
        return "left"
    if value in {"right", "r"} or "右" in value:
        return "right"
    return None


class LabelPolicy:
    def __init__(self, root):
        from pathlib import Path
        self.events = read_jsonl(Path(root) / "labels" / "events_raw.jsonl")
        self.unknown = read_jsonl(Path(root) / "labels" / "unknown_intervals.jsonl")
        policy_path = Path(root) / "labels" / "label_policy.json"
        import json
        native_policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
        # No table means inference-only; never manufacture background labels.
        self.annotated = (Path(root) / "labels" / "events_raw.jsonl").exists() and native_policy.get("enabled", True)
        self.fingerprint = fingerprint({"version": LABEL_VERSION, "events": self.events, "unknown": self.unknown,
                                        "annotated": self.annotated})

    def participant_events(self, key):
        return [event for event in self.events if event.get("participant_key") == key
                and event.get("start_epoch_ms") is not None and event.get("end_epoch_ms") is not None
                and int(event["end_epoch_ms"]) > int(event["start_epoch_ms"])]

    @staticmethod
    def same_hand(event):
        explicit = event.get("same_hand")
        if explicit in (True, "true", "True"):
            return True
        if explicit in (False, "false", "False"):
            return False
        row = event.get("source_row", {})
        worn, eating = hand(row.get("wearHand")), hand(row.get("dietaryHand"))
        return worn == eating if worn and eating else None

    def targets(self, key, ticks, observed):
        # Rational comparison avoids rounding milliseconds onto a false 15 Hz boundary.
        ticks = np.asarray(ticks, dtype=np.int64)
        all_target = np.zeros(len(ticks), np.float32)
        same_target = np.zeros(len(ticks), np.float32)
        # 0=background, 1=same-hand meal, 2=different-hand meal,
        # -1=unknown hand/unknown interval. This relation is metadata for
        # task-aware IMU weighting; it never changes the all-event target.
        imu_relation = np.zeros(len(ticks), np.int8)
        hand_flags = np.zeros(len(ticks), np.uint8)
        all_mask = np.asarray(observed, bool).copy() & self.annotated
        same_mask = all_mask.copy()
        for event in self.participant_events(key):
            inside = ((ticks * 1000 >= int(event["start_epoch_ms"]) * 15)
                      & (ticks * 1000 < int(event["end_epoch_ms"]) * 15))
            all_target[inside] = 1
            relation = self.same_hand(event)
            if relation is True:
                same_target[inside] = 1
                hand_flags[inside] |= 1
            elif relation is False:
                hand_flags[inside] |= 2
                same_mask[inside] = False
            else:
                hand_flags[inside] |= 4
                same_mask[inside] = False
            if event.get("unknown", False):
                all_mask[inside] = same_mask[inside] = False
                hand_flags[inside] |= 4
        for interval in self.unknown:
            if interval.get("participant_key") == key:
                inside = ((ticks * 1000 >= int(interval["start_epoch_ms"]) * 15)
                          & (ticks * 1000 < int(interval["end_epoch_ms"]) * 15))
                all_mask[inside] = same_mask[inside] = False
        imu_relation[hand_flags == 1] = 1
        imu_relation[hand_flags == 2] = 2
        imu_relation[(hand_flags > 2) | ~all_mask] = -1
        return {"target_all": all_target, "target_same": same_target,
                "imu_relation": imu_relation, "supervision_all": all_mask, "supervision_same": same_mask,
                "evaluation_mask": all_mask.copy()}

    def frame_targets(self, key, starts, observed_fraction):
        starts = np.asarray(starts, np.int64)
        fractions = np.zeros(len(starts), np.float32)
        boundary = np.zeros((len(starts), 2), np.float32)
        # Merge overlapping source labels before calculating overlap.
        intervals = []
        events = self.participant_events(key)
        for event in sorted(events, key=lambda e: int(e["start_epoch_ms"])):
            a, b = int(event["start_epoch_ms"]), int(event["end_epoch_ms"])
            if intervals and a <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(b, intervals[-1][1]))
            else:
                intervals.append((a, b))
        for a, b in intervals:
            fractions += np.maximum(0, np.minimum(starts + 30000, b) - np.maximum(starts, a)) / 30000
        for event in events:
            boundary[:, 0] = np.maximum(boundary[:, 0], ((starts <= int(event["start_epoch_ms"])) &
                                                        (int(event["start_epoch_ms"]) < starts + 30000)))
            boundary[:, 1] = np.maximum(boundary[:, 1], ((starts <= int(event["end_epoch_ms"])) &
                                                        (int(event["end_epoch_ms"]) < starts + 30000)))
        target = np.where(fractions >= .5, 1, np.where(fractions == 0, 0, -1)).astype(np.float32)
        known = (np.asarray(observed_fraction) >= .999) & self.annotated
        for interval in self.unknown + [e for e in events if e.get("unknown", False)]:
            if interval.get("participant_key") == key:
                known &= ~((starts < int(interval["end_epoch_ms"])) & (starts + 30000 > int(interval["start_epoch_ms"])))
        target[~known] = -1
        return {"target": target, "label_weight": known.astype(np.float32),
                "start_target": boundary[:, 0], "end_target": boundary[:, 1],
                "boundary_mask": np.repeat(known[:, None], 2, axis=1),
                "meal_overlap_fraction": fractions}

    def joint_boundary_targets(self, key, starts, observed_fraction):
        """Real endpoint classification plus offsets; never invent window-edge endpoints."""
        starts = np.asarray(starts, np.int64)
        known = (np.asarray(observed_fraction) >= .999) & self.annotated
        events = self.participant_events(key)
        for interval in self.unknown + [e for e in events if e.get("unknown", False)]:
            if interval.get("participant_key") == key:
                known &= ~((starts < int(interval["end_epoch_ms"])) & (starts + 30000 > int(interval["start_epoch_ms"])))
        counts = np.zeros((len(starts), 2), np.int64)
        offsets = np.zeros((len(starts), 2), np.float32)
        for event in events:
            if event.get("unknown", False):
                continue
            for j, endpoint in enumerate(("start_epoch_ms", "end_epoch_ms")):
                value = int(event[endpoint])
                inside = (starts <= value) & (value < starts + 30000)
                counts[inside, j] += 1
                offsets[inside, j] = (value - starts[inside]) / 30000
        return {"joint_boundary_target": (counts > 0).astype(np.float32),
                "joint_boundary_mask": known, "joint_offset_target": offsets,
                "joint_offset_mask": (counts == 1) & known[:, None]}
