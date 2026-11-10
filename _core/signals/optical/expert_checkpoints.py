"""Independent expert EMA snapshot bank, atomically embedded in last.pt.

No validation tensor or label is retained. Scores are supplied exclusively by
inner validation. Snapshots have independent epochs and are stitched only
because the model explicitly declares disjoint expert state ownership.
"""
from __future__ import annotations
import copy
import math
import torch


class ExpertCheckpointBank:
    def __init__(self, model, min_delta=1e-4):
        self.keys = {name: tuple(model.expert_state_keys(name)) for name in ("feature", "raw")}
        flat = [key for keys in self.keys.values() for key in keys]
        if len(flat) != len(set(flat)):
            raise ValueError("Expert state ownership overlaps")
        template = model.state_dict()
        self.tensor_contract = {key: (tuple(template[key].shape), template[key].dtype) for key in flat}
        self.min_delta = float(min_delta)
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Expert min_delta must be finite and nonnegative")
        self.items = {}

    def update(self, model, selections, epoch):
        if set(selections) != set(self.keys):
            raise ValueError("Both expert selection records are required")
        current = model.state_dict()
        improved = {}
        for name, keys in self.keys.items():
            row = selections[name]
            score = tuple(float(value) for value in row["score"])
            if len(score) != 3 or not all(math.isfinite(value) for value in score):
                raise ValueError("Expert score must be finite [mean rule F1, AP, -BCE]")
            previous = self.items.get(name)
            better = previous is None
            if previous is not None:
                old = tuple(previous["score"])
                better = score[0] > old[0] + self.min_delta or (
                    score[0] >= old[0] and abs(score[0] - old[0]) <= self.min_delta and score[1:] > old[1:])
            if better:
                self.items[name] = {"epoch": int(epoch), "score": list(score), "stalled": 0,
                    "metrics": copy.deepcopy(row["metrics"]),
                    "state": {key: current[key].detach().cpu().clone() for key in keys}}
            else:
                self.items[name]["stalled"] += 1
            improved[name] = better
        return improved

    def all_stalled(self, patience):
        return len(self.items) == len(self.keys) and all(v["stalled"] >= patience for v in self.items.values())

    def epochs(self):
        return {name: value["epoch"] for name, value in self.items.items()}

    def summary(self):
        return {name: {key: copy.deepcopy(value[key]) for key in ("epoch", "score", "stalled", "metrics")}
                for name, value in self.items.items()}

    def compose(self, template):
        if set(self.items) != set(self.keys):
            raise ValueError("Cannot compose incomplete expert checkpoints")
        result = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
        for name, keys in self.keys.items():
            for key in keys:
                saved = self.items[name]["state"][key]
                if saved.shape != result[key].shape or saved.dtype != result[key].dtype:
                    raise ValueError("Expert snapshot tensor contract changed")
                result[key] = saved.clone()
        return result

    def state_dict(self):
        return {"format": "independent-expert-bank-v1", "min_delta": self.min_delta,
                "keys": self.keys, "items": copy.deepcopy(self.items)}

    def load_state_dict(self, state):
        if state.get("format") != "independent-expert-bank-v1" or state.get("min_delta") != self.min_delta:
            raise ValueError("Expert checkpoint selection contract changed on resume")
        if {key: tuple(value) for key, value in state["keys"].items()} != self.keys:
            raise ValueError("Expert snapshot state ownership changed on resume")
        if set(state["items"]) != set(self.keys):
            raise ValueError("Resume snapshot bank incomplete")
        for name, value in state["items"].items():
            if set(value["state"]) != set(self.keys[name]) or value["epoch"] < 1 or value["stalled"] < 0:
                raise ValueError("Invalid expert snapshot metadata")
            if len(value["score"]) != 3 or not all(math.isfinite(float(s)) for s in value["score"]):
                raise ValueError("Invalid saved expert score")
            # Reject incompatible snapshots before any training can resume,
            # rather than waiting until a later compose/save operation.
            for key, tensor in value["state"].items():
                shape, dtype = self.tensor_contract[key]
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise ValueError(f"Saved expert snapshot tensor contract changed: {key}")
        self.items = copy.deepcopy(state["items"])
