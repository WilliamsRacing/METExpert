
from __future__ import annotations














import torch

import torch.nn.functional as F

















def training_loss(out, batch, core_frames=16, role_weight=.2, dice_weight=.2,
                  boundary_weight=.1, minimum_supervision_coverage=.5):
    """Unknown-role positives supervise the sum, never an invented role."""
    logits = out["logits"].float()
    valid = (batch["label_mask"].bool() & out["valid"]
             & (batch["point_mask"].float().mean(-1) >= minimum_supervision_coverage))
    frames = logits.shape[1]
    start = max(0, (frames - core_frames) // 2)
    core = torch.zeros_like(valid)
    core[:, start:start + core_frames] = True
    valid = valid & core
    fraction = batch["target_fraction"].float().clamp(0, 1)
    binary_logit = torch.logsumexp(logits[..., 1:], -1) - logits[..., 0]
    if not bool(valid.any()):
        zero = logits.sum() * 0
        return zero, dict(total=0., event_bce=0., role=0., dice=0., boundary=0., frames=0)
    bce = F.binary_cross_entropy_with_logits(binary_logit[valid], fraction[valid])
    role_mask = valid & ((batch["target"] == 1) | (batch["target"] == 2))
    role = (F.cross_entropy(logits[..., 1:][role_mask], batch["target"][role_mask].long() - 1)
            if bool(role_mask.any()) else logits.sum() * 0)
    probability = binary_logit.sigmoid()
    p = torch.where(valid, probability, torch.zeros_like(probability))
    y = torch.where(valid, fraction, torch.zeros_like(fraction))
    # Pure-background samples remain governed by BCE; Dice applies to meal bags.
    has_meal = y.sum(-1) > 0
    dice_per = 1 - (2 * (p * y).sum(-1) + 1) / (p.sum(-1) + y.sum(-1) + 1)
    dice = dice_per[has_meal].mean() if bool(has_meal.any()) else logits.sum() * 0
    boundary_logits = out["boundary_logits"].float()
    boundary = F.binary_cross_entropy_with_logits(
        boundary_logits[valid], batch["boundary"].float()[valid],
        pos_weight=boundary_logits.new_tensor([8., 8.]))
    loss = bce + role_weight * role + dice_weight * dice + boundary_weight * boundary
    return loss, dict(total=float(loss.detach()), event_bce=float(bce.detach()),
                      role=float(role.detach()), dice=float(dice.detach()),
                      boundary=float(boundary.detach()), frames=int(valid.sum()))


def on_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def model_forward(model, batch):
    return model(batch["imu"], batch["point_mask"], batch["frame_quality"], batch["segment_ids"])








def worker_init(worker_id):
    torch.set_num_threads(1)


def augment_training(batch, normalization, probability=.1):
    """Small consistent gain and missing-frame augmentation, never label-driven."""
    x = batch["imu"]
    mean = x.new_tensor(normalization["mean"])[None, None, :, None]
    std = x.new_tensor(normalization["std"])[None, None, :, None]
    physical = x * std + mean
    gain_acc = 1 + .03 * torch.randn(x.shape[0], 1, 1, 1, device=x.device)
    gain_gyro = 1 + .03 * torch.randn(x.shape[0], 1, 1, 1, device=x.device)
    gain = torch.cat((gain_acc.expand(-1, 1, 3, 1), gain_gyro.expand(-1, 1, 3, 1),
                      gain_acc.expand(-1, 1, 6, 1)), dim=2)
    augmented = (physical * gain - mean) / std
    mask = batch["point_mask"].clone()
    # A missing whole frame creates a real context break inside the network.
    selected = torch.rand(x.shape[0], device=x.device) < probability
    frame = torch.randint(x.shape[1], (x.shape[0],), device=x.device)
    for i in selected.nonzero(as_tuple=False).flatten().tolist():
        mask[i, frame[i]] = False
    result = dict(batch)
    result["point_mask"] = mask
    result["imu"] = torch.where(mask[:, :, None], augmented, torch.zeros_like(augmented))
    return result












