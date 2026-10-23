"""Documented range/BN safeguards for the second reproduction run."""
import hashlib
import torch
from torch import nn
from .quantization import named_quantizers, set_stage


def training_mode(model, epoch, preq_epochs, policy="standard"):
    if policy not in {"standard", "freeze_after_preq"}:
        raise ValueError("Unknown BN policy: " + policy)
    model.train()
    frozen = policy == "freeze_after_preq" and epoch > preq_epochs
    if frozen:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  # Affine parameters still learn through autograd.
    return frozen


def bn_digest(model):
    digest = hashlib.sha256()
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            for key in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(layer, key)
                digest.update((name + "." + key).encode())
                digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def range_health(model):
    rows = []
    for name, layer in model.named_modules():
        if hasattr(layer, "weight_quantizer"):
            q, weight = layer.weight_quantizer, layer.weight.detach()
            rows.append({"name": name, "alpha": q.alpha.item(), "bits": int(q.bitwidth().item()),
                         "clipped_fraction": (weight.abs() > q.alpha).float().mean().item(),
                         "weight_absmax": weight.abs().max().item()})
    return {"weight_layers": len(rows), "layers_over_95pct_clipped": sum(r["clipped_fraction"] > .95 for r in rows),
            "worst_weight_layers": sorted(rows, key=lambda r: -r["clipped_fraction"])[:5]}


@torch.no_grad()
def stage_consistency(model, images):
    """Same training images, fixed BN/dropout: compare only quantizer paths."""
    modes = {m: m.training for m in model.modules()}
    stages = {q: q.stage for _, q in named_quantizers(model)}
    before = bn_digest(model)
    try:
        model.eval()
        set_stage(model, "preq")
        for _, q in named_quantizers(model):
            q.train()
        preq = model(images)
        model.eval()
        set_stage(model, "adaqat")
        formal = model(images)
        maximum = (preq - formal).abs().max().item()
        if not torch.isfinite(preq).all() or not torch.isfinite(formal).all() or maximum > 1e-5:
            raise RuntimeError("PreQ/formal forward mismatch: " + str(maximum))
        return {"maximum_logit_difference": maximum,
                "binary_prediction_disagreements": int(((preq > 0) != (formal > 0)).sum().item()),
                "samples": len(images), "source": "fixed training images; BN/dropout eval"}
    finally:
        for m, training in modes.items():
            m.training = training
        for q, stage in stages.items():
            q.stage = stage
        if bn_digest(model) != before:
            raise RuntimeError("Stage probe mutated BN statistics")
