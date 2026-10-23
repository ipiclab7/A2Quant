"""Reproducible ACM/SMARt/AQP training, checkpointing, and final test evaluation."""

import hashlib
import json
import os
from pathlib import Path
import random
import tarfile
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SirstDataset, seed_worker
from .models import ASKCResUNet
from .quantization import (AdaptiveQuantizer, convert_model, named_quantizers, set_stage,
                           constrain_quantizers, bit_statistics, quantizer_manifest)
from .objectives import soft_iou_loss, smart_loss
from .splits import load_verified_splits
from .evaluation import DetectionMetrics
from .model_cost import model_cost
from .stability import training_mode, bn_digest, range_health, stage_consistency


def stage_for_epoch(epoch, preq_epochs):
    if epoch < 1:
        raise ValueError("Epochs are one-based")
    return "preq" if epoch <= preq_epochs else "adaqat"


def build_model(config):
    model = ASKCResUNet(**config["model"])
    return convert_model(model, **config["quantization"])


class OptimizerPair:
    """Baseline network optimizer and separately specified quantizer optimizer."""
    def __init__(self, network, quantization):
        self.network, self.quantization = network, quantization

    @property
    def param_groups(self):
        return self.network.param_groups + self.quantization.param_groups

    def zero_grad(self, set_to_none=True):
        self.network.zero_grad(set_to_none=set_to_none)
        self.quantization.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.network.step()
        self.quantization.step()

    def state_dict(self):
        return {"network": self.network.state_dict(), "quantization": self.quantization.state_dict()}

    def load_state_dict(self, state):
        self.network.load_state_dict(state["network"])
        self.quantization.load_state_dict(state["quantization"])


def build_optimizer(model, config):
    quantization = [parameter for _, q in named_quantizers(model) for parameter in (q.alpha, q.beta)]
    quant_ids = {id(parameter) for parameter in quantization}
    network = [parameter for parameter in model.parameters() if id(parameter) not in quant_ids]
    if config["network_optimizer"] != "Adagrad" or config["quantization_optimizer"] != "Adam":
        raise ValueError("Only the recorded Adagrad/Adam configuration is implemented")
    return OptimizerPair(
        torch.optim.Adagrad([{"params": network, "lr": config["network_lr"],
                             "weight_decay": config["weight_decay"], "name": "network"}]),
        torch.optim.Adam([{"params": quantization, "lr": config["quantization_lr"],
                          "weight_decay": 0., "name": "quantization"}]))


def update_learning_rates(optimizer, config, epoch):
    schedule = config["schedule"]
    warmup, total = schedule["warmup_epochs"], config["epochs"]
    if epoch <= warmup:
        factor = epoch / max(1, warmup)
    else:
        floor = schedule["min_lr_factor"]
        factor = floor + (1. - floor) * (1. - (epoch - warmup) / (total - warmup + 1)) ** schedule["power"]
    for group in optimizer.param_groups:
        group["lr"] = config[group["name"] + "_lr"] * factor


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((root / "src/a2quant").rglob("*.py")):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def validate(model, loader, device, max_batches=None, objects=False):
    model.eval()  # Always assess formal quantization, including during PreQ.
    loss_sum, count = 0., 0
    intersections, unions = [], []
    metric = DetectionMetrics() if objects else None
    with torch.no_grad():
        for index, (images, targets, _) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Nonfinite validation output")
            loss_sum += soft_iou_loss(logits, targets).item() * len(images)
            count += len(images)
            predicted, truth = logits.sigmoid() > .5, targets > .5
            intersection = (predicted & truth).sum((1, 2, 3)).cpu().tolist()
            union = (predicted | truth).sum((1, 2, 3)).cpu().tolist()
            intersections.extend(intersection); unions.extend(union)
            if metric is not None:
                for prediction, target in zip(predicted[:, 0].cpu().numpy(), truth[:, 0].cpu().numpy()):
                    metric.update(prediction, target)
    report = {"images": count, "loss": loss_sum / count,
              "IoU": sum(intersections) / max(sum(unions), 1),
              "nIoU": sum(i / u if u else 0. for i, u in zip(intersections, unions)) / count}
    if metric is not None:
        report.update(metric.result())
    return report


def run(config, root, output, device="cuda:0", smoke=False, stop_after=None, resume=False):
    root, output = Path(root), Path(output)
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("This paper run requires a CUDA device")
    if (config["epochs"], config["preq_epochs"], config["image_size"], config["batch_size"]) != (1500, 300, 512, 16):
        raise ValueError("Paper schedule/input/batch must remain 1500/300/512/16; use --smoke for diagnostics")
    if config["dtype"] != "float32" or config["amp"] or config["allow_tf32"]:
        raise ValueError("This implementation records FP32 with AMP/TF32 disabled")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = config["training_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dataset_config, splits = load_verified_splits(config["data_root"], root / config["dataset_config"])
    dataset_provenance = json.loads((Path(config["data_root"]) / "provenance.json").read_text())
    identity = {"config": config, "dataset_config": dataset_config,
                "split_hashes": dataset_provenance["split_file_sha256"], "code_sha256": source_hash(root), "smoke": smoke}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    total, boundary = (2, 1) if smoke else (config["epochs"], config["preq_epochs"])
    max_batches = 2 if smoke else None
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty run directory; use --resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    workers = 0 if smoke else config["num_workers"]
    train_loader = DataLoader(SirstDataset(config["data_root"], splits["train"], True),
        batch_size=16, shuffle=True, num_workers=workers, pin_memory=True, generator=generator, worker_init_fn=seed_worker)
    val_loader = DataLoader(SirstDataset(config["data_root"], splits["val"]),
        batch_size=16, shuffle=False, num_workers=workers, pin_memory=True, worker_init_fn=seed_worker)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    bn_policy = config.get("bn_policy", "standard")
    probe_images = None
    if config.get("range_consistency_checks", False):
        probe_data = SirstDataset(config["data_root"], splits["train"][:4])
        probe_images = torch.stack([probe_data[i][0] for i in range(len(probe_data))]).to(device)
    start_epoch, best_iou, best_epoch, elapsed_previous = 1, -1., None, 0.
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Resume config/code/split differs from saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_iou, best_epoch = state["epoch"] + 1, state["best_iou"], state["best_epoch"]
        elapsed_previous = state["elapsed_training_seconds"]
        restore_rng(state["rng"], generator)
        final_path = output / "final-test.json"
        if final_path.exists():
            if json.loads(final_path.read_text(encoding="utf-8"))["signature"] != signature:
                raise ValueError("Existing test result has a different run signature")
            print("RUN ALREADY COMPLETE; existing final test retained", flush=True)
            return
    else:
        (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Preserve the exact implementation and configuration used by this run.
        with tarfile.open(output / "source-snapshot.tar.gz", "w:gz") as archive:
            files = list((root / "src/a2quant").rglob("*.py"))
            files += list((root / "configs").glob("*.json"))
            files += [root / "scripts/train.py", root / "LICENSE"]
            for path in sorted(files):
                archive.add(path, arcname=str(path.relative_to(root)))
    metadata = dict(identity, signature=signature, result_type="smoke_only" if smoke else "new_training_with_documented_unverified_choices",
                    torch_version=torch.__version__, cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    effective_total_epochs=total, effective_preq_epochs=boundary, test_accessed=False)
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    end_epoch = min(total, stop_after) if stop_after is not None else total
    frozen_bn_hash = bn_digest(model) if start_epoch > boundary and bn_policy == "freeze_after_preq" else None
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_started = time.monotonic()
        stage = stage_for_epoch(epoch, boundary)
        set_stage(model, stage)
        bn_frozen = training_mode(model, epoch, boundary, bn_policy)
        if bn_frozen and frozen_bn_hash is None:
            frozen_bn_hash = bn_digest(model)
        update_learning_rates(optimizer, config, epoch)
        sums = {"task": 0., "regularization": 0., "total": 0.}
        samples, task_gradient_probe = 0, None
        for batch, (images, targets, _) in enumerate(train_loader):
            if max_batches is not None and batch >= max_batches:
                break
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            task = soft_iou_loss(logits, targets)
            regularization, terms = smart_loss(model, config["targets"], config["lambda_encoder"], config["lambda_decoder"])
            loss = task + regularization
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite training loss")
            if smoke and batch == 0:
                quantizers = named_quantizers(model)
                gradients = torch.autograd.grad(task, [q.beta for _, q in quantizers], retain_graph=True, allow_unused=True)
                task_gradient_probe = {kind: sum(abs(g.item()) for (_, q), g in zip(quantizers, gradients)
                    if g is not None and q.signed == (kind == "weight")) for kind in ("weight", "activation")}
                if not all(np.isfinite(value) and value > 0 for value in task_gradient_probe.values()):
                    raise RuntimeError("Task gradient does not reach W/A beta: " + str(task_gradient_probe))
            loss.backward()
            finite_gradients = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
            if not torch.stack(finite_gradients).all().item():
                raise RuntimeError("Nonfinite parameter gradient")
            optimizer.step()
            constrain_quantizers(model)
            for name, value in (("task", task), ("regularization", regularization), ("total", loss)):
                sums[name] += value.item() * len(images)
            samples += len(images)
        bn_hash = bn_digest(model)
        if bn_frozen and bn_hash != frozen_bn_hash:
            raise RuntimeError("Frozen BN buffers changed during AdaQAT")
        consistency = stage_consistency(model, probe_images) if probe_images is not None else None
        validation = validate(model, val_loader, device, max_batches)
        improved = stage == "adaqat" and validation["IoU"] > best_iou
        if improved:
            best_iou, best_epoch = validation["IoU"], epoch
        row = {"epoch": epoch, "stage": stage, "train_samples": samples,
               "train": {key: value / samples for key, value in sums.items()}, "validation": validation,
               "bits": bit_statistics(model), "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
               "task_beta_gradient_probe": task_gradient_probe, "epoch_seconds": time.monotonic() - epoch_started,
               "cuda_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
               "best_adaqat_epoch": best_epoch}
        row.update({"bn": {"policy": bn_policy, "frozen": bn_frozen, "buffer_sha256": bn_hash},
                    "range_health": range_health(model) if probe_images is not None else None,
                    "stage_consistency": consistency})
        state = {"epoch": epoch, "stage": stage, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": rng_state(generator), "signature": signature, "config": config,
                 "best_iou": best_iou, "best_epoch": best_epoch,
                 "elapsed_training_seconds": elapsed_previous + time.monotonic() - started}
        atomic_save(state, output / "last.pt")
        if improved:
            atomic_save(state, output / "best_adaqat.pt")
        if epoch == boundary:
            atomic_save(state, output / "preq_final.pt")
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output / "quantizers-latest.json").write_text(json.dumps(quantizer_manifest(model), indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if end_epoch < total:
        print("STOPPED_AT_EPOCH_LIMIT; resume supported; test set untouched", flush=True)
        return
    if smoke:
        (output / "smoke-complete.json").write_text(json.dumps({"passed": True, "test_accessed": False, "stages": ["preq", "adaqat"]}))
        print("SMOKE COMPLETE; not a paper result; test set untouched", flush=True)
        return
    best = torch.load(output / "best_adaqat.pt", map_location="cpu")
    model.load_state_dict(best["model"])
    set_stage(model, "adaqat")
    costs = model_cost(model, torch.zeros(1, 3, 512, 512, device=device))
    costs["actual_checkpoint_bytes"] = (output / "best_adaqat.pt").stat().st_size
    (output / "final-quantization-cost.json").write_text(json.dumps(costs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_loader = DataLoader(SirstDataset(config["data_root"], splits["test"]), batch_size=16, shuffle=False,
                             num_workers=workers, pin_memory=True, worker_init_fn=seed_worker)
    test = validate(model, test_loader, device, objects=True)
    test.update({"selected_epoch": best["epoch"], "validation_IoU_at_selection": best["best_iou"],
                 "split_sha256": dataset_provenance["split_file_sha256"]["test"], "bits": bit_statistics(model),
                 "authors_exact_configuration_verified": False, "signature": signature,
                 "elapsed_training_seconds": elapsed_previous + time.monotonic() - started,
                 "result_type": metadata["result_type"]})
    (output / "final-test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata["test_accessed"] = True
    metadata["training_complete"] = True
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TRAINING AND FINAL TEST COMPLETE", json.dumps(test, ensure_ascii=False), flush=True)
