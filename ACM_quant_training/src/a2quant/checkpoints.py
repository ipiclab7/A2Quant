"""Restricted loading for the three user-provided historical checkpoints."""

import hashlib
import importlib
import io
import pickle
import pickletools
import sys
import types
import zipfile
from pathlib import Path

import torch


ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"), ("__builtin__", "set"), ("builtins", "set"),
    ("torch", "FloatStorage"), ("torch", "LongStorage"), ("torch", "Size"),
    ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_tensor_v2"),
    ("torch.nn.modules.container", "Sequential"),
    ("torch.nn.modules.batchnorm", "BatchNorm2d"),
    ("torch.nn.modules.pooling", "MaxPool2d"),
    ("torch.nn.modules.pooling", "AdaptiveAvgPool2d"),
    ("torch.nn.modules.activation", "Sigmoid"),
    ("torch.nn.modules.activation", "ReLU"),
    ("torch.nn.modules.dropout", "Dropout"),
    ("torch.nn.modules.conv", "Conv2d"),
    ("torch.nn.modules.conv", "ConvTranspose2d"),
    ("model.segmentation_nipq", "ASKCResUNet"),
    ("model.segmentation_nipq", "ASKCResNetFPN"),
    ("model.segmentation_nipq", "ResidualBlock"),
    ("model.segmentation_nipq", "_FCNHead"),
    ("model.nipq", "Q_Conv2d"), ("model.nipq", "Q_ConvTranspose2d"),
    ("model.nipq", "Q_Linear"), ("model.nipq", "Q_ReLU"),
    ("model.nipq", "Q_Sym"), ("model.nipq", "Q_HSwish"),
    ("model.fusion_nipq", "AsymBiChaFuseReduce"),
    ("model.fusion_nipq", "BiLocalChaFuseReduce"),
    ("model.fusion_nipq", "BiGlobalChaFuseReduce"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")
        return super().find_class(module, name)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_checkpoint(path, project_root, device="cpu"):
    path = Path(path).resolve()
    # Inspect protocol-2 globals before loading any checkpoint objects.
    with zipfile.ZipFile(path) as archive:
        payload_names = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(payload_names) != 1:
            raise ValueError("Expected one PyTorch data.pkl entry")
        for op, arg, _ in pickletools.genops(archive.read(payload_names[0])):
            if op.name == "STACK_GLOBAL":
                raise ValueError("Unsupported pickle protocol: STACK_GLOBAL")
            if op.name == "GLOBAL" and tuple(arg.split(" ", 1)) not in ALLOWED_GLOBALS:
                raise ValueError(f"Unsupported checkpoint global: {arg}")
    sys.dont_write_bytecode = True
    vendor_root = Path(project_root).resolve() / "legacy_reference"
    if "model" in sys.modules:
        loaded_path = Path(sys.modules["model"].__file__).resolve()
        if vendor_root not in loaded_path.parents:
            raise RuntimeError("Conflicting model compatibility namespace")
    sys.path.insert(0, str(vendor_root))
    importlib.import_module("model.segmentation_nipq")
    restricted = types.ModuleType("a2quant_restricted_pickle")
    restricted.Unpickler = RestrictedUnpickler
    restricted.load = lambda source, **kwargs: RestrictedUnpickler(source, **kwargs).load()
    restricted.loads = lambda data, **kwargs: RestrictedUnpickler(io.BytesIO(data), **kwargs).load()
    model = torch.load(str(path), map_location=device, pickle_module=restricted)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Historical checkpoint did not contain an nn.Module")
    return model
