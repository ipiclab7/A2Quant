"""Use the pinned acm-pytorch model, initializer, criterion and LR update.

Upstream files remain byte-identical. Only the data/split, run recording,
checkpoint/resume and unified final evaluation adapters are project code.
"""
import ast
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tarfile
import types


def verify_sources(config, root):
    for relative, expected in config['source_file_sha256'].items():
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Upstream/adapter source changed: ' + relative)


def load_source_components(root):
    import torch
    import torch.nn as nn
    source = root / 'reference/acm-pytorch-source'
    package_name = '_pinned_acm_pytorch_model'
    package = types.ModuleType(package_name)
    package.__path__ = [str(source / 'model')]
    sys.modules[package_name] = package
    model = importlib.import_module(package_name + '.segmentation')
    loss = importlib.import_module(package_name + '.loss')
    spec = importlib.util.spec_from_file_location('_pinned_acm_lr', source / 'utils/lr_scheduler.py')
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    # Import only the audited initializer method, avoiding Tensorboard and the
    # upstream hard-coded data paths/whole-model pickle training entry point.
    parsed = ast.parse((source / 'train.py').read_text(encoding='utf8'))
    trainer = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'weight_init')
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(module, str(source / 'train.py'), 'exec'), namespace)
    return model.ASKCResUNet, loss.SoftLoULoss(), scheduler.adjust_learning_rate, namespace['weight_init']



