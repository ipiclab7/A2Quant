"""Pinned PyTorch ACM quantization adapter; vendor and legacy core remain unchanged.

Coverage and dynamic alpha envelope are documented project choices, not author code.
"""
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from a2quant.quantization import (AdaptiveQuantizer, QuantizedConv2d,
    QuantizedConvTranspose2d, named_quantizers, component_role, constrain_quantizers)
from upstream_pytorch_bridge import load_source_components, verify_sources


class SourceQuantizedReLU(nn.Module):
    def __init__(self, inplace, activation_bit, min_bit, max_bit):
        super().__init__()
        self.inplace = inplace
        self.quantizer = AdaptiveQuantizer(False, activation_bit, 6., min_bit, max_bit)

    def forward(self, x):
        return self.quantizer(F.relu(x, inplace=self.inplace))


def source_residual_forward(self, x):
    residual = x
    x = self.body(x)
    if self.downsample:
        residual = self.downsample(residual)
    return self.output_quantizer(F.relu(x + residual, True))


def convert_source(model, weight_bit=8., activation_bit=8., min_bit=2, max_bit=14):
    inventory = []
    def add(path, kind, q, actual):
        q.role = component_role(actual)
        inventory.append(dict(path=path, kind=kind, role=q.role, quantizer_path=actual))
    def visit(parent, prefix=''):
        for name, child in list(parent.named_children()):
            path = prefix + name
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                if hasattr(child, 'weight_quantizer'):
                    raise ValueError('Already converted: ' + path)
                child.__class__ = QuantizedConv2d if isinstance(child, nn.Conv2d) else QuantizedConvTranspose2d
                child.weight_quantizer = AdaptiveQuantizer(True, weight_bit,
                    max(child.weight.detach().abs().max().item(), 1e-6), min_bit, max_bit)
                add(path, 'weight', child.weight_quantizer, path + '.weight_quantizer')
            elif isinstance(child, nn.ReLU):
                replacement = SourceQuantizedReLU(child.inplace, activation_bit, min_bit, max_bit)
                setattr(parent, name, replacement)
                add(path, 'activation', replacement.quantizer, path + '.quantizer')
            else:
                visit(child, path + '.')
                if child.__class__.__name__ == 'ResidualBlock':
                    child.output_quantizer = AdaptiveQuantizer(False, activation_bit, 6., min_bit, max_bit)
                    child.forward = types.MethodType(source_residual_forward, child)
                    add(path + '.forward:F.relu_after_add', 'activation', child.output_quantizer,
                        path + '.output_quantizer')
    visit(model)
    model.quantization_coverage = sorted(inventory, key=lambda x:(x['kind'], x['path']))
    return model


def build_source(config, root):
    verify_sources(config, Path(root))
    cls, _, _, initialize = load_source_components(Path(root))
    model = cls(**config['model'])
    model.apply(lambda m: initialize(None, m))
    return model


@torch.no_grad()
def constrain_weight_envelope(model, enabled=True):
    """After each update, cap learned W alpha at current master absmax.

    Alpha can still learn clipping below absmax. No beta/bit floor change, no
    optimizer-state reset. Zero/tiny masters are separately reported.
    """
    constrain_quantizers(model)
    changed = []
    if enabled:
        for name, layer in model.named_modules():
            if hasattr(layer, 'weight_quantizer'):
                q = layer.weight_quantizer
                maximum = layer.weight.detach().abs().amax().clamp_min(1e-6)
                if q.alpha > maximum:
                    changed.append(name)
                q.alpha.copy_(torch.minimum(q.alpha, maximum))
    return changed


@torch.no_grad()
def tensor_health(value, q):
    x = value.detach()
    scale = q.scale()
    lower = -q.alpha.clamp_min(1e-6) if q.signed else torch.zeros_like(q.alpha)
    codes = (x.clamp(min=lower, max=q.alpha.clamp_min(1e-6)) / scale).round()
    reconstructed = codes * scale
    master_nonzero = int(torch.count_nonzero(x))
    quant_nonzero = int(torch.count_nonzero(codes))
    maximum = x.abs().amax().item()
    return dict(elements=x.numel(), master_nonzero=master_nonzero,
        quantized_nonzero=quant_nonzero, unique_codes=int(codes.unique().numel()),
        nonzero_master_quantized_all_zero=master_nonzero > 0 and quant_nonzero == 0,
        master_all_zero=master_nonzero == 0, alpha=q.alpha.item(), beta=q.beta.item(),
        bits=int(q.bitwidth().item()), step=scale.item(), absmax=maximum,
        half_step_over_absmax=scale.item()/2/maximum if maximum else None,
        relative_mse=((reconstructed-x).square().sum()/x.square().sum().clamp_min(1e-30)).item(),
        clipped_fraction=((x < lower) | (x > q.alpha.clamp_min(1e-6))).float().mean().item(),
        finite=bool(torch.isfinite(x).all() and torch.isfinite(reconstructed).all()))


@torch.no_grad()
def weight_health(model):
    rows = [dict(name=n, **tensor_health(layer.weight, layer.weight_quantizer))
            for n, layer in model.named_modules() if hasattr(layer, 'weight_quantizer')]
    return dict(weight_layers=len(rows), quantized_dead_weight_layers=sum(
        x['nonzero_master_quantized_all_zero'] for x in rows), weight_rows=rows)


def configure_training(config, root, output):
    import json, tarfile
    import a2quant.training as training
    from a2quant.stability import stage_consistency as original_consistency, bn_digest
    root, output = Path(root).resolve(), Path(output).resolve()
    verify_sources(config, root)
    def constrain(model):
        constrain_weight_envelope(model, config['weight_alpha_policy']=='current_absmax_upper_envelope')
    def consistency(model, images):
        report=original_consistency(model,images)
        modes={m:m.training for m in model.modules()};before=bn_digest(model);hooks=[];activations=[]
        try:
            model.eval()
            for name,q in named_quantizers(model):
                if not q.signed:
                    def capture(module,args,name=name):
                        activations.append(dict(name=name,**tensor_health(args[0],module)))
                    hooks.append(q.register_forward_pre_hook(capture))
            with torch.no_grad(): model(images)
        finally:
            for hook in hooks:hook.remove()
            for m,mode in modes.items():m.training=mode
        assert bn_digest(model)==before
        model.quantization_activation_health=activations
        report['health_probe_source']='same fixed four TRAIN images; BN/dropout eval; no test'
        return report
    def health(model):
        report=weight_health(model)
        report['activation_rows']=getattr(model,'quantization_activation_health',[])
        report['weight_alpha_policy']=config['weight_alpha_policy']
        return report
    def lr(optimizer,cfg,epoch):
        factor=1. if epoch==1 else pow(1-float(epoch-1)/(cfg['epochs']+1),.9)
        for group in optimizer.param_groups:
            base=cfg[group['name']+'_lr']
            floor=base*1e-4
            group['lr']=floor+(base-floor)*factor
    training.constrain_quantizers=constrain
    training.stage_consistency=consistency
    training.range_health=health
    training.update_learning_rates=lr
