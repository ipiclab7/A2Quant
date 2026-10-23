"""Generic weight-range safeguards and numeric diagnostics."""
import torch
from .quantizers import constrain_quantizers

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
