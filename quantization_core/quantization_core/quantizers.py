"""Backbone-independent low-bit numerical operators and statistics."""
import torch
from torch import nn
from torch.nn import functional as F

def round_pass(value):
    return value + (value.round() - value).detach()


class PreQProxy(torch.autograd.Function):
    """Eq. 18 forward, explicitly POSITIVE surrogate from Eq. 19 backward."""
    @staticmethod
    def forward(ctx, value, scale):
        preliminary_scale = scale.detach()
        error = (value / preliminary_scale).round() * preliminary_scale - value
        error_prime = error * preliminary_scale
        ctx.save_for_backward(error_prime, scale)
        # Algebraically value + error_prime / scale at the current scale. Returning
        # the quantized value directly avoids cancellation at stage boundaries.
        return (value / preliminary_scale).round() * preliminary_scale

    @staticmethod
    def backward(ctx, grad):
        error_prime, scale = ctx.saved_tensors
        grad_scale = (grad * error_prime / scale.square()).sum().reshape_as(scale)
        return grad, grad_scale


class AdaptiveQuantizer(nn.Module):
    def __init__(self, signed, initial_bit=8., alpha=6., min_bit=2, max_bit=14):
        super().__init__()
        if not 2 <= min_bit <= initial_bit <= max_bit <= 16 or alpha <= 0:
            raise ValueError("Invalid quantizer initialization")
        self.beta = nn.Parameter(torch.tensor(float(initial_bit)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.signed, self.min_bit, self.max_bit = signed, min_bit, max_bit
        self.enabled, self.stage = True, "preq"
        self.last_elements_per_sample = 0

    def bitwidth(self):
        return round_pass(self.beta).clamp(self.min_bit, self.max_bit)

    def scale(self):
        exponent = self.bitwidth() - (1 if self.signed else 0)
        return self.alpha.clamp_min(1e-6) / (2. ** exponent - 1.)

    def forward(self, value):
        self.last_elements_per_sample = value[0].numel() if not self.signed else value.numel()
        if not self.enabled:
            return value
        scale = self.scale()
        bound = self.alpha.clamp_min(1e-6)
        lower = -bound if self.signed else torch.zeros_like(bound)
        clipped = torch.maximum(torch.minimum(value, bound), lower)
        if self.training and self.stage == "preq":
            # Eq. 19 applies within the quantization interval. Outside it, use
            # the formal saturation boundary (including its alpha gradient),
            # instead of letting a nominal b-bit proxy use unbounded codes.
            return PreQProxy.apply(clipped, scale)
        return round_pass(clipped / scale) * scale

    @torch.no_grad()
    def constrain(self):
        self.alpha.clamp_(1e-6, 1e4)
        self.beta.clamp_(self.min_bit, self.max_bit)


class QuantizedReLU(nn.Module):
    def __init__(self, initial_bit, min_bit=2, max_bit=14):
        super().__init__()
        self.quantizer = AdaptiveQuantizer(False, initial_bit, 6., min_bit, max_bit)

    def forward(self, value):
        return self.quantizer(F.relu(value, inplace=False))


class QuantizedConv2d(nn.Conv2d):
    def forward(self, value):
        return self._conv_forward(value, self.weight_quantizer(self.weight), self.bias)


class QuantizedConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, value, output_size=None):
        padding = self._output_padding(value, output_size, self.stride, self.padding,
                                       self.kernel_size, 2, self.dilation)
        return F.conv_transpose2d(value, self.weight_quantizer(self.weight), self.bias,
                                  self.stride, self.padding, padding, self.groups, self.dilation)


def named_quantizers(model):
    return [(name, layer) for name, layer in model.named_modules() if isinstance(layer, AdaptiveQuantizer)]


def set_stage(model, stage):
    if stage not in {"preq", "adaqat"}:
        raise ValueError(stage)
    for _, quantizer in named_quantizers(model):
        quantizer.stage = stage


def constrain_quantizers(model):
    for _, quantizer in named_quantizers(model):
        quantizer.constrain()


def quantizer_manifest(model):
    return [{"name": name, "role": quantizer.role, "kind": "weight" if quantizer.signed else "activation",
             "beta": quantizer.beta.item(), "forward_bits": int(quantizer.bitwidth().item()),
             "alpha": quantizer.alpha.item(), "elements": quantizer.last_elements_per_sample}
            for name, quantizer in named_quantizers(model)]


def bit_statistics(model):
    manifest = quantizer_manifest(model)
    if not manifest:
        return {kind: {"module_mean": 32., "continuous_module_mean": 32., "element_mean": 32.}
                for kind in ("weight", "activation")}
    report = {}
    for kind in ("weight", "activation"):
        rows = [r for r in manifest if r["kind"] == kind]
        elements = sum(r["elements"] for r in rows)
        report[kind] = {
            "module_mean": sum(r["forward_bits"] for r in rows) / len(rows),
            "continuous_module_mean": sum(r["beta"] for r in rows) / len(rows),
            "element_mean": sum(r["forward_bits"] * r["elements"] for r in rows) / elements if elements else None}
    return report
