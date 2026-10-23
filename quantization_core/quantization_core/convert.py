"""Convert explicit Conv/ReLU modules; caller supplies encoder/decoder roles."""
from torch import nn
from .quantizers import (AdaptiveQuantizer, QuantizedConv2d, QuantizedConvTranspose2d,
                         QuantizedReLU, named_quantizers)

def convert_modules(model, role_resolver, weight_bit=8., activation_bit=8., min_bit=2, max_bit=14):
    def visit(parent, prefix=''):
        for name, child in list(parent.named_children()):
            path = prefix + name
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                if hasattr(child, 'weight_quantizer'):
                    raise ValueError('Already quantized: ' + path)
                role = role_resolver(path)
                if role not in {'encoder', 'decoder'}:
                    raise ValueError('Invalid quantizer role: ' + path)
                child.__class__ = QuantizedConvTranspose2d if isinstance(child, nn.ConvTranspose2d) else QuantizedConv2d
                child.weight_quantizer = AdaptiveQuantizer(True, weight_bit,
                    max(child.weight.detach().abs().max().item(), 1e-6), min_bit, max_bit)
                child.weight_quantizer.role = role
            elif isinstance(child, nn.ReLU):
                if role_resolver(path) not in {'encoder', 'decoder'}:
                    raise ValueError('Invalid quantizer role: ' + path)
                replacement = QuantizedReLU(activation_bit, min_bit, max_bit)
                replacement.quantizer.role = role_resolver(path)
                setattr(parent, name, replacement)
            else:
                visit(child, path + '.')
    visit(model)
    return model

def assign_roles(model, role_resolver):
    for name, quantizer in named_quantizers(model):
        role = role_resolver(name)
        if role not in {'encoder', 'decoder'}:
            raise ValueError('Invalid quantizer role: ' + name)
        quantizer.role = role
    return model
