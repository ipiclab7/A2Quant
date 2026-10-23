"""Module-group bitwidth regularization; task loss is supplied by the caller."""
import math
import torch
from torch.nn import functional as F
from .quantizers import named_quantizers

def log_cosh(value):
    absolute = value.abs()
    return absolute + F.softplus(-2. * absolute) - math.log(2.)


def smart_loss(model, targets, lambda_encoder=.5, lambda_decoder=.5):
    groups = {(role, kind): [] for role in ("encoder", "decoder") for kind in ("weight", "activation")}
    for _, quantizer in named_quantizers(model):
        groups[(quantizer.role, "weight" if quantizer.signed else "activation")].append(quantizer.bitwidth())
    terms = {}
    for (role, kind), values in groups.items():
        if not values:
            raise ValueError("Empty SMARt group: " + role + "/" + kind)
        mean = torch.stack(values).mean()
        deviation = mean / targets[role][kind] - 1.
        terms[role + "_" + kind] = log_cosh(deviation) if role == "encoder" else .5 * deviation.square()
    encoder = terms["encoder_weight"] + terms["encoder_activation"]
    decoder = terms["decoder_weight"] + terms["decoder_activation"]
    return lambda_encoder * encoder + lambda_decoder * decoder, terms
