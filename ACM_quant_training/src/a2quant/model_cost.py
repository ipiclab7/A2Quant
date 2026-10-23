"""State explicitly what is measured and what is a theoretical bit-cost estimate."""

import weakref
import torch
from torch import nn
from .quantization import (AdaptiveQuantizer, QuantizedConv2d, QuantizedConvTranspose2d,
                           quantizer_manifest, bit_statistics)


def model_cost(model, image):
    precision, hooks, layers, weight_widths = {}, [], [], {}
    training_flags = {module: module.training for module in model.modules()}

    def remember(value, width):
        if torch.is_tensor(value):
            precision[id(value)] = (weakref.ref(value), width)

    def width_of(value):
        item = precision.get(id(value))
        return item[1] if item is not None and item[0]() is value else 32

    def after(name, module, inputs, output):
        value = inputs[0]
        input_bits = width_of(value)
        if isinstance(module, AdaptiveQuantizer):
            remember(output, int(module.bitwidth().item()) if module.enabled else input_bits)
            return
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            quantizer = getattr(module, "weight_quantizer", None)
            bits = int(quantizer.bitwidth().item()) if quantizer is not None and quantizer.enabled else 32
            weight_widths[id(module.weight)] = bits
            if isinstance(module, nn.ConvTranspose2d):
                macs = value.numel() * module.out_channels // module.groups
            else:
                macs = output.numel() * module.in_channels // module.groups
            macs *= module.kernel_size[0] * module.kernel_size[1]
            layers.append({"name": name, "macs": macs, "weight_bits": bits, "input_bits": input_bits,
                           "bops": macs * bits * input_bits})
            remember(output, 32)
        elif isinstance(module, (nn.MaxPool2d, nn.Identity)) or isinstance(module, nn.Dropout) and not module.training:
            remember(output, input_bits)
        elif not list(module.children()):
            remember(output, 32)

    try:
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, (AdaptiveQuantizer, nn.Conv2d, nn.ConvTranspose2d)) or not list(module.children()):
                hooks.append(module.register_forward_hook(lambda layer, inputs, output, n=name: after(n, layer, inputs, output)))
        remember(image, 32)
        with torch.no_grad():
            output = model(image)
        if not torch.isfinite(output).all().item():
            raise ValueError("Nonfinite profiling output")
    finally:
        for hook in hooks:
            hook.remove()
        for module, flag in training_flags.items():
            module.training = flag
    macs, bops = sum(r["macs"] for r in layers), sum(r["bops"] for r in layers)
    fp_bytes = sum(p.numel() * 4 for p in model.parameters())
    packed_bytes = sum(p.numel() * weight_widths.get(id(p), 32) / 8 for p in model.parameters())
    buffers = sum(p.numel() * p.element_size() for p in model.buffers())
    return {"input_shape": list(image.shape), "bits": bit_statistics(model),
            "parameters": {"fp32_bytes": fp_bytes, "ideal_packed_bytes": packed_bytes,
                           "buffer_bytes": buffers, "ideal_compression": fp_bytes / packed_bytes},
            "compute": {"conv_transpose_macs": macs, "structural_flops_2mac": 2 * macs,
                        "fp32_reference_bops": macs * 32 * 32, "conservative_bops": bops,
                        "bop_reduction_fraction": 1. - bops / (macs * 32 * 32)},
            "conventions": {"storage": "Theoretical packing estimate, not actual checkpoint bytes or deployed model size.",
                "flops": "Two FLOPs per Conv/ConvTranspose MAC; other operations excluded. Quantization does not change this structural count.",
                "bops": "MAC*Wbits*tracked input bits; untracked functions, BN and unfused arithmetic default FP32. This is not the paper's unspecified FLOPs metric.",
                "transpose": "Input-elements*Cout/groups*kernel-area, including cropped boundary contributions.",
                "latency": "Not measured. No integer inference backend is used by this profiler."},
            "layers": layers, "quantizers": quantizer_manifest(model)}
