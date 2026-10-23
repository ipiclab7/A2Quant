"""Precision and convolution cost statistics with explicit accounting conventions."""

import math
import weakref

import torch
from torch import nn


ENCODER_ROOTS = {"stem", "layer1", "layer2", "layer3"}
DECODER_ROOTS = {"deconv1", "deconv2", "fuse1", "fuse2", "fuse12", "fuse23", "uplayer1", "uplayer2", "head"}


def component_role(name):
    root = name.split(".", 1)[0]
    if root in ENCODER_ROOTS:
        return "encoder"
    if root in DECODER_ROOTS:
        return "decoder"
    return "unassigned"


def legacy_bits(module):
    continuous = float((2 + 12 * torch.sigmoid(module.bit.detach())).item())
    discrete = int(torch.round(torch.tensor(continuous, dtype=module.bit.dtype)).item())
    return continuous, discrete


def _summary(rows, count_key):
    total = sum(r[count_key] for r in rows)
    if not rows or total == 0:
        return {"components": 0, "elements": total, "continuous_element_mean": None,
                "discrete_element_mean": None, "continuous_component_mean": None,
                "discrete_component_mean": None}
    return {
        "components": len(rows), "elements": total,
        "continuous_element_mean": sum(r["continuous_bits"] * r[count_key] for r in rows) / total,
        "discrete_element_mean": sum(r["discrete_bits"] * r[count_key] for r in rows) / total,
        "continuous_component_mean": sum(r["continuous_bits"] for r in rows) / len(rows),
        "discrete_component_mean": sum(r["discrete_bits"] for r in rows) / len(rows),
    }


def profile_legacy_model(model, image):
    """Profile an eval forward without resetting checkpoint quantization parameters.

    Ordinary operations are FP32 boundaries for the conservative BOP estimate.
    MaxPool and identity quantizer bypasses preserve known logical precision.
    Untraced functional operations default to FP32; no fusion is assumed.
    """
    if image.dtype != torch.float32:
        raise ValueError("Use FP32 input for a consistent reference")
    weights, activations, operations = [], [], []
    weight_widths, precision = {}, {}
    hooks = []

    def remember(tensor, width):
        if isinstance(tensor, torch.Tensor):
            precision[id(tensor)] = (weakref.ref(tensor), width)

    def lookup(tensor):
        found = precision.get(id(tensor))
        return found[1] if found is not None and found[0]() is tensor else None

    def after(name, module, inputs, output):
        if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            return
        input_tensor = inputs[0]
        known_input = lookup(input_tensor)
        activation_width = known_input if known_input is not None else 32
        quantized = hasattr(module, "bit") and bool(getattr(module, "quant", False))
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            continuous, discrete = legacy_bits(module) if quantized else (32.0, 32)
            row = {
                "name": name, "type": type(module).__name__, "role": component_role(name),
                "quantized": quantized, "weight_elements": module.weight.numel(),
                "continuous_bits": continuous, "discrete_bits": discrete,
                "input_shape": list(input_tensor.shape), "output_shape": list(output.shape),
            }
            weights.append(row)
            weight_widths[id(module.weight)] = discrete
            if isinstance(module, nn.ConvTranspose2d):
                macs = input_tensor.numel() * (module.out_channels // module.groups) * math.prod(module.kernel_size)
            elif isinstance(module, nn.Conv2d):
                macs = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            else:
                macs = output.numel() * module.in_features
            operations.append({
                "name": name, "macs": macs, "weight_bits": discrete,
                "input_activation_bits": activation_width,
                "input_precision_source": "tracked" if known_input is not None else "fp32_fallback",
                "bops_conservative": macs * discrete * activation_width,
            })
            remember(output, 32)
        elif hasattr(module, "bit"):
            if quantized:
                continuous, discrete = legacy_bits(module)
                activations.append({"name": name, "type": type(module).__name__,
                                    "role": component_role(name), "activation_elements": output.numel(),
                                    "continuous_bits": continuous, "discrete_bits": discrete,
                                    "output_shape": list(output.shape)})
                remember(output, discrete)
            else:
                # Legacy quantizer bypass is identity, not a new FP32 operation.
                remember(output, activation_width)
        elif isinstance(module, (nn.MaxPool2d, nn.Identity)):
            remember(output, activation_width)
        else:
            remember(output, 32)

    remember(image, 32)
    original_training = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for name, module in model.named_modules():
            if not list(module.children()):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=name: after(n, m, i, o)))
        with torch.no_grad():
            prediction = model(image)
        if not torch.isfinite(prediction).all().item():
            raise RuntimeError("Nonfinite model output")
    finally:
        for hook in hooks:
            hook.remove()
        for module, training in original_training.items():
            module.training = training

    quantized_weights = [r for r in weights if r["quantized"]]
    parameter_elements = sum(p.numel() for p in model.parameters())
    parameter_fp32_bits = parameter_elements * 32
    parameter_packed_bits = sum(p.numel() * weight_widths.get(id(p), 32) for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    macs = sum(r["macs"] for r in operations)
    fp32_bops = macs * 32 * 32
    bops = sum(r["bops_conservative"] for r in operations)
    grouped = {}
    for role in ["encoder", "decoder", "unassigned"]:
        grouped[role] = {
            "weights": _summary([r for r in quantized_weights if r["role"] == role], "weight_elements"),
            "activations": _summary([r for r in activations if r["role"] == role], "activation_elements"),
        }
    return {
        "input_shape": list(image.shape), "output_shape": list(prediction.shape),
        "weights_quantized_only": _summary(quantized_weights, "weight_elements"),
        "weights_all_conv_linear": _summary(weights, "weight_elements"),
        "activations_quantized_only": _summary(activations, "activation_elements"),
        "groups": grouped,
        "parameters": {
            "elements_including_quantizer_parameters": parameter_elements,
            "fp32_reference_bytes": parameter_fp32_bits / 8,
            "packed_parameter_estimate_bytes": parameter_packed_bits / 8,
            "buffer_bytes_unchanged": buffer_bytes,
            "packed_parameters_and_buffers_estimate_bytes": parameter_packed_bits / 8 + buffer_bytes,
            "parameter_compression_estimate": parameter_fp32_bits / parameter_packed_bits,
        },
        "compute": {
            "conv_linear_macs": macs, "conv_linear_flops_2mac": 2 * macs,
            "fp32_reference_bops": fp32_bops, "conservative_quantized_bops": bops,
            "conservative_bops_reduction_fraction": 1 - bops / fp32_bops if fp32_bops else 0,
            "macs_with_explicit_lowbit_input": sum(r["macs"] for r in operations if r["input_activation_bits"] < 32),
        },
        "conventions": {
            "weight_average": "Quantized-only, weighted by weight element count; component means also reported.",
            "activation_average": "Quantized activation outputs weighted by actual runtime elements, at the reported input shape.",
            "packed_size": "Theoretical packed parameters only; biases, BN and quantizer parameters retain FP32. Actual checkpoint is not packed.",
            "macs": "Conv/ConvTranspose/Linear only; excludes BN, interpolation, activation and elementwise arithmetic. ConvTranspose counts input elements times output channels/group times kernel area, including cropped boundary contributions.",
            "flops": "Two FLOPs per MAC. Quantization preserves this structural count; BOP reduction is a separate estimate.",
            "bops": "MAC * weight bits * tracked input bits. Nonquantized/untraced operations default to FP32; no graph fusion or integer backend is assumed. Not the paper's unspecified FLOPs reduction metric.",
        },
        "weight_layers": weights, "activation_layers": activations, "operation_layers": operations,
    }
