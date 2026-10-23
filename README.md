# **$A^2\text{QUANT}$: Asymmetric and Adaptive Quantization for Efficient Infrared Small Target Detection**
This is the official repository of the paper **$A^2\text{QUANT}$: Asymmetric and Adaptive Quantization for Efficient Infrared Small Target Detection**.

## Code Structure

### `quantization_core`: Core Quantization Code from the Paper

To use the quantization core, add this directory to your Python path or copy `quantization_core` into your own project.

```python
from quantization_core import convert_modules, smart_loss, set_stage

quant_model = convert_modules(
    your_model,
    role_resolver=your_role_resolver
)
```

### `ACM_quant_training`: Complete Quantization Training Code Using ACM as an Example

This directory provides a complete example of integrating $A^2\text{QUANT}$ into the ACM model.

It includes:

- Quantization integration
- Data preprocessing
- Model evaluation
- Optimizer configuration
- Two-stage quantization training
- Checkpoint resumption
- Training configuration
- Required input handling

## Environment, Data, and Getting Started

The recorded environment is:

- Linux
- Python 3.8.10
- PyTorch 1.13.0
- torchvision 0.14.0
- Single GPU

The required dependencies can be listed using:

```bash
python scripts/train.py --print-requirements
```

and installed with:

```bash
python scripts/train.py --print-requirements > /tmp/a2quant-requirements.txt
python -m pip install -r /tmp/a2quant-requirements.txt
```

## Training

The following commands provide examples for starting and resuming a quantization experiment.

### Start a New Experiment

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --train \
    --data-root /path/to/SIRST \
    --parent-checkpoint /path/to/FP104.pt \
    --activation-ranges /path/to/activation_ranges.json \
    --output-dir /path/outside/code/new-Q-run
```

### Resume an Existing Experiment

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --train \
    --data-root /path/to/SIRST \
    --parent-checkpoint /path/to/FP104.pt \
    --activation-ranges /path/to/activation_ranges.json \
    --output-dir /path/outside/code/new-Q-run \
    --resume
```

When resuming an unfinished run, the source code, configuration, and sample lists should remain identical to those used in the original experiment.

## FLOPs Proxy for Quantized Models

By default, THOP estimates computation based on the network topology and reports the parameter count, without considering the bit width used by each quantized layer.

Therefore, for models with the same topology, reducing weight or activation bit widths typically does not change the FLOPs reported by THOP, even though the actual low-bit computation cost may be significantly reduced.

Since this project simulates quantization using floating-point tensors, we additionally report a theoretical linearly scaled FLOPs proxy:

\[
\mathrm{FLOPs}_{\mathrm{proxy}}(\mathrm{G})
=
C_{\mathrm{conv}}\frac{\bar b_A}{32}
+
C_{\mathrm{BN}}
+
C_{\mathrm{pool}},
\]

where:

- \(C_{\mathrm{conv}}\) denotes the convolutional computation cost,
- \(C_{\mathrm{BN}}\) denotes the BatchNorm computation cost,
- \(C_{\mathrm{pool}}\) denotes the pooling computation cost,
- \(\bar b_A\) is the unweighted mean of the discrete activation bit widths used by all activation quantizer modules during the forward pass at the current epoch,
- \(\mathrm{G}\) denotes \(10^9\).

This proxy is intended to approximately reflect the theoretical computational benefit introduced by activation bit-width reduction.
