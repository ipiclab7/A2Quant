"""Display saved per-epoch Q evaluation records without loading a model.

IoU/nIoU are percentages. W/A are discrete, equally weighted module means.
FLOPs_proxy is the documented ACM512 activation-bitwidth linear estimate in G.
Complete records remain in the input JSONL; this only changes presentation.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path


DEFAULT_PROFILE = None

def load_cost_profile(path=DEFAULT_PROFILE):
    if path is None:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from configuration import FLOPS_PROFILE
        profile = dict(FLOPS_PROFILE)
    else:
        profile = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (profile["input_shape"] != [1, 3, 512, 512]
            or profile["reference_bits"] != 32
            or profile["bitwidth_statistic"] != "discrete_module_mean"):
        raise ValueError("Expected ACM512 discrete-module-mean display profile")
    counters = [profile[key] for key in ("conv_counter", "bn_counter", "pool_counter")]
    if not all(math.isfinite(value) and value >= 0 for value in counters):
        raise ValueError("Invalid display cost counter")
    if sum(counters) != profile["fp_reference_counter"]:
        raise ValueError("Display cost components do not match the FP reference")
    return profile


def linear_flops_g(activation, profile):
    if not math.isfinite(activation) or not 0 < activation <= 32:
        raise ValueError("Invalid activation bitwidth")
    return (profile["conv_counter"] * activation / profile["reference_bits"]
            + profile["bn_counter"] + profile["pool_counter"]) / 1e9


def format_epoch(record, cost_profile=None):
    profile = load_cost_profile() if cost_profile is None else cost_profile
    validation, bits = record["validation"], record["bits"]
    values = (validation["IoU"], validation["nIoU"],
              bits["weight"]["module_mean"], bits["activation"]["module_mean"])
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite evaluation metric")
    iou, niou, weight, activation = values
    # The retained numeric history exposes the module counts. Reject a different
    # topology instead of silently reusing ACM's recorded cost constants.
    health = record.get("range_health")
    if health is not None and (len(health["weight_rows"]) != profile["weight_quantizer_modules"]
                               or len(health["activation_rows"]) != profile["activation_quantizer_modules"]):
        raise ValueError("Quantizer counts disagree with the ACM display profile")
    flops_g = linear_flops_g(activation, profile)
    return (f"[{record['epoch']}] IoU={iou * 100:.4f}% "
            f"nIoU={niou * 100:.4f}% W/A={weight:.4f}/{activation:.4f} "
            f"FLOPs_proxy={flops_g:.6f}G")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epochs", type=Path, help="Saved epochs.jsonl")
    parser.add_argument("--follow", action="store_true", help="Display each newly completed record")
    parser.add_argument("--cost-profile", type=Path, default=DEFAULT_PROFILE,
                        help="Optional external profile; defaults to Python configuration constants")
    args = parser.parse_args()
    profile = load_cost_profile(args.cost_profile)
    with args.epochs.open(encoding="utf-8-sig") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if line and (line.endswith("\n") or not args.follow):
                print(format_epoch(json.loads(line), profile), flush=True)
            elif args.follow:
                stream.seek(position)
                time.sleep(0.5)
            else:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
