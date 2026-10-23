"""Load the frozen paper-ratio split and reject inconsistent dataset lists."""

import hashlib
import json
from pathlib import Path


def load_verified_splits(data_root, config_path):
    root = Path(data_root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if config["paper_ratios"] != {"train": 0.5, "val": 0.3, "test": 0.2}:
        raise ValueError("Expected the paper's 50/30/20 proportions")
    if config["split_group"] != provenance["primary_split"] or config["seed"] != provenance["paper_ratio_seed"]:
        raise ValueError("Split configuration differs from dataset provenance")
    if config["counts"] != provenance["paper_ratio_split_counts"]:
        raise ValueError("Split counts differ from dataset provenance")
    splits = {}
    for role in ("train", "val", "test"):
        path = root / "splits" / config["split_group"] / (role + ".txt")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != provenance["split_file_sha256"][role]:
            raise ValueError("Split file changed: " + role)
        ids = raw.decode("utf-8").splitlines()
        if len(ids) != config["counts"][role] or len(ids) != len(set(ids)):
            raise ValueError("Invalid sample count or duplicate IDs: " + role)
        if any(not name or Path(name).name != name or "\\" in name or name in {".", ".."} for name in ids):
            raise ValueError("Unsafe sample ID")
        splits[role] = ids
    all_ids = [name for ids in splits.values() for name in ids]
    manifest_ids = {row["id"] for row in provenance["samples_manifest"]}
    if len(all_ids) != config["total_samples"] or len(all_ids) != len(set(all_ids)) or set(all_ids) != manifest_ids:
        raise ValueError("Split overlap or incomplete dataset coverage")
    for name in all_ids:
        if not (root / "images" / (name + ".png")).is_file() or not (root / "masks" / (name + ".png")).is_file():
            raise FileNotFoundError("Incomplete image/mask pair: " + name)
    return config, splits
