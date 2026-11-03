"""Fail-fast checks to run before a full Kaggle Task 1 experiment."""

from __future__ import annotations

import json
import argparse
import shutil
from pathlib import Path

import torch
import yaml

try:
    from .op_lora import TARGET_MODULES
except ImportError:
    from op_lora import TARGET_MODULES


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "task1_llm_ft" / "configs"
PROMPTS = ROOT / "first_plot_questions.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU accelerator.")
    free, total = torch.cuda.mem_get_info()
    disk = shutil.disk_usage(ROOT)
    free_disk_gib = disk.free / 2**30
    print(f"Available disk space: {free_disk_gib:.2f} GiB")
    if free_disk_gib < 10:
        raise RuntimeError(
            f"Less than 10 GiB free disk space: {free_disk_gib:.1f} GiB"
        )
    if free_disk_gib < 20:
        print(
            f"WARNING: only {free_disk_gib:.1f} GiB free disk space; "
            "the full run may fail while saving checkpoints."
        )
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    if not args.smoke and len(configs) != 5:
        raise RuntimeError(f"Expected five Task 1 configs, found {len(configs)}")
    if not PROMPTS.is_file():
        raise RuntimeError(f"Canonical prompt file not found: {PROMPTS}")
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        required = {
            "seed",
            "model_name",
            "dataset_name",
            "num_examples",
            "num_updates",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "save_example_counts",
        }
        missing = required.difference(config)
        if missing:
            raise RuntimeError(f"{path.name} is missing keys: {sorted(missing)}")
        if config["seed"] != 6304:
            raise RuntimeError(f"{path.name} does not use the required seed/example count")
        if not args.smoke and config["num_examples"] != 40000:
            raise RuntimeError(f"{path.name} does not use the required example count")
        if not args.smoke and config["save_example_counts"] != [9984, 20000, 29984, 40000]:
            raise RuntimeError(f"{path.name} has incorrect checkpoint boundaries")
    groups = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))
    if len(groups) != 8 or any(len(group.get("formats", {})) != 3 for group in groups):
        raise RuntimeError("first_plot_questions.yaml must contain 8 questions × 3 formats")
    print(json.dumps({
        "status": "pass",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_free_gib": round(free / 2**30, 2),
        "gpu_total_gib": round(total / 2**30, 2),
        "disk_free_gib": round(free_disk_gib, 2),
        "target_modules": TARGET_MODULES,
        "mode": "smoke" if args.smoke else "full",
        "configs": [path.name for path in configs],
        "qualitative_prompts": len(groups) * 3,
    }, indent=2))


if __name__ == "__main__":
    main()
