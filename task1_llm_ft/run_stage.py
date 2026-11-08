"""Run exactly one full Task 1 method, resuming from private remote storage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .remote import require_remote, restore_artifacts, restore_evaluation_checkpoint, restore_latest
    from .utils import load_config
except ImportError:
    from remote import require_remote, restore_artifacts, restore_evaluation_checkpoint, restore_latest
    from utils import load_config


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("full_ft", "full_ft_l2sp", "lora4", "lora16", "op_lora16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    args = parser.parse_args()
    require_remote()
    config_path = ROOT / "task1_llm_ft" / "configs" / f"{args.method}.yaml"
    config = load_config(str(config_path))
    output = ROOT / config["output_dir"]
    prior_methods = METHODS[:METHODS.index(args.method)]
    if prior_methods:
        restore_artifacts(ROOT, [
            f"results/task1/{name}/{filename}"
            for name in prior_methods
            for filename in ("metadata.json", "remote_checkpoints.json")
        ])
    for name in prior_methods:
        previous = ROOT / "results/task1" / name
        index_path = previous / "remote_checkpoints.json"
        if not (previous / "metadata.json").exists() or not index_path.exists():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for saved in previous.glob("checkpoint-*"):
            if saved.is_dir() and index.get(saved.name, {}).get("verified"):
                if not saved.resolve().is_relative_to((ROOT / "results/task1").resolve()):
                    raise RuntimeError(f"Refusing to remove checkpoint outside Task 1 results: {saved}")
                shutil.rmtree(saved)
    if args.method != "full_ft":
        restore_artifacts(ROOT, ["results/task1/full_ft/selection_manifest.json"])
        if not (ROOT / "results/task1/full_ft/selection_manifest.json").exists():
            raise RuntimeError("Full FT selection manifest must exist before other methods")
    final = output / "checkpoint-40000"
    metadata_path = output / "metadata.json"
    restore_artifacts(ROOT, [
        f"results/task1/{args.method}/metadata.json",
        f"results/task1/{args.method}/remote_checkpoints.json",
    ])
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("skipped_updates", 0):
            raise RuntimeError(f"{args.method}: completed run had skipped optimizer updates")
        moved = (metadata.get("final_parameter_displacement_l2", 0) if args.method.startswith("full_ft")
                 else metadata.get("adapter_delta_frobenius_norm", 0))
        if moved > 0:
            restore_evaluation_checkpoint(output, 40000)
            print(f"{args.method}: complete; keeping existing artifacts")
            return
        raise RuntimeError(f"{args.method}: completed run has zero parameter displacement")
    checkpoint = restore_latest(output)
    command = [sys.executable, "-m", "task1_llm_ft.train", "--config", str(config_path)]
    if checkpoint:
        command += ["--resume", str(checkpoint)]
        print(f"{args.method}: resuming from {checkpoint}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.method in ("lora4", "lora16", "op_lora16") and metadata["adapter_delta_frobenius_norm"] <= 0:
        raise RuntimeError(f"{args.method}: trained adapter has zero effective update")
    if metadata["skipped_updates"]:
        raise RuntimeError(f"{args.method}: {metadata['skipped_updates']} optimizer updates were skipped")
    print(f"{args.method}: complete; checkpoint and log uploaded to private remote")


if __name__ == "__main__":
    main()
