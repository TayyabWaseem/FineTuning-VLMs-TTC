"""Verify that all long-running training stages produced complete artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/task1")
    parser.add_argument("--checkpoint-counts", default="9984,20000,29984,40000")
    args = parser.parse_args()
    root = Path(args.root)
    methods = ["full_ft", "full_ft_l2sp", "lora4", "lora16", "op_lora16"]
    required_checkpoints = [
        f"checkpoint-{value.strip()}"
        for value in args.checkpoint_counts.split(",")
        if value.strip()
    ]
    missing: list[str] = []
    for method in methods:
        output = root / method
        index_path = output / "remote_checkpoints.json"
        remote_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        for name in ["train.jsonl", "metadata.json", "run_config.json"]:
            if not (output / name).is_file():
                missing.append(str(output / name))
        for checkpoint in required_checkpoints:
            path = output / checkpoint
            if remote_index.get(checkpoint, {}).get("verified"):
                continue
            for name in ["run_config.json", "training_state.pt"]:
                if not (path / name).is_file():
                    missing.append(str(path / name))
            if not ((path / "config.json").is_file() or (path / "adapter_config.json").is_file()):
                missing.append(str(path / "config.json or adapter_config.json"))
            if not any(path.glob("*.safetensors")):
                missing.append(str(path / "*.safetensors"))
    if missing:
        raise RuntimeError("Missing artifacts:\n" + "\n".join(missing))
    print(f"Verified artifacts for {len(methods)} methods under {root}")


if __name__ == "__main__":
    main()
