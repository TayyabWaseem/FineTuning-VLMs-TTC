"""Export an adapter checkpoint as a standard merged Transformers checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .modeling import compare_merged_unmerged_logits, load_from_config, merge_projected_adapters
    from .utils import load_config
except ImportError:
    from modeling import compare_merged_unmerged_logits, load_from_config, merge_projected_adapters
    from utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, tokenizer = load_from_config(config, args.checkpoint, "cuda")
    merged_model, _ = load_from_config(config, args.checkpoint, "cuda")
    probe = tokenizer("Verification prompt.", return_tensors="pt").to("cuda")
    model.eval()
    merged_model = merge_projected_adapters(merged_model)
    verification = compare_merged_unmerged_logits(model, merged_model, probe)
    model = merged_model
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / "merge_verification.json", "w", encoding="utf-8") as handle:
        import json

        json.dump(verification, handle, indent=2)


if __name__ == "__main__":
    main()
