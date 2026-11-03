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
    expected_weights = {}
    adapter_delta_max = 0.0
    adapter_delta_norm = 0.0
    for name, module in model.named_modules():
        if hasattr(module, "delta_weight") and hasattr(module, "base"):
            delta = module.delta_weight().detach().float()
            expected_weights[name] = module.base.weight.detach().float() + delta
            adapter_delta_max = max(adapter_delta_max, delta.abs().max().item())
            adapter_delta_norm += delta.square().sum().item()
    merged_model = merge_projected_adapters(merged_model)
    verification = compare_merged_unmerged_logits(model, merged_model, probe)
    weight_error_max = 0.0
    weight_error_sum = 0.0
    weight_error_count = 0
    for name, expected in expected_weights.items():
        actual = merged_model.get_submodule(name).weight.detach().float()
        weight_error = (actual - expected).abs()
        weight_error_max = max(weight_error_max, weight_error.max().item())
        weight_error_sum += weight_error.sum().item()
        weight_error_count += weight_error.numel()
    if weight_error_count:
        verification.update(
            {
                "adapter_delta_max_absolute": adapter_delta_max,
                "adapter_delta_frobenius_norm": adapter_delta_norm**0.5,
                "merged_weight_max_absolute_error": weight_error_max,
                "merged_weight_mean_absolute_error": weight_error_sum / weight_error_count,
            }
        )
    model = merged_model
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / "merge_verification.json", "w", encoding="utf-8") as handle:
        import json

        json.dump(verification, handle, indent=2)


if __name__ == "__main__":
    main()
