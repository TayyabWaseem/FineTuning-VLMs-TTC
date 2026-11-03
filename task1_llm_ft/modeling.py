"""Model construction and checkpoint loading for Task 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

try:
    from .op_lora import ProjectedLoRALinear, add_lora
except ImportError:
    from op_lora import ProjectedLoRALinear, add_lora


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    checkpoint: str | Path | None = None,
    method: str = "full_ft",
    lora_rank: int = 0,
    lora_alpha: int = 0,
    lora_dropout: float = 0.0,
    protection_rank: int = 0,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if lora_rank:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        )
        add_lora(model, lora_rank, lora_alpha, lora_dropout, protection_rank)
        if checkpoint:
            state_path = Path(checkpoint) / "adapter_model.safetensors"
            if not state_path.exists():
                state_path = Path(checkpoint) / "model.safetensors"
            state = load_file(str(state_path), device="cpu")
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected adapter checkpoint keys: {incompatible.unexpected_keys[:5]}"
                )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint or model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    if device is not None:
        model.to(device)
    model.eval()
    return model


@torch.no_grad()
def merge_projected_adapters(model: torch.nn.Module) -> torch.nn.Module:
    """Merge custom adapter deltas into frozen Linear weights for evaluation."""
    for module in model.modules():
        if isinstance(module, ProjectedLoRALinear):
            module.base.weight.add_(module.delta_weight().to(module.base.weight))
            module.base.requires_grad_(False)
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, ProjectedLoRALinear):
                setattr(parent, child_name, child.base)
    return model


@torch.no_grad()
def compare_merged_unmerged_logits(
    adapter_model: torch.nn.Module,
    merged_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    unmerged = adapter_model(**batch).logits.float()
    merged = merged_model(**batch).logits.float()
    difference = (unmerged - merged).abs()
    return {
        "max_absolute_difference": difference.max().item(),
        "mean_absolute_difference": difference.mean().item(),
        "max_relative_difference": (difference / merged.abs().clamp_min(1e-8)).max().item(),
    }


def load_from_config(
    config: dict[str, Any],
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, Any]:
    tokenizer = load_tokenizer(config["model_name"])
    model = load_model(
        config["model_name"],
        checkpoint=checkpoint,
        method=config["method"],
        lora_rank=int(config.get("lora_rank", 0)),
        lora_alpha=int(config.get("lora_alpha", 0)),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        protection_rank=int(config.get("op_lora_protection_rank", 0)),
        device=device,
    )
    return model, tokenizer
