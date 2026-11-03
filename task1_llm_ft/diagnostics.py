"""Task 1 weight-space and activation diagnostics."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

try:
    from .modeling import load_from_config
    from .op_lora import ProjectedLoRALinear
    from .utils import load_config, seed_everything, write_json
except ImportError:
    from modeling import load_from_config
    from op_lora import ProjectedLoRALinear
    from utils import load_config, seed_everything, write_json


def _named_weight(model: torch.nn.Module, layer: int) -> tuple[str, torch.Tensor]:
    needle = f"layers.{layer}.mlp.down_proj"
    for name, module in model.named_modules():
        if name.endswith(needle):
            if isinstance(module, ProjectedLoRALinear):
                return name, module.base.weight
            return name, module.weight
    raise KeyError(f"Could not find {needle}")


def _weight_delta(model: torch.nn.Module, base: torch.nn.Module, layer: int) -> tuple[str, torch.Tensor, torch.Tensor]:
    name, weight = _named_weight(model, layer)
    _, base_weight = _named_weight(base, layer)
    if isinstance(model.get_submodule(name), ProjectedLoRALinear):
        delta = model.get_submodule(name).delta_weight()
    else:
        delta = weight.float() - base_weight.float()
    return name, base_weight.float(), delta.float()


@torch.no_grad()
def weight_diagnostics(
    model: torch.nn.Module,
    base: torch.nn.Module,
    layers: list[int],
    protection_rank: int = 128,
) -> list[dict[str, Any]]:
    rows = []
    for layer in layers:
        name, w0, delta = _weight_delta(model, base, layer)
        if delta.norm() == 0:
            rho = 0.0
        else:
            u, _, _ = torch.linalg.svd(w0, full_matrices=False)
            k = min(protection_rank, u.shape[1])
            projected = u[:, :k] @ (u[:, :k].T @ delta)
            rho = (projected.norm() ** 2 / delta.norm() ** 2).item()
        rows.append(
            {
                "layer": layer,
                "parameter": name,
                "rho_k": rho,
                "eta": (delta.norm() / w0.norm()).item(),
                "delta_frobenius": delta.norm().item(),
                "weight_frobenius": w0.norm().item(),
            }
        )
    return rows


def _prompt_texts(
    seed: int, n: int = 32
) -> tuple[list[str], list[str], list[str], list[str]]:
    seed_everything(seed)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    generator = torch.Generator().manual_seed(seed)
    gsm_indices = torch.randperm(len(gsm), generator=generator)[:n].tolist()
    hs_indices = torch.randperm(len(hs), generator=generator)[:n].tolist()
    math = [gsm[i]["question"] for i in gsm_indices]
    nonmath = [hs[i]["ctx"] for i in hs_indices]
    return math, nonmath, [str(i) for i in gsm_indices], [str(i) for i in hs_indices]


@torch.no_grad()
def activation_diagnostics(
    model: torch.nn.Module,
    base: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        for layer in layers:
            name, w0, delta = _weight_delta(model, base, layer)
            module = base.get_submodule(name)
            captured: list[torch.Tensor] = []

            def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                captured.append(args[0][:, -1, :].detach().float())

            handle = module.register_forward_pre_hook(hook)
            base(**encoded, use_cache=False)
            handle.remove()
            z = captured[-1]
            numerator = (delta.to(z.device) @ z.T).norm().item()
            denominator = (w0.to(z.device) @ z.T).norm().item()
            rows.append(
                {
                    "layer": layer,
                    "parameter": name,
                    "prompt": prompt,
                    "correction_ratio": numerator / (denominator + 1e-8),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=6304)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_from_config(config, args.checkpoint, device)
    base_config = copy.deepcopy(config)
    base_config["lora_rank"] = 0
    base_config["lora_alpha"] = 0
    base_config["op_lora_protection_rank"] = 0
    base, _ = load_from_config(base_config, args.base_model or config["model_name"], device)
    layers = [0, 13, 27]
    math, nonmath, math_ids, nonmath_ids = _prompt_texts(args.seed)
    result = {
        "config": config,
        "checkpoint": args.checkpoint,
        "weight": weight_diagnostics(
            model, base, layers, int(config.get("op_lora_protection_rank", 128))
        ),
        "activation": {
            "math": activation_diagnostics(model, base, tokenizer, math, layers),
            "non_math": activation_diagnostics(model, base, tokenizer, nonmath, layers),
        },
        "probe_ids": {"gsm8k": math_ids, "hellaswag": nonmath_ids},
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
