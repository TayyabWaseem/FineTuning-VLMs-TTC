"""Task 1 weight-space and activation diagnostics."""

from __future__ import annotations

import argparse
import copy
import re
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
    # Match the zero-shot stock task prompts used in evaluate.py.
    math = [f"Q: {gsm[i]['question']}\nA:" for i in gsm_indices]
    def hellaswag_query(item: dict[str, Any]) -> str:
        context = item["ctx_a"] + " " + item["ctx_b"].capitalize()
        text = (item["activity_label"] + ": " + context).strip()
        text = text.replace(" [title]", ". ")
        return re.sub(r"\[.*?\]", "", text)
    nonmath = [hellaswag_query(hs[i]) for i in hs_indices]
    return math, nonmath, [str(i) for i in gsm_indices], [str(i) for i in hs_indices]


@torch.no_grad()
def activation_diagnostics(
    model: torch.nn.Module,
    base: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    chat_prompts: bool = False,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        rendered = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) if chat_prompts else prompt
        )
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            name, _, _ = _weight_delta(model, base, layer)
            def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], layer_id: int = layer) -> None:
                captured[layer_id] = args[0][:, -1, :].detach().float()
            handles.append(base.get_submodule(name).register_forward_pre_hook(hook))
        try:
            base(**encoded, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            name, w0, delta = _weight_delta(model, base, layer)
            z = captured[layer]
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
    base.float().eval()
    layers = [0, 13, 27]
    math, nonmath, math_ids, nonmath_ids = _prompt_texts(args.seed)
    result = {
        "config": config,
        "checkpoint": args.checkpoint,
        "weight": weight_diagnostics(
            model, base, layers, 128
        ),
        "activation": {
            "math": activation_diagnostics(model, base, tokenizer, math, layers, chat_prompts=True),
            "non_math": activation_diagnostics(model, base, tokenizer, nonmath, layers),
        },
        "probe_ids": {"gsm8k": math_ids, "hellaswag": nonmath_ids},
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
