"""Task 1 weight and activation diagnostics using a shared FP32 probe cache."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

try:
    from .modeling import load_from_config, load_tokenizer
    from .op_lora import ProjectedLoRALinear
    from .utils import load_config, seed_everything, write_json
except ImportError:
    from modeling import load_from_config, load_tokenizer
    from op_lora import ProjectedLoRALinear
    from utils import load_config, seed_everything, write_json


LAYERS = (0, 13, 27)


def _module(model: torch.nn.Module, layer: int) -> tuple[str, torch.nn.Module]:
    suffix = f"layers.{layer}.mlp.down_proj"
    for name, module in model.named_modules():
        if name.endswith(suffix):
            return name, module
    raise KeyError(f"Missing {suffix}")


def _weight_and_delta(
    model: torch.nn.Module, base: torch.nn.Module, layer: int
) -> tuple[str, torch.Tensor, torch.Tensor]:
    name, module = _module(model, layer)
    _, base_module = _module(base, layer)
    w0 = base_module.weight.detach().float()
    if isinstance(module, ProjectedLoRALinear):
        delta = module.delta_weight().detach().float()
    else:
        delta = module.weight.detach().float() - w0
    return name, w0, delta


def _prompt_texts(seed: int, n: int = 32) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hellaswag = load_dataset("Rowan/hellaswag", split="validation")
    generator = torch.Generator().manual_seed(seed)
    math_ids = torch.randperm(len(gsm), generator=generator)[:n].tolist()
    nonmath_ids = torch.randperm(len(hellaswag), generator=generator)[:n].tolist()

    def hellaswag_query(item: dict[str, Any]) -> str:
        context = item["ctx_a"] + " " + item["ctx_b"].capitalize()
        text = (item["activity_label"] + ": " + context).strip()
        text = text.replace(" [title]", ". ")
        return re.sub(r"\[.*?\]", "", text)

    prompts = {
        "math": [f"Q: {gsm[index]['question']}\nA:" for index in math_ids],
        "non_math": [hellaswag_query(hellaswag[index]) for index in nonmath_ids],
    }
    ids = {"gsm8k": [str(index) for index in math_ids],
           "hellaswag": [str(index) for index in nonmath_ids]}
    return prompts, ids


def _weight_hash(weight: torch.Tensor) -> str:
    return hashlib.sha256(weight.detach().float().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def prepare_probe_cache(
    base: torch.nn.Module,
    tokenizer: Any,
    cache_path: Path,
    seed: int,
) -> dict[str, Any]:
    prompts, ids = _prompt_texts(seed)
    hashes = {layer: _weight_hash(_module(base, layer)[1].weight) for layer in LAYERS}
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (cache.get("seed") != seed or cache.get("probe_ids") != ids
                or cache.get("starting_weight_sha256") != hashes
                or tuple(cache.get("layers", ())) != LAYERS):
            raise RuntimeError("Shared probe cache does not match starting model or sampled IDs")
        return cache

    device = next(base.parameters()).device
    basis = {}
    for layer in LAYERS:
        weight = _module(base, layer)[1].weight.detach().float()
        u, _, _ = torch.linalg.svd(weight, full_matrices=False)
        basis[layer] = u[:, :128].contiguous().cpu()

    activations: dict[str, dict[int, list[torch.Tensor]]] = {
        kind: {layer: [] for layer in LAYERS} for kind in prompts
    }
    for kind, texts in prompts.items():
        for prompt in texts:
            rendered = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                ) if kind == "math" else prompt
            )
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
            captured: dict[int, torch.Tensor] = {}
            handles = []
            for layer in LAYERS:
                module = _module(base, layer)[1]
                def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], layer_id: int = layer) -> None:
                    captured[layer_id] = args[0][:, -1, :].detach().float().cpu().squeeze(0)
                handles.append(module.register_forward_pre_hook(hook))
            try:
                base(**encoded, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            for layer in LAYERS:
                activations[kind][layer].append(captured[layer])

    cache = {
        "seed": seed,
        "layers": LAYERS,
        "probe_ids": ids,
        "prompts": prompts,
        "starting_weight_sha256": hashes,
        "u_128": basis,
        "z": {
            kind: {layer: torch.stack(values) for layer, values in by_layer.items()}
            for kind, by_layer in activations.items()
        },
        "precision": "FP32 starting model, FP32 SVD, FP32 cached activations",
        "prompt_formats": {"math": "gsm8k_cot non-thinking chat", "non_math": "hellaswag stock completion"},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


@torch.no_grad()
def weight_diagnostics(
    model: torch.nn.Module, base: torch.nn.Module, cache: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for layer in LAYERS:
        name, w0, delta = _weight_and_delta(model, base, layer)
        u = cache["u_128"][layer].to(delta.device)
        delta_norm = torch.linalg.vector_norm(delta)
        projected = u @ (u.T @ delta)
        rows.append({
            "layer": layer,
            "parameter": name,
            "rho_k": ((projected.square().sum() / delta.square().sum()).item()
                      if delta_norm > 0 else 0.0),
            "eta": (delta_norm / torch.linalg.vector_norm(w0)).item(),
            "delta_frobenius": delta_norm.item(),
            "weight_frobenius": torch.linalg.vector_norm(w0).item(),
        })
    return rows


@torch.no_grad()
def activation_diagnostics(
    model: torch.nn.Module, base: torch.nn.Module, cache: dict[str, Any], kind: str
) -> list[dict[str, Any]]:
    rows = []
    ids = cache["probe_ids"]["gsm8k" if kind == "math" else "hellaswag"]
    for layer in LAYERS:
        name, w0, delta = _weight_and_delta(model, base, layer)
        z = cache["z"][kind][layer].to(w0.device)
        numerator = torch.linalg.vector_norm(z @ delta.T, dim=1)
        denominator = torch.linalg.vector_norm(z @ w0.T, dim=1) + 1e-8
        ratios = (numerator / denominator).cpu().tolist()
        for index, ratio in enumerate(ratios):
            rows.append({
                "layer": layer,
                "parameter": name,
                "probe_id": ids[index],
                "prompt": cache["prompts"][kind][index],
                "correction_ratio": ratio,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--base-model")
    parser.add_argument("--cache", default="results/task1/probe_cache.pt")
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and (not args.checkpoint or not args.output):
        parser.error("--checkpoint and --output are required unless --prepare-only is set")
    config = load_config(args.config)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(config["model_name"])
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model or config["model_name"], torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device).eval()
    cache_path = Path(args.cache)
    cache = prepare_probe_cache(base, tokenizer, cache_path, args.seed)
    write_json(cache_path.with_suffix(".ids.json"), {
        "seed": args.seed, "probe_ids": cache["probe_ids"],
        "prompt_formats": cache["prompt_formats"],
        "starting_weight_sha256": cache["starting_weight_sha256"],
    })
    if args.prepare_only:
        print(f"Prepared {cache_path} for 32 GSM8K and 32 HellaSwag prompts")
        return
    if int(config.get("lora_rank", 0)):
        model, _ = load_from_config(config, args.checkpoint, device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=torch.float32, trust_remote_code=True,
        ).to(device).eval()
    result = {
        "config": config,
        "checkpoint": args.checkpoint,
        "probe_ids": cache["probe_ids"],
        "starting_weight_sha256": cache["starting_weight_sha256"],
        "weight": weight_diagnostics(model, base, cache),
        "activation": {
            "math": activation_diagnostics(model, base, cache, "math"),
            "non_math": activation_diagnostics(model, base, cache, "non_math"),
        },
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
