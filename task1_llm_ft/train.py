"""Train one Task 1 method from a YAML configuration."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.amp import GradScaler
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .data import CausalSFTDataset, SFTCollator, select_metamath_examples
    from .op_lora import add_lora, export_projected_adapters
    from .utils import (
        JsonlLogger, cosine_lr, load_config, model_parameter_count,
        parameter_groups, seed_everything, write_json,
    )
except ImportError:
    from data import CausalSFTDataset, SFTCollator, select_metamath_examples
    from op_lora import add_lora, export_projected_adapters
    from utils import (
        JsonlLogger, cosine_lr, load_config, model_parameter_count,
        parameter_groups, seed_everything, write_json,
    )


def l2_sp_penalty(model: torch.nn.Module, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    penalty = None
    for name, parameter in model.named_parameters():
        if name not in reference:
            continue
        reference_value = reference[name].to(device=parameter.device, dtype=torch.float32)
        term = (parameter.float() - reference_value).pow(2).sum()
        penalty = term if penalty is None else penalty + term
    return penalty if penalty is not None else torch.zeros((), device=next(model.parameters()).device)


def shift_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shifted_labels = labels[:, 1:]
    valid_tokens = int((shifted_labels != -100).sum().item())
    loss_sum = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, valid_tokens


def parameter_displacement(
    model: torch.nn.Module, reference: dict[str, torch.Tensor]
) -> float | None:
    total = torch.zeros((), dtype=torch.float64)
    found = False
    for name, parameter in model.named_parameters():
        if name in reference:
            found = True
            total += (parameter.detach().float().cpu() - reference[name]).pow(2).sum().double()
    return total.sqrt().item() if found else None


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    output_dir: Path,
    examples_seen: int,
    update: int,
    config: dict[str, Any],
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{examples_seen}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "module"):
        model = model.module
    if int(config["lora_rank"]) > 0:
        adapter_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        save_file(adapter_state, str(checkpoint_dir / "adapter_model.safetensors"))
    else:
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    if int(config["lora_rank"]) > 0:
        write_json(checkpoint_dir / "adapter_config.json", config)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "examples_seen": examples_seen,
            "update": update,
            "scheduler": {
                "type": "cosine",
                "total_updates": int(config["num_updates"]),
                "warmup_updates": int(config["warmup_updates"]),
                "current_update": update,
            },
            "data_position": {
                "examples_seen": examples_seen,
                "next_index": examples_seen % int(config["num_examples"]),
            },
        },
        checkpoint_dir / "training_state.pt",
    )
    write_json(checkpoint_dir / "run_config.json", config)
    write_json(
        checkpoint_dir / "checkpoint_metrics.json",
        {
            "examples_seen": examples_seen,
            "update": update,
            "checkpoint_size_bytes": directory_size(checkpoint_dir),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    is_adapter_run = int(config["lora_rank"]) > 0
    model_dtype = (
        torch.float16
        if is_adapter_run and device.type == "cuda"
        else torch.float32
    )
    model_source = args.resume if args.resume and not is_adapter_run else config["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    reference_model = model
    if args.resume and float(config["l2_sp_lambda"]) > 0:
        reference_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], torch_dtype=torch.float32, trust_remote_code=True
        )
    reference = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in reference_model.named_parameters()
    }
    if reference_model is not model:
        del reference_model
    adapter_stats = None
    if int(config["lora_rank"]) > 0:
        adapter_stats = add_lora(
            model,
            rank=int(config["lora_rank"]),
            alpha=int(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
            protection_rank=int(config["op_lora_protection_rank"]),
        )
        if args.resume:
            state_path = Path(args.resume) / "adapter_model.safetensors"
            if not state_path.exists():
                state_path = Path(args.resume) / "model.safetensors"
            incompatible = model.load_state_dict(load_file(str(state_path), device="cpu"), strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(f"Unexpected adapter keys: {incompatible.unexpected_keys[:5]}")
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    model.to(device)

    dataset = select_metamath_examples(
        config["dataset_name"], tokenizer, int(config["max_length"]), int(config["num_examples"])
    )
    train_dataset = CausalSFTDataset(dataset, tokenizer, int(config["max_length"]))
    # The assignment requires retained examples to remain in source order.
    sampler = SequentialSampler(train_dataset)
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["micro_batch_size"]),
        sampler=sampler,
        collate_fn=SFTCollator(tokenizer),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(model, float(config["weight_decay"])),
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    resume_state = None
    if args.resume:
        resume_state = torch.load(
            Path(args.resume) / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        torch.set_rng_state(resume_state["rng_cpu"])
        if torch.cuda.is_available() and resume_state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(resume_state["rng_cuda"])
    logger = JsonlLogger(output_dir / "train.jsonl")
    updates = int(config["num_updates"])
    accumulation = int(config["gradient_accumulation_steps"])
    save_counts = set(int(value) for value in config["save_example_counts"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    examples_seen = int(resume_state["examples_seen"]) if resume_state else 0
    update = int(resume_state["update"]) if resume_state else 0
    started = time.perf_counter()
    data_iterator = iter(loader)
    for _ in range(examples_seen // int(config["micro_batch_size"])):
        next(data_iterator)
    accumulated_tokens = 0
    accumulated_sft_sum = 0.0
    accumulated_penalty = torch.zeros((), device=device)
    while update < updates:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(**batch)
            sft_sum, valid_tokens = shift_loss(outputs.logits, batch["labels"])
        scaler.scale(sft_sum).backward()
        accumulated_tokens += valid_tokens
        accumulated_sft_sum += sft_sum.detach().float().item()
        is_update_boundary = (examples_seen + int(config["micro_batch_size"])) % (
            int(config["micro_batch_size"]) * accumulation
        ) == 0
        examples_seen += int(config["micro_batch_size"])
        if not is_update_boundary:
            continue
        penalty = (
            l2_sp_penalty(model, reference)
            if config["l2_sp_lambda"]
            else torch.zeros((), device=device)
        )
        if config["l2_sp_lambda"]:
            scaler.scale(float(config["l2_sp_lambda"]) * penalty).backward()
        accumulated_penalty = penalty.detach()
        scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(max(1, accumulated_tokens))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
        current_lr = cosine_lr(update, updates, int(config["warmup_updates"]), float(config["learning_rate"]))
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        optimizer.zero_grad(set_to_none=True)
        update += 1
        logger.write(
            {
                "update": update,
                "examples_seen": examples_seen,
                "sft_loss": accumulated_sft_sum / max(1, accumulated_tokens),
                "l2_sp_omega": accumulated_penalty.float().item(),
                "l2_sp_term": float(config["l2_sp_lambda"]) * accumulated_penalty.float().item(),
                "total_loss": accumulated_sft_sum / max(1, accumulated_tokens)
                + float(config["l2_sp_lambda"]) * accumulated_penalty.float().item(),
                "learning_rate": current_lr,
                "grad_norm": float(grad_norm),
                "skipped_update": skipped,
            }
        )
        if examples_seen in save_counts:
            save_checkpoint(
                model, tokenizer, optimizer, scaler, output_dir, examples_seen, update, config
            )
        accumulated_tokens = 0
        accumulated_sft_sum = 0.0
        accumulated_penalty = torch.zeros((), device=device)

    elapsed = time.perf_counter() - started
    metadata = {
        "method": config["method"],
        "device": str(device),
        "training_seconds": elapsed,
        "trainable_parameters": model_parameter_count(model, trainable_only=True),
        "total_parameters": model_parameter_count(model),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "adapter_stats": adapter_stats.__dict__ if adapter_stats else None,
        "reference_storage_bytes": sum(value.numel() * value.element_size() for value in reference.values())
        if config["l2_sp_lambda"]
        else 0,
        "final_parameter_displacement_l2": parameter_displacement(model, reference)
        if not is_adapter_run
        else None,
        "checkpoint_sizes_bytes": {
            path.name: directory_size(path)
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir()
        },
    }
    if adapter_stats:
        torch.save(export_projected_adapters(model), output_dir / "projected_adapters.pt")
    write_json(output_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
