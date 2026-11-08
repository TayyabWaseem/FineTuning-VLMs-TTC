"""Train one Task 1 method from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.amp import GradScaler
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .data import CausalSFTDataset, SFTCollator, select_metamath_examples
    from .op_lora import ProjectedLoRALinear, add_lora, export_projected_adapters
    from .remote import sync_artifact, sync_checkpoint
    from .utils import (
        JsonlLogger, cosine_lr, load_config, model_parameter_count,
        parameter_groups, seed_everything, write_json,
    )
except ImportError:
    from data import CausalSFTDataset, SFTCollator, select_metamath_examples
    from op_lora import ProjectedLoRALinear, add_lora, export_projected_adapters
    from remote import sync_artifact, sync_checkpoint
    from utils import (
        JsonlLogger, cosine_lr, load_config, model_parameter_count,
        parameter_groups, seed_everything, write_json,
    )


@torch.no_grad()
def l2_sp_penalty(model: torch.nn.Module, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    penalty = None
    for name, parameter in model.named_parameters():
        if name not in reference:
            continue
        reference_value = reference[name].to(device=parameter.device, dtype=torch.float32)
        term = 0.5 * (parameter.float() - reference_value).pow(2).sum()
        penalty = term if penalty is None else penalty + term
    return penalty if penalty is not None else torch.zeros((), device=next(model.parameters()).device)


def add_l2_sp_gradients(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
    coefficient: float,
) -> torch.Tensor:
    penalty = torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        if name not in reference:
            continue
        reference_value = reference[name].to(device=parameter.device, dtype=torch.float32)
        displacement = parameter.detach().float() - reference_value
        penalty += 0.5 * displacement.square().sum()
        penalty_gradient = coefficient * displacement
        penalty_gradient = penalty_gradient.to(dtype=parameter.dtype)
        if parameter.grad is None:
            parameter.grad = penalty_gradient
        else:
            parameter.grad.add_(penalty_gradient)
    return penalty


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
    adapter_stats: dict[str, Any] | None = None,
    elapsed_seconds: float = 0.0,
    peak_gpu_memory_bytes: int | None = None,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{examples_seen}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "module"):
        model = model.module
    if int(config["lora_rank"]) > 0:
        adapter_state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
            if name.endswith(("lora_A", "lora_B"))
            or (name.endswith(("u_protected", "v_protected")) and value.numel() > 0)
        }
        save_file(adapter_state, str(checkpoint_dir / "adapter_model.safetensors"))
    else:
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    if int(config["lora_rank"]) > 0:
        write_json(checkpoint_dir / "adapter_config.json", config)
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rng_python = random.getstate()
    rng_numpy = np.random.get_state()
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_cpu": rng_cpu,
            "rng_cuda": rng_cuda,
            "rng_python": rng_python,
            "rng_numpy": rng_numpy,
            "examples_seen": examples_seen,
            "update": update,
            "adapter_stats": adapter_stats,
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "selection_sha256": json.loads(
                (output_dir / "selection_manifest.json").read_text(encoding="utf-8")
            )["selection_sha256"],
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
    sync_checkpoint(checkpoint_dir, output_dir)
    # Network clients may consume host RNGs. Match the saved recovery state.
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    random.setstate(rng_python)
    np.random.set_state(rng_numpy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume:
        prior_config = json.loads((Path(args.resume) / "run_config.json").read_text(encoding="utf-8"))
        if prior_config != config:
            raise RuntimeError("Resume configuration differs from the recovery checkpoint")
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
    needs_fp32_bases = bool(config["op_lora_protection_rank"]) and not args.resume
    model_dtype = (
        torch.float16
        if is_adapter_run and device.type == "cuda" and not needs_fp32_bases
        else torch.float32
    )
    model_source = args.resume if args.resume and not is_adapter_run else config["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    if config["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    reference_model = None
    if not is_adapter_run:
        reference_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], torch_dtype=torch.float32, trust_remote_code=True
        )
    reference = (
        {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in reference_model.named_parameters()
        }
        if reference_model is not None
        else {}
    )
    if reference_model is not None:
        del reference_model
    adapter_stats = None
    if int(config["lora_rank"]) > 0:
        model.to(device)
        adapter_stats = add_lora(
            model,
            rank=int(config["lora_rank"]),
            alpha=int(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
            protection_rank=int(config["op_lora_protection_rank"]),
            initialize_protection=not bool(args.resume),
        )
        if needs_fp32_bases and device.type == "cuda":
            # The SVD used original FP32 weights. Freeze the resulting backbone
            # in FP16 while preserving FP32 factors and protected bases.
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    parameter.data = parameter.data.to(dtype=torch.float16)
        if args.resume:
            state_path = Path(args.resume) / "adapter_model.safetensors"
            if not state_path.exists():
                state_path = Path(args.resume) / "model.safetensors"
            incompatible = model.load_state_dict(load_file(str(state_path), device="cpu"), strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(f"Unexpected adapter keys: {incompatible.unexpected_keys[:5]}")
            if config["op_lora_protection_rank"] and any(
                key.endswith(("u_protected", "v_protected"))
                for key in incompatible.missing_keys
            ):
                raise RuntimeError("OPLoRA recovery checkpoint is missing protected bases")
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    if config["gradient_checkpointing"] and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(device)
    adapter_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    ]
    dataset = select_metamath_examples(
        config["dataset_name"], tokenizer, int(config["max_length"]),
        int(config["num_examples"]), output_dir / "selection_manifest.json",
    )
    reference_manifest = output_dir.parent / "full_ft" / "selection_manifest.json"
    if config["method"] != "full_ft" and not str(config["method"]).startswith("smoke_"):
        if not reference_manifest.exists():
            raise RuntimeError("Full FT selection manifest is missing")
        current_hash = json.loads((output_dir / "selection_manifest.json").read_text(encoding="utf-8"))["selection_sha256"]
        reference_hash = json.loads(reference_manifest.read_text(encoding="utf-8"))["selection_sha256"]
        if current_hash != reference_hash:
            raise RuntimeError("MetaMathQA selection differs from the Full FT run")
    train_dataset = CausalSFTDataset(dataset, tokenizer, int(config["max_length"]))
    if str(config["method"]).startswith("smoke_") and train_dataset.items:
        supervised = [
            token
            for token, label in zip(
                train_dataset.items[0].input_ids, train_dataset.items[0].labels
            )
            if label != -100
        ]
        print(
            "Supervised tokens example:",
            tokenizer.decode(supervised, skip_special_tokens=False),
            flush=True,
        )
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
    # FP16 backbone gradients can overflow at the default 65536 scale before
    # a two-step smoke run has enough updates to recover.
    scaler = GradScaler("cuda", init_scale=1024.0, enabled=device.type == "cuda")
    resume_state = None
    if args.resume:
        resume_state = torch.load(
            Path(args.resume) / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        current_selection = json.loads((output_dir / "selection_manifest.json").read_text(encoding="utf-8"))["selection_sha256"]
        if resume_state.get("selection_sha256") != current_selection:
            raise RuntimeError("Resume selection differs from the recovery checkpoint")
        # Restore RNG after constructing/skipping the DataLoader iterator below;
        # iterator construction itself consumes a Torch RNG draw.
        log_path = output_dir / "train.jsonl"
        if log_path.exists():
            prior = [
                line for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and int(json.loads(line)["update"]) <= int(resume_state["update"])
            ]
            log_path.write_text("\n".join(prior) + ("\n" if prior else ""), encoding="utf-8")
    logger = JsonlLogger(output_dir / "train.jsonl")
    updates = int(config["num_updates"])
    accumulation = int(config["gradient_accumulation_steps"])
    save_counts = set(int(value) for value in config["save_example_counts"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    examples_seen = int(resume_state["examples_seen"]) if resume_state else 0
    update = int(resume_state["update"]) if resume_state else 0
    started = time.perf_counter()
    previous_elapsed = float(resume_state.get("elapsed_seconds", 0.0)) if resume_state else 0.0
    previous_peak = int(resume_state.get("peak_gpu_memory_bytes") or 0) if resume_state else 0
    effective_adapter_stats = (
        resume_state.get("adapter_stats") if resume_state and resume_state.get("adapter_stats")
        else adapter_stats.__dict__ if adapter_stats else None
    )
    skipped_updates = sum(
        bool(json.loads(line).get("skipped_update"))
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ) if resume_state else 0
    data_iterator = iter(loader)
    for _ in range(examples_seen // int(config["micro_batch_size"])):
        next(data_iterator)
    if resume_state:
        torch.set_rng_state(resume_state["rng_cpu"])
        if torch.cuda.is_available() and resume_state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(resume_state["rng_cuda"])
        if "rng_python" in resume_state:
            random.setstate(resume_state["rng_python"])
        if "rng_numpy" in resume_state:
            np.random.set_state(resume_state["rng_numpy"])
    while update < updates:
        microbatches = []
        total_tokens = 0
        for _ in range(accumulation):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                batch = next(data_iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            total_tokens += int((batch["labels"][:, 1:] != -100).sum().item())
            microbatches.append(batch)
            examples_seen += int(config["micro_batch_size"])

        accumulated_sft_sum = 0.0
        for batch in microbatches:
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                outputs = model(**batch)
                sft_sum, _ = shift_loss(outputs.logits, batch["labels"])
            scaler.scale(sft_sum / max(1, total_tokens)).backward()
            accumulated_sft_sum += sft_sum.detach().float().item()

        scaler.unscale_(optimizer)
        penalty = (
            add_l2_sp_gradients(model, reference, float(config["l2_sp_lambda"]))
            if config["l2_sp_lambda"] else torch.zeros((), device=device)
        )
        accumulated_penalty = penalty.detach()
        adapter_grad_max = max(
            (
                parameter.grad.detach().abs().max().item()
                for _, parameter in adapter_parameters
                if parameter.grad is not None
            ),
            default=0.0,
        )
        adapter_grad_count = sum(
            parameter.grad is not None for _, parameter in adapter_parameters
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
        current_lr = cosine_lr(update, updates, int(config["warmup_updates"]), float(config["learning_rate"]))
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        skipped_updates += int(skipped)
        optimizer.zero_grad(set_to_none=True)
        update += 1
        logger.write(
            {
                "update": update,
                "examples_seen": examples_seen,
                "sft_loss": accumulated_sft_sum / max(1, total_tokens),
                "l2_sp_omega": accumulated_penalty.float().item(),
                "l2_sp_term": float(config["l2_sp_lambda"]) * accumulated_penalty.float().item(),
                "total_loss": accumulated_sft_sum / max(1, total_tokens)
                + float(config["l2_sp_lambda"]) * accumulated_penalty.float().item(),
                "learning_rate": current_lr,
                "grad_norm": float(grad_norm),
                "skipped_update": skipped,
                "scaler_scale": old_scale,
                "adapter_grad_count": adapter_grad_count,
                "adapter_grad_max": adapter_grad_max,
            }
        )
        if skipped:
            raise RuntimeError(
                f"FP16 scaler skipped optimizer update {update} after {examples_seen} examples; "
                "stopping so the prescribed 1,250 successful updates are not silently reduced. "
                "Inspect train.jsonl and restart from the last recovery checkpoint after fixing the scale."
            )
        if examples_seen in save_counts:
            if is_adapter_run and not any(
                module.delta_weight().detach().abs().max().item() > 0
                for module in model.modules() if isinstance(module, ProjectedLoRALinear)
            ):
                raise RuntimeError(f"Adapter correction is still zero at checkpoint {examples_seen}")
            save_checkpoint(
                model, tokenizer, optimizer, scaler, output_dir, examples_seen, update, config,
                effective_adapter_stats,
                previous_elapsed + time.perf_counter() - started,
                max(previous_peak, torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
            )

    elapsed = previous_elapsed + time.perf_counter() - started
    metadata = {
        "method": config["method"],
        "device": str(device),
        "training_seconds": elapsed,
        "trainable_parameters": model_parameter_count(model, trainable_only=True),
        "total_parameters": model_parameter_count(model),
        "peak_gpu_memory_bytes": max(previous_peak, torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "skipped_updates": skipped_updates,
        "adapter_stats": effective_adapter_stats,
        "adapter_delta_frobenius_norm": (
            sum(
                module.delta_weight().detach().float().square().sum().item()
                for module in model.modules() if isinstance(module, ProjectedLoRALinear)
            ) ** 0.5 if adapter_parameters else 0.0
        ),
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
        "remote_checkpoint_sizes_bytes": {
            name: details["size_bytes"]
            for name, details in json.loads((output_dir / "remote_checkpoints.json").read_text(encoding="utf-8")).items()
        } if (output_dir / "remote_checkpoints.json").exists() else {},
    }
    if adapter_stats:
        torch.save(export_projected_adapters(model), output_dir / "projected_adapters.pt")
        sync_artifact(output_dir / "projected_adapters.pt", Path(__file__).resolve().parents[1])
    write_json(output_dir / "metadata.json", metadata)
    sync_artifact(output_dir / "selection_manifest.json", Path(__file__).resolve().parents[1])
    sync_artifact(output_dir / "metadata.json", Path(__file__).resolve().parents[1])
    if (output_dir / "remote_checkpoints.json").exists():
        # The private remote copy retains optimizer/scaler/RNG recovery state.
        # Local final weights remain available for evaluation and diagnostics.
        (output_dir / f"checkpoint-{examples_seen}" / "training_state.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
