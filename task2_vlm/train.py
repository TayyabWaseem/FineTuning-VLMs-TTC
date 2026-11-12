"""Three-stage Task 2 connector/LoRA training with private recovery checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from .development import evaluate_development
from .feature_store import FeatureStore
from .prompts import build_example
from .recovery import _client, load_weights, restore_latest, save_checkpoint
from .sampler import CurriculumSampler, STAGE_UPDATES
from .vlm import VisionLanguageModel


PEAKS = {1: (1e-3, None), 2: (1e-4, 2e-5), 3: (5e-5, 1e-5)}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def stage_lr(stage: int, update: int) -> float:
    total = STAGE_UPDATES[stage - 1]
    warmup = math.ceil(total * 0.05)
    if update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup + 1) / (total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def optimizer_for(model: VisionLanguageModel, stage: int):
    connector = list(model.connector.parameters())
    adapter = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": connector, "lr": PEAKS[stage][0], "kind": "connector"}]
    if stage > 1:
        if not adapter:
            raise RuntimeError("LoRA parameters are missing")
        groups.append({"params": adapter, "lr": PEAKS[stage][1], "kind": "lora"})
    return torch.optim.AdamW(groups, weight_decay=0.01, foreach=False)


@torch.no_grad()
def calibrate(model, tokenizer, features, captions) -> dict:
    chosen = captions[:128]
    if len(chosen) != 128:
        raise RuntimeError("Connector calibration requires 128 training captions")
    device = next(model.connector.parameters()).device
    visual_sum, text_sum = 0.0, 0.0
    model.eval()
    for start in range(0, len(chosen), 8):
        rows = chosen[start:start + 8]
        patches = torch.tensor(np.stack([features[row["image_id"]] for row in rows]), device=device, dtype=torch.float32)
        projected = model.connector(patches)
        visual_sum += projected.square().mean().sqrt().item() * len(rows)
        tokens = tokenizer([row["question"] for row in rows], padding=True, truncation=True,
                           max_length=80, return_tensors="pt").to(device)
        embeddings = model.backbone.get_input_embeddings()(tokens["input_ids"]).float()
        mask = tokens["attention_mask"].unsqueeze(-1)
        text_sum += (embeddings.square() * mask).sum().div(mask.sum() * embeddings.shape[-1]).sqrt().item() * len(rows)
    visual_rms, text_rms = visual_sum / len(chosen), text_sum / len(chosen)
    if visual_rms <= 0 or text_rms <= 0:
        raise RuntimeError("Connector calibration RMS must be positive")
    scale = text_rms / visual_rms
    model.connector.scale.fill_(scale)
    return {"rule": "training-only mean batch RMS of text embeddings / unscaled connector outputs",
            "images": 128, "visual_rms": visual_rms, "text_rms": text_rms, "scale": scale}


def append_json(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/task2"))
    parser.add_argument("--until", type=int, choices=(2250, 3750, 4500), default=4500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Task 2 training requires CUDA")
    root = args.output.resolve()
    project_root = Path(__file__).resolve().parents[1]
    if root != (project_root / "results/task2").resolve():
        raise ValueError("Recovery storage requires output results/task2")
    root.mkdir(parents=True, exist_ok=True)
    random.seed(6304); np.random.seed(6304); torch.manual_seed(6304); torch.cuda.manual_seed_all(6304)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    features = FeatureStore(args.features)
    api, repo, _ = _client()
    model_revision = api.model_info("Qwen/Qwen3-0.6B").sha
    identity = {"manifest_sha256": file_hash(args.manifest), "feature_index_sha256": file_hash(args.features / "feature_index.json"),
                "seed": 6304, "model": "Qwen/Qwen3-0.6B", "model_revision": model_revision,
                "source_sha256": source_hash(), "recipe": "task2_196_v1",
                "gpu": torch.cuda.get_device_name(0),
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "huggingface-hub")}}
    checkpoint = restore_latest(root, project_root)
    saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False) if checkpoint else None
    if saved and saved["metadata"]["identity"] != identity:
        raise RuntimeError("Task 2 dataset/cache identity differs from the saved checkpoint")
    if saved and (saved["scheduler"]["stage"] != saved["sampler"]["stage"]
                  or saved["scheduler"]["completed_stage_updates"] != saved["sampler"]["update"]):
        raise RuntimeError("Checkpoint scheduler and sampler positions disagree")
    if saved and int(saved["metadata"]["global_step"]) >= args.until:
        print(f"Task 2 already reached update {saved['metadata']['global_step']}; skipping", flush=True)
        return
    run_config = {"identity": identity, "visual_tokens": 196, "connector": [768, 1024, 1024],
                  "batch_size": {"micro": 8, "accumulation": 2, "effective": 16},
                  "optimizer": {"name": "AdamW", "weight_decay": 0.01, "gradient_clip": 1.0,
                                "reset_at_stage_boundaries": True},
                  "schedule": "5% warmup then cosine to 10% of peak",
                  "stages": [{"stage": stage, "updates": STAGE_UPDATES[stage - 1],
                              "connector_peak_lr": PEAKS[stage][0], "lora_peak_lr": PEAKS[stage][1]}
                             for stage in (1, 2, 3)],
                  "lora": {"rank": 16, "alpha": 32, "dropout": 0.05,
                           "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                  "question_token_cap": 80, "caption_token_cap": 64, "answer_token_cap": 16,
                  "checkpoint_interval": 250, "evaluation": "every 500 updates and stage endpoints"}
    config_path = root / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(config_path),
                    path_in_repo="results/task2/run_config.json")
    stage = int(saved["sampler"]["stage"]) if saved else 1
    tokenizer = AutoTokenizer.from_pretrained(identity["model"], revision=model_revision, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(identity["model"], revision=model_revision, torch_dtype=torch.float16, trust_remote_code=True)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.to("cuda")
    model = VisionLanguageModel(backbone).to("cuda")
    if stage > 1:
        model.enable_lora()
    sampler = CurriculumSampler(manifest["train"])
    optimizer = optimizer_for(model, stage)
    scaler = GradScaler("cuda", init_scale=1024.0)
    if saved:
        load_weights(checkpoint, model.connector, model.backbone)
        sampler.load_state_dict(saved["sampler"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        random.setstate(saved["rng_python"]); np.random.set_state(saved["rng_numpy"])
        torch.set_rng_state(saved["rng_torch"])
        if saved["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        global_step = int(saved["metadata"]["global_step"])
        calibration = saved["metadata"]["calibration"]
        stage_seconds = saved["metadata"]["stage_training_seconds"]
        stage_peaks = saved["metadata"].get("stage_peak_gpu_memory_bytes", {"1": 0, "2": 0, "3": 0})
        print(f"Resuming Task 2 from {checkpoint}, update {global_step}", flush=True)
        for name in ("train.jsonl", "development.jsonl"):
            path = root / name
            if path.exists():
                rows = [line for line in path.read_text(encoding="utf-8").splitlines() if int(json.loads(line)["step"]) <= global_step]
                path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    else:
        for name in ("train.jsonl", "development.jsonl"):
            (root / name).unlink(missing_ok=True)
        global_step = 0
        stage_seconds = {"1": 0.0, "2": 0.0, "3": 0.0}
        stage_peaks = {"1": 0, "2": 0, "3": 0}
        calibration = calibrate(model, tokenizer, features, manifest["train"]["caption"])
        append_json(root / "development.jsonl", evaluate_development(model, tokenizer, features, manifest["development"], 0))
    torch.cuda.reset_peak_memory_stats()
    while global_step < args.until:
        stage = sampler.stage
        if sampler.update == STAGE_UPDATES[stage - 1]:
            sampler.next_stage()
            stage = sampler.stage
            if stage == 2:
                count = model.enable_lora()
                print(f"Enabled LoRA in {count} attention projections", flush=True)
            optimizer = optimizer_for(model, stage)
            torch.cuda.reset_peak_memory_stats()
        model.train()
        started = time.perf_counter()
        examples, mixture = sampler.next_update()
        rate = stage_lr(stage, sampler.update - 1)
        for group in optimizer.param_groups:
            group["lr"] = PEAKS[stage][0 if group["kind"] == "connector" else 1] * rate
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for start in (0, 8):
            rows = examples[start:start + 8]
            prepared = [build_example(tokenizer, row["question"], row["answer"], 64 if "caption_id" in row else 16,
                                      features[row["image_id"]]) for row in rows]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model.loss(prepared) / 2
            scaler.scale(loss).backward()
            total_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        trainable = [p for group in optimizer.param_groups for p in group["params"]]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale:
            raise RuntimeError(f"GradScaler skipped Task 2 update {global_step + 1}")
        global_step += 1
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - started
        stage_seconds[str(stage)] += update_seconds
        stage_peaks[str(stage)] = max(stage_peaks[str(stage)], torch.cuda.max_memory_allocated())
        append_json(root / "train.jsonl", {"step": global_step, "stage": stage, "stage_update": sampler.update,
                                           "loss": total_loss, "mixture": mixture, "grad_norm": float(gradient_norm),
                                           "lr_multiplier": rate, "update_seconds": update_seconds})
        if global_step % 25 == 0:
            average = stage_seconds[str(stage)] / sampler.update
            remaining_hours = average * (STAGE_UPDATES[stage - 1] - sampler.update) / 3600
            print(f"Task 2 stage {stage}, update {global_step}/4500, loss={total_loss:.4f}, "
                  f"stage average {average:.1f}s/update, estimated remaining training {remaining_hours:.1f}h "
                  f"excluding evaluations and checkpoint uploads", flush=True)
        if global_step % 500 == 0 or global_step in (2250, 3750, 4500):
            append_json(root / "development.jsonl", evaluate_development(model, tokenizer, features, manifest["development"], global_step))
        if global_step % 250 == 0:
            metadata = {"identity": identity, "global_step": global_step, "stage": stage,
                        "stage_update": sampler.update, "calibration": calibration,
                        "stage_training_seconds": stage_seconds,
                        "stage_peak_gpu_memory_bytes": stage_peaks,
                        "trainable_connector_parameters": sum(p.numel() for p in model.connector.parameters() if p.requires_grad),
                        "trainable_adapter_parameters": sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)}
            save_checkpoint(root, global_step, model.connector, model.backbone, optimizer, scaler, sampler, metadata)
    print(f"Task 2 training complete through update {global_step}", flush=True)


if __name__ == "__main__":
    main()
