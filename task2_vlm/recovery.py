"""Save small connector/adapter checkpoints and publish a verified commit marker."""
from __future__ import annotations

import json
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from .sampler import STAGE_UPDATES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _client():
    from huggingface_hub import HfApi
    repo, token = os.environ.get("PA3_HF_REPO", "").strip(), os.environ.get("HF_TOKEN", "").strip()
    if not repo or not token:
        raise RuntimeError("Set PA3_HF_REPO and HF_TOKEN to a private Hugging Face dataset")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise RuntimeError("Recovery dataset must be private")
    return api, repo, token


def save_checkpoint(root: Path, step: int, connector, model, optimizer, scaler, sampler, metadata: dict) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().cpu().contiguous() for name, value in connector.state_dict().items()},
              str(checkpoint / "connector.safetensors"))
    adapter = {name: value.detach().cpu().contiguous() for name, value in model.named_parameters()
               if value.requires_grad}
    save_file(adapter, str(checkpoint / "adapter.safetensors"))
    sampler_state = sampler.state_dict()
    stage = int(sampler_state["stage"])
    scheduler = {"kind": "stage_warmup_cosine_10pct", "stage": stage,
                 "completed_stage_updates": int(sampler_state["update"]),
                 "stage_total_updates": STAGE_UPDATES[stage - 1]}
    state = {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "sampler": sampler_state, "scheduler": scheduler, "rng_python": random.getstate(),
             "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
             "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "metadata": metadata}
    torch.save(state, checkpoint / "training_state.pt")
    (checkpoint / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    data_files = ("connector.safetensors", "adapter.safetensors", "training_state.pt", "metadata.json")
    (checkpoint / "checksums.json").write_text(json.dumps({name: _sha256(checkpoint / name) for name in data_files}), encoding="utf-8")
    api, repo, _ = _client()
    prefix = f"results/task2/{checkpoint.name}"
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(checkpoint), path_in_repo=prefix)
    remote = set(api.list_repo_files(repo, repo_type="dataset"))
    required = (*data_files, "checksums.json")
    if any(f"{prefix}/{name}" not in remote for name in required):
        raise RuntimeError(f"Incomplete remote checkpoint {prefix}")
    # The marker is written last, after the checkpoint and corresponding log.
    for name in ("train.jsonl", "development.jsonl"):
        log = root / name
        if log.exists():
            api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(log), path_in_repo=f"results/task2/{name}")
    marker = root / "latest_verified.json"
    marker.write_text(json.dumps({"checkpoint": checkpoint.name, "step": step}), encoding="utf-8")
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(marker), path_in_repo="results/task2/latest_verified.json")
    # Upload libraries may consume RNG; match a restart from the saved state.
    random.setstate(state["rng_python"]); np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    if state["rng_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return checkpoint


def restore_latest(root: Path, project_root: Path) -> Path | None:
    from huggingface_hub import snapshot_download
    api, repo, token = _client()
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    if "results/task2/latest_verified.json" not in files:
        return None
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(project_root),
                      allow_patterns=["results/task2/latest_verified.json", "results/task2/train.jsonl", "results/task2/development.jsonl"])
    marker = json.loads((root / "latest_verified.json").read_text(encoding="utf-8"))
    name = marker["checkpoint"]
    if not name.startswith("checkpoint-") or not name.removeprefix("checkpoint-").isdigit():
        raise RuntimeError("Invalid checkpoint marker")
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(project_root),
                      allow_patterns=[f"results/task2/{name}/*"])
    checkpoint = root / name
    for required in ("connector.safetensors", "adapter.safetensors", "training_state.pt", "metadata.json", "checksums.json"):
        if not (checkpoint / required).is_file():
            raise RuntimeError(f"Missing recovery file {checkpoint / required}")
    checksums = json.loads((checkpoint / "checksums.json").read_text(encoding="utf-8"))
    if set(checksums) != {"connector.safetensors", "adapter.safetensors", "training_state.pt", "metadata.json"}:
        raise RuntimeError("Recovery checksum manifest is incomplete")
    for name, expected in checksums.items():
        if _sha256(checkpoint / name) != expected:
            raise RuntimeError(f"Recovery checksum failed: {name}")
    return checkpoint


def load_weights(checkpoint: Path, connector, model) -> None:
    connector.load_state_dict(load_file(str(checkpoint / "connector.safetensors")))
    adapter = load_file(str(checkpoint / "adapter.safetensors"))
    expected = {name for name, value in model.named_parameters() if value.requires_grad}
    if set(adapter) != expected:
        raise RuntimeError("Recovery adapter parameters differ from the constructed training stage")
    incompatible = model.load_state_dict(adapter, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {incompatible.unexpected_keys[:5]}")
