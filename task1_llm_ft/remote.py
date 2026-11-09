"""Private, restartable checkpoint storage for long Kaggle sessions.

Set PA3_HF_REPO to a private Hugging Face dataset repo (owner/name) and
HF_TOKEN to a write token. Neither value is stored in the notebook or outputs.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "results/task1"


def _client():
    repo_id = os.environ.get("PA3_HF_REPO", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not repo_id or not token:
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise RuntimeError("Checkpoint repository must be private")
    return api, repo_id, token


def require_remote() -> None:
    if _client() is None:
        raise RuntimeError("Set PA3_HF_REPO and HF_TOKEN for restartable Kaggle training")


def sync_checkpoint(checkpoint_dir: Path, output_dir: Path) -> None:
    relative_output = output_dir.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    if relative_output.startswith("results/task1_smoke/"):
        # Keep smoke checkpoints local: the smoke suite intentionally resumes
        # from an earlier boundary and checks each recovery file afterward.
        return
    client = _client()
    if client is None:
        return
    api, repo_id, _ = client
    method = output_dir.name
    path_in_repo = f"{REMOTE_ROOT}/{method}/{checkpoint_dir.name}"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(checkpoint_dir),
        path_in_repo=path_in_repo,
        commit_message=f"{method} {checkpoint_dir.name} recovery checkpoint",
    )
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    required = ["training_state.pt", "run_config.json"]
    required.append(
        "adapter_model.safetensors" if (checkpoint_dir / "adapter_model.safetensors").exists()
        else "model.safetensors"
    )
    for filename in required:
        if f"{path_in_repo}/{filename}" not in remote_files:
            raise RuntimeError(f"Remote checkpoint incomplete: {path_in_repo}/{filename}")
    log_path = output_dir / "train.jsonl"
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(log_path),
        path_in_repo=f"{REMOTE_ROOT}/{method}/train.jsonl",
        commit_message=f"{method} progress through {checkpoint_dir.name}",
    )
    index_path = output_dir / "remote_checkpoints.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    index[checkpoint_dir.name] = {
        "repo": repo_id, "path": path_in_repo, "verified": True,
        "size_bytes": sum(p.stat().st_size for p in checkpoint_dir.rglob("*") if p.is_file()),
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    sync_artifact(index_path, PROJECT_ROOT)
    # A successfully uploaded checkpoint is recoverable without consuming local disk.
    for old in output_dir.glob("checkpoint-*"):
        if old == checkpoint_dir or not old.is_dir():
            continue
        if method == "full_ft" and old.name == "checkpoint-20000":
            (old / "training_state.pt").unlink(missing_ok=True)
            continue
        if old.name in index:
            if not old.resolve().is_relative_to(output_dir.resolve()):
                raise RuntimeError(f"Refusing to remove checkpoint outside output directory: {old}")
            shutil.rmtree(old)


def sync_artifact(path: Path, root: Path) -> None:
    client = _client()
    if client is None:
        return
    api, repo_id, _ = client
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(path),
        path_in_repo=relative,
        commit_message=f"Save {relative}",
    )


def restore_latest(output_dir: Path) -> Path | None:
    client = _client()
    if client is None:
        return None
    api, repo_id, token = client
    from huggingface_hub import snapshot_download

    method = output_dir.name
    paths = api.list_repo_files(repo_id, repo_type="dataset")
    prefix = f"{REMOTE_ROOT}/{method}/checkpoint-"
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", token=token,
        local_dir=str(PROJECT_ROOT),
        allow_patterns=[f"{REMOTE_ROOT}/{method}/remote_checkpoints.json"],
    )
    index_path = output_dir / "remote_checkpoints.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    counts = sorted(
        int(path[len(prefix):].split("/")[0])
        for path in paths
        if path.startswith(prefix) and path.endswith("/training_state.pt")
        and path[len(prefix):].split("/")[0].isdigit()
        and index.get(f"checkpoint-{path[len(prefix):].split('/')[0]}", {}).get("verified")
        and (
            f"{prefix}{path[len(prefix):].split('/')[0]}/adapter_model.safetensors" in paths
            or f"{prefix}{path[len(prefix):].split('/')[0]}/model.safetensors" in paths
        )
    )
    if not counts:
        return None
    latest = f"checkpoint-{counts[-1]}"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        local_dir=str(PROJECT_ROOT),
        allow_patterns=[f"{REMOTE_ROOT}/{method}/{latest}/*", f"{REMOTE_ROOT}/{method}/train.jsonl", f"{REMOTE_ROOT}/{method}/remote_checkpoints.json", f"{REMOTE_ROOT}/{method}/metadata.json"],
    )
    checkpoint = output_dir / latest
    required = ["training_state.pt", "run_config.json"]
    required.append("adapter_model.safetensors" if (checkpoint / "adapter_model.safetensors").exists()
                    else "model.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded checkpoint is incomplete at {checkpoint}; missing {missing}")
    return checkpoint


def restore_evaluation_checkpoint(output_dir: Path, count: int) -> Path:
    checkpoint = output_dir / f"checkpoint-{count}"
    if (checkpoint / "adapter_model.safetensors").exists() or (checkpoint / "config.json").exists():
        return checkpoint
    client = _client()
    if client is None:
        raise RuntimeError(f"Missing local checkpoint and remote storage: {checkpoint}")
    _, repo_id, token = client
    from huggingface_hub import snapshot_download

    pattern = f"{REMOTE_ROOT}/{output_dir.name}/{checkpoint.name}/*"
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", token=token,
        local_dir=str(PROJECT_ROOT),
        allow_patterns=[pattern],
        ignore_patterns=[f"{REMOTE_ROOT}/{output_dir.name}/{checkpoint.name}/training_state.pt"],
    )
    return checkpoint


def restore_artifacts(root: Path, patterns: list[str]) -> None:
    client = _client()
    if client is None:
        raise RuntimeError("Remote artifacts require PA3_HF_REPO and HF_TOKEN")
    _, repo_id, token = client
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id, repo_type="dataset", token=token,
        local_dir=str(root), allow_patterns=patterns,
    )
