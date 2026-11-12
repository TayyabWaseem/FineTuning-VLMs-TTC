"""Publish and restore the fixed data manifest and complete feature cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .recovery import _client


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def publish(project: Path) -> None:
    root = project / "results/task2"
    paths = [root / "manifest.json", root / "features/features.npy", root / "features/feature_index.json"]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("Cannot publish incomplete data preparation")
    api, repo, _ = _client()
    hashes = {}
    for path in paths:
        relative = path.relative_to(project).as_posix()
        hashes[relative] = sha256(path)
        print(f"Uploading {relative}", flush=True)
        api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(path), path_in_repo=relative)
    marker = root / "prepared.json"
    marker.write_text(json.dumps({"files": hashes}, indent=2), encoding="utf-8")
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(marker), path_in_repo="results/task2/prepared.json")
    print("Prepared data and feature cache uploaded", flush=True)


def restore(project: Path) -> bool:
    from huggingface_hub import snapshot_download
    api, repo, token = _client()
    if "results/task2/prepared.json" not in api.list_repo_files(repo, repo_type="dataset"):
        return False
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(project),
                      allow_patterns=["results/task2/prepared.json", "results/task2/manifest.json", "results/task2/features/*"])
    marker = json.loads((project / "results/task2/prepared.json").read_text(encoding="utf-8"))
    for name, digest in marker["files"].items():
        path = (project / name).resolve()
        if not path.is_relative_to((project / "results/task2").resolve()) or not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Prepared artifact is missing or changed: {name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("restore", "publish"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.stage == "publish":
        publish(project)
    else:
        print("Prepared artifacts available" if restore(project) else "No prepared artifacts found")


if __name__ == "__main__":
    main()
