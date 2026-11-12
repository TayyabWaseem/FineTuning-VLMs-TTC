"""Cache SigLIP features in remotely recoverable shards, then assemble a memmap."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel
from huggingface_hub import snapshot_download

from .artifacts import sha256
from .download import COCO, fetch
from .recovery import _client


@torch.no_grad()
def cache_features(images: list[dict], output: Path, batch_size: int = 16, shard_size: int = 128) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Feature caching requires a CUDA GPU")
    if not images or len({str(row["id"]) for row in images}) != len(images):
        raise ValueError("Image manifest must have unique, nonempty image IDs")
    project = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output != (project / "results/task2/features").resolve():
        raise ValueError("Recovery storage requires results/task2/features")
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    minimum = len(images) * 196 * 768 * 2 + 3 * 2**30
    if free < minimum:
        raise RuntimeError(f"Need at least {minimum / 2**30:.2f} GiB free for feature cache and working files; found {free / 2**30:.2f} GiB")
    api, repo, token = _client()
    revision = api.model_info("google/siglip-base-patch16-224").sha
    ids = [str(row["id"]) for row in images]
    cache_source = hashlib.sha256(Path(__file__).read_bytes() + (Path(__file__).parent / "download.py").read_bytes()).hexdigest()
    plan = {"model": "google/siglip-base-patch16-224", "model_revision": revision,
            "image_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "cache_source_sha256": cache_source,
            "count": len(ids), "shard_size": shard_size, "batch_size": batch_size,
            "device": torch.cuda.get_device_name(0),
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers")}
    remote = set(api.list_repo_files(repo, repo_type="dataset"))
    prefix = "results/task2/feature_shards"
    if f"{prefix}/plan.json" in remote:
        snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(project),
                          allow_patterns=[f"{prefix}/plan.json"])
        previous = json.loads((project / prefix / "plan.json").read_text(encoding="utf-8"))
        if previous != plan:
            raise RuntimeError("Existing feature shards use a different model or image selection")
    else:
        plan_path = project / prefix / "plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(plan_path), path_in_repo=f"{prefix}/plan.json")
    processor = AutoProcessor.from_pretrained(plan["model"], revision=revision)
    encoder = SiglipVisionModel.from_pretrained(plan["model"], revision=revision, torch_dtype=torch.float16).to("cuda").eval()
    encoder.requires_grad_(False)
    temporary = output / "features.incomplete.npy"
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(len(images), 196, 768))
    elapsed_total, peak = 0.0, 0
    for start in range(0, len(images), shard_size):
        rows = images[start:start + shard_size]
        name = f"shard-{start // shard_size:05d}"
        shard = project / prefix / f"{name}.npy"
        metadata = project / prefix / f"{name}.json"
        shard.parent.mkdir(parents=True, exist_ok=True)
        if f"{prefix}/{name}.json" in remote and f"{prefix}/{name}.npy" in remote:
            snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(project),
                              allow_patterns=[f"{prefix}/{name}.json", f"{prefix}/{name}.npy"])
            details = json.loads(metadata.read_text(encoding="utf-8"))
            if sha256(shard) != details["sha256"] or details["count"] != len(rows):
                raise RuntimeError(f"Corrupt feature shard {name}")
            values = np.load(shard, mmap_mode="r")
        else:
            started = time.perf_counter()
            def get_image(row):
                fetch(f"{COCO}/{row['split']}/{row['filename']}", Path(row["path"]))
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(get_image, rows))
            torch.cuda.reset_peak_memory_stats()
            values = np.empty((len(rows), 196, 768), dtype=np.float16)
            for offset in range(0, len(rows), batch_size):
                subset = rows[offset:offset + batch_size]
                pictures = []
                for row in subset:
                    with Image.open(row["path"]) as image:
                        pictures.append(image.convert("RGB"))
                pixels = processor(images=pictures, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
                features = encoder(pixel_values=pixels).last_hidden_state
                if tuple(features.shape[1:]) != (196, 768):
                    raise RuntimeError(f"Unexpected SigLIP features: {tuple(features.shape)}")
                values[offset:offset + len(subset)] = features.cpu().numpy()
            np.save(shard, values)
            details = {"count": len(rows), "sha256": sha256(shard), "elapsed_seconds": time.perf_counter() - started,
                       "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
            metadata.write_text(json.dumps(details, indent=2), encoding="utf-8")
            api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(shard), path_in_repo=f"{prefix}/{name}.npy")
            api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(metadata), path_in_repo=f"{prefix}/{name}.json")
        if tuple(values.shape) != (len(rows), 196, 768):
            raise RuntimeError(f"Invalid feature shard dimensions: {name}")
        array[start:start + len(rows)] = values
        elapsed_total += details["elapsed_seconds"]
        peak = max(peak, details["peak_gpu_memory_bytes"])
        del values
        shard.unlink(missing_ok=True)  # A verified remote shard can be restored if needed.
        print(f"Feature cache: {start + len(rows)}/{len(images)} images verified", flush=True)
    array.flush()
    del array
    temporary.replace(output / "features.npy")
    index = {"model": plan["model"], "model_revision": revision, "dtype": "float16", "shape": [len(images), 196, 768],
             "image_ids": ids, "elapsed_seconds": elapsed_total, "peak_gpu_memory_bytes": peak,
             "timing_definition": "sum of per-shard image fetch and encoding time; excludes remote upload and shard restoration",
             "normalization": "SiglipVisionModel.last_hidden_state (post_layernorm applied once)"}
    (output / "feature_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Feature cache complete: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    data = json.loads(args.images.read_text(encoding="utf-8"))
    cache_features(data["images"] if isinstance(data, dict) else data, args.output, args.batch_size)


if __name__ == "__main__":
    main()
