"""Cache normalized SigLIP patch embeddings for an explicit image manifest."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel


@torch.no_grad()
def cache_features(images: list[dict], output: Path, batch_size: int = 16) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Feature caching requires a CUDA GPU")
    if not images or len({str(row["id"]) for row in images}) != len(images):
        raise ValueError("Image manifest must have unique, nonempty image IDs")
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.float16).to("cuda").eval()
    encoder.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    temporary = output / "features.incomplete.npy"
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(len(images), 196, 768))
    for start in range(0, len(images), batch_size):
        rows = images[start:start + batch_size]
        with_images = []
        for row in rows:
            with Image.open(row["path"]) as image:
                with_images.append(image.convert("RGB"))
        pixels = processor(images=with_images, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
        # SiglipVisionModel's last_hidden_state already includes post_layernorm.
        features = encoder(pixel_values=pixels).last_hidden_state
        if tuple(features.shape[1:]) != (196, 768):
            raise RuntimeError(f"Unexpected SigLIP features: {tuple(features.shape)}")
        array[start:start + len(rows)] = features.cpu().numpy()
        print(f"Cached {start + len(rows)}/{len(images)} images", flush=True)
    array.flush()
    del array
    temporary.replace(output / "features.npy")
    metadata = {"model": "google/siglip-base-patch16-224", "dtype": "float16", "shape": [len(images), 196, 768],
                "image_ids": [str(row["id"]) for row in images],
                "elapsed_seconds": time.perf_counter() - started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "normalization": "SiglipVisionModel.last_hidden_state (post_layernorm applied once)"}
    (output / "feature_index.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Feature cache complete: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="JSON array of {id, path} records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    cache_features(json.loads(args.images.read_text(encoding="utf-8")), args.output, args.batch_size)


if __name__ == "__main__":
    main()
