"""Build Task 2 split manifests from extracted official COCO and VQAv2 JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .normalize import normalize_answer
from .selection import (assert_split_isolation, grounding_pools, natural_candidates,
                        select_captions, select_natural_splits)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def images_in_pairs(pairs: list[dict]) -> set[int]:
    return {int(example["image_id"]) for pair in pairs for example in pair["examples"]}


def mismatched_map(pairs: list[dict], rng: np.random.Generator) -> dict[str, list[int]]:
    result = {}
    for pair in pairs:
        own = {item["image_id"] for item in pair["examples"]}
        candidates = sorted({item["image_id"] for other in pairs if other["pair_id"] != pair["pair_id"]
                             for item in other["examples"] if item["image_id"] not in own})
        if not candidates:
            raise ValueError(f"No eligible mismatched image for {pair['pair_id']}")
        result[pair["pair_id"]] = [int(candidates[int(rng.integers(len(candidates)))]) for _ in range(2)]
    return result


def prepare(root: Path, output: Path, image_dir: Path) -> None:
    rng = np.random.default_rng(6304)
    coco_train = load(root / "captions_train2014.json")
    coco_val = load(root / "captions_val2014.json")
    instance_train = load(root / "instances_train2014.json")
    instance_val = load(root / "instances_val2014.json")
    train_q = load(root / "v2_OpenEnded_mscoco_train2014_questions.json")["questions"]
    val_q = load(root / "v2_OpenEnded_mscoco_val2014_questions.json")["questions"]
    train_a = load(root / "v2_mscoco_train2014_annotations.json")["annotations"]
    val_a = load(root / "v2_mscoco_val2014_annotations.json")["annotations"]
    train_pairs = load(root / "v2_mscoco_train2014_complementary_pairs.json")
    val_pairs = load(root / "v2_mscoco_val2014_complementary_pairs.json")
    natural = select_natural_splits(natural_candidates(train_q, train_a, train_pairs, normalize_answer),
                                    natural_candidates(val_q, val_a, val_pairs, normalize_answer))
    train_ids = {int(row["id"]) for row in coco_train["images"]}
    val_ids = {int(row["id"]) for row in coco_val["images"]}
    train_captions = select_captions(coco_train["annotations"], train_ids, 6000, 3, rng)
    train_caption_ids = {row["image_id"] for row in train_captions}
    natural_dev, natural_test = images_in_pairs(natural["development"]), images_in_pairs(natural["test"])
    remaining = sorted(val_ids - natural_dev - natural_test)
    dev_captions = select_captions(coco_val["annotations"], set(remaining), 96, 1, rng)
    caption_dev_ids = {row["image_id"] for row in dev_captions}
    remaining = np.array(sorted(set(remaining) - caption_dev_ids), dtype=np.int64)
    shuffled = rng.permutation(remaining)
    ground_dev_ids = set(map(int, shuffled[:len(shuffled)//2]))
    ground_test_ids = set(map(int, shuffled[len(shuffled)//2:]))
    assert_split_isolation({"natural_dev": natural_dev, "natural_test": natural_test,
                            "caption_dev": caption_dev_ids, "ground_dev": ground_dev_ids,
                            "ground_test": ground_test_ids, "train": train_ids})
    train_ground = grounding_pools(train_caption_ids, instance_train["annotations"], instance_train["categories"], rng)
    dev_ground = grounding_pools(ground_dev_ids, instance_val["annotations"], instance_val["categories"], rng, 12)
    test_ground = grounding_pools(ground_test_ids, instance_val["annotations"], instance_val["categories"], rng, 32)
    ground_dev = dev_ground["presence"] + dev_ground["counting"]
    ground_test = test_ground["presence"] + test_ground["counting"]
    for row in train_captions + dev_captions:
        row["question"] = "Describe this image."
    manifest = {"seed": 6304, "normalizer": "task2_vlm.normalize.normalize_answer",
                "train": {"caption": train_captions, "natural": natural["train"], **train_ground},
                "development": {"caption": dev_captions, "natural": natural["development"], "grounding": ground_dev},
                "test": {"natural": natural["test"], "grounding": ground_test},
                "mismatched_images": {"natural": mismatched_map(natural["test"], rng),
                                       "grounding": mismatched_map(ground_test, rng)}}
    all_train = train_caption_ids | images_in_pairs(natural["train"])
    all_val = caption_dev_ids | natural_dev | natural_test | images_in_pairs(ground_dev) | images_in_pairs(ground_test)
    image_files = {int(row["id"]): ("train2014", row["file_name"]) for row in coco_train["images"]}
    image_files.update({int(row["id"]): ("val2014", row["file_name"]) for row in coco_val["images"]})
    manifest["images"] = [{"id": image, "split": image_files[image][0], "filename": image_files[image][1],
                           "path": str(image_dir / image_files[image][0] / image_files[image][1])}
                          for image in sorted(all_train | all_val)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Selected {len(train_captions)} captions, {len(natural['train'])} VQA train pairs, {len(manifest['images'])} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.annotations, args.output, args.images)


if __name__ == "__main__":
    main()
