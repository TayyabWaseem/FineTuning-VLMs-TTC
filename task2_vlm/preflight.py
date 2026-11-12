"""Fail before training if selected Task 2 data violate the required split."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .feature_store import FeatureStore
from .selection import assert_split_isolation, TRAIN_QUOTAS, DEV_QUOTAS, TEST_QUOTAS


def pair_images(pairs):
    return {int(example["image_id"]) for pair in pairs for example in pair["examples"]}


def check_natural(pairs, quotas, name):
    if Counter(pair["question_type"] for pair in pairs) != Counter(quotas):
        raise RuntimeError(f"{name}: incorrect natural VQA category counts")
    for pair in pairs:
        items = pair["examples"]
        if (len(items) != 2 or items[0]["image_id"] == items[1]["image_id"]
                or re.sub(r"\s+", " ", items[0]["question"].strip()).casefold() != re.sub(r"\s+", " ", items[1]["question"].strip()).casefold()
                or items[0]["canonical_normalized"] == items[1]["canonical_normalized"]
                or any(row["agreement"] < 8 or len(row["references"]) != 10 for row in items)):
            raise RuntimeError(f"{name}: invalid natural pair {pair['pair_id']}")


def validate(manifest: dict, features: FeatureStore) -> dict:
    train, dev, test = manifest["train"], manifest["development"], manifest["test"]
    if len(train["caption"]) != 18000 or len(dev["caption"]) != 96:
        raise RuntimeError("Caption example counts do not match the assignment")
    caption_counts = Counter(row["image_id"] for row in train["caption"])
    if len(caption_counts) != 6000 or set(caption_counts.values()) != {3}:
        raise RuntimeError("Training captions must contain three captions for each of 6,000 images")
    if len({row["image_id"] for row in dev["caption"]}) != 96:
        raise RuntimeError("Development captions must use 96 distinct images")
    for name, pairs, quotas in (("train", train["natural"], TRAIN_QUOTAS),
                                ("development", dev["natural"], DEV_QUOTAS),
                                ("test", test["natural"], TEST_QUOTAS)):
        check_natural(pairs, quotas, name)
    for name, pairs, count in (("development", dev["grounding"], 24), ("test", test["grounding"], 64)):
        if len(pairs) != count or Counter(pair["question_type"] for pair in pairs) != {"presence": count // 2, "counting": count // 2}:
            raise RuntimeError(f"{name}: grounding pair counts are incorrect")
    training_images = set(caption_counts) | pair_images(train["natural"]) | pair_images(train["presence"]) | pair_images(train["counting"])
    groups = {"train": training_images, "caption_dev": {row["image_id"] for row in dev["caption"]},
              "natural_dev": pair_images(dev["natural"]), "ground_dev": pair_images(dev["grounding"]),
              "natural_test": pair_images(test["natural"]), "ground_test": pair_images(test["grounding"])}
    assert_split_isolation(groups)
    required = set.union(*groups.values())
    missing = {str(image) for image in required} - set(features.rows)
    if missing:
        raise RuntimeError(f"Feature cache is missing images: {sorted(missing)[:5]}")
    for suite in ("natural", "grounding"):
        pairs = test[suite]
        all_images = pair_images(pairs)
        mapping = manifest["mismatched_images"][suite]
        for pair in pairs:
            own = {item["image_id"] for item in pair["examples"]}
            mapped = mapping[pair["pair_id"]]
            question = " ".join(pair["examples"][0]["question"].casefold().split())
            other_images = pair_images([other for other in pairs if other["pair_id"] != pair["pair_id"]
                and " ".join(other["examples"][0]["question"].casefold().split()) != question])
            if len(mapped) != 2 or any(image in own or image not in all_images or image not in other_images for image in mapped):
                raise RuntimeError(f"Invalid mismatched-image control for {pair['pair_id']}")
    return {"status": "PASS", "train_captions": len(train["caption"]), "natural_pairs": {"train": len(train["natural"]),
            "development": len(dev["natural"]), "test": len(test["natural"])},
            "grounding_pairs": {"development": len(dev["grounding"]), "test": len(test["grounding"])},
            "cached_images": len(features.rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.manifest.read_text(encoding="utf-8")), FeatureStore(args.features)), indent=2))


if __name__ == "__main__":
    main()
