"""Deterministic VQAv2 pair filtering and image-disjoint selection."""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import numpy as np


TRAIN_QUOTAS = {"yes/no": 640, "number": 256, "other": 640}
DEV_QUOTAS = {"yes/no": 12, "number": 8, "other": 12}
TEST_QUOTAS = {"yes/no": 36, "number": 24, "other": 36}


def natural_candidates(questions: list[dict], annotations: list[dict], pairs: list[list[int]],
                       normalize: Callable[[str], str]) -> list[dict]:
    """Filter official VQAv2 records; retain all raw and normalized references."""
    question_map = {int(item["question_id"]): item for item in questions}
    answer_map = {int(item["question_id"]): item for item in annotations}
    candidates = []
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1]:
            continue
        try:
            qs = [question_map[int(q)] for q in pair]
            ans = [answer_map[int(q)] for q in pair]
        except KeyError:
            continue
        if len({int(q["image_id"]) for q in qs}) != 2:
            continue
        if re.sub(r"\s+", " ", qs[0]["question"].strip()).casefold() != re.sub(r"\s+", " ", qs[1]["question"].strip()).casefold():
            continue
        categories = [item.get("answer_type") for item in ans]
        if categories[0] != categories[1] or categories[0] not in TRAIN_QUOTAS:
            continue
        examples = []
        for q, annotation in zip(qs, ans):
            references = [item["answer"] for item in annotation["answers"]]
            if len(references) != 10:
                break
            normalized = [normalize(answer) for answer in references]
            canonical, count = Counter(normalized).most_common(1)[0]
            if not canonical or count < 8:
                break
            raw_target = next(raw for raw, norm in zip(references, normalized) if norm == canonical)
            examples.append({"question_id": int(q["question_id"]), "image_id": int(q["image_id"]),
                             "question": q["question"], "answer": raw_target,
                             "canonical_normalized": canonical, "references": references,
                             "references_normalized": normalized, "agreement": count})
        if len(examples) == 2 and examples[0]["canonical_normalized"] != examples[1]["canonical_normalized"]:
            candidates.append({"pair_id": f"{int(pair[0])}-{int(pair[1])}", "question_type": categories[0], "examples": examples})
    return candidates


def _draw(candidates: list[dict], quotas: dict[str, int], rng: np.random.Generator,
          forbidden_images: set[int], forbidden_questions: set[int]) -> list[dict]:
    remaining = dict(quotas)
    chosen, used_questions = [], set(forbidden_questions)
    for idx in rng.permutation(len(candidates)):
        item = candidates[int(idx)]
        kind = item["question_type"]
        images = {example["image_id"] for example in item["examples"]}
        questions = {example["question_id"] for example in item["examples"]}
        if remaining.get(kind, 0) <= 0 or images & forbidden_images or questions & used_questions:
            continue
        chosen.append(item)
        used_questions.update(questions)
        remaining[kind] -= 1
        if not any(remaining.values()):
            return chosen
    raise ValueError(f"Insufficient eligible image-disjoint VQA pairs; missing {remaining}")


def select_natural_splits(train: list[dict], validation: list[dict], seed: int = 6304) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    training = _draw(train, TRAIN_QUOTAS, rng, set(), set())
    test = _draw(validation, TEST_QUOTAS, rng, set(), set())
    test_images = {e["image_id"] for pair in test for e in pair["examples"]}
    test_questions = {e["question_id"] for pair in test for e in pair["examples"]}
    development = _draw(validation, DEV_QUOTAS, rng, test_images, test_questions)
    return {"train": training, "development": development, "test": test}


def assert_split_isolation(splits: dict[str, set[int]]) -> None:
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            overlap = splits[name] & splits[other]
            if overlap:
                raise ValueError(f"Image leakage between {name} and {other}: {sorted(overlap)[:5]}")


def select_captions(annotations: list[dict], image_ids: set[int], image_count: int,
                    captions_per_image: int, rng: np.random.Generator) -> list[dict]:
    by_image: dict[int, list[dict]] = {}
    for row in annotations:
        image_id = int(row["image_id"])
        if image_id in image_ids and str(row.get("caption", "")).strip():
            by_image.setdefault(image_id, []).append(row)
    eligible = sorted(image for image, captions in by_image.items() if len(captions) >= captions_per_image)
    if len(eligible) < image_count:
        raise ValueError(f"Need {image_count} caption images; found {len(eligible)}")
    selected = rng.choice(eligible, image_count, replace=False)
    return [{"image_id": int(image), "caption_id": int(row["id"]), "answer": row["caption"]}
            for image in selected
            for row in sorted(by_image[int(image)], key=lambda item: int(item["id"]))[:captions_per_image]]


def grounding_pools(image_ids: set[int], annotations: list[dict], categories: list[dict],
                    rng: np.random.Generator, draws_per_type: int = 20000) -> dict[str, list[dict]]:
    """Sample reusable annotation-derived presence and counting pairs.

    Missing COCO annotations are treated as absence; this is an annotation
    heuristic and must be documented when interpreting the results.
    """
    image_ids = set(map(int, image_ids))
    counts: dict[int, Counter] = {int(category["id"]): Counter() for category in categories}
    crowded: dict[int, set[int]] = {category: set() for category in counts}
    for annotation in annotations:
        image, category = int(annotation["image_id"]), int(annotation["category_id"])
        if image in image_ids and category in counts:
            if annotation.get("iscrowd", 0):
                crowded[category].add(image)
            else:
                counts[category][image] += 1
    pools = {"presence": [], "counting": []}
    seen = {"presence": set(), "counting": set()}
    names = {int(category["id"]): category["name"] for category in categories}
    ids = np.array(sorted(image_ids), dtype=np.int64)
    if len(ids) < 2:
        raise ValueError("Grounding pool requires at least two images")
    eligible = {}
    for category, counter in counts.items():
        buckets = {count: [int(image) for image in ids if int(image) not in crowded[category] and counter[int(image)] == count] for count in range(5)}
        for kind in pools:
            possible = ([0, 1, 2, 3, 4] if kind == "presence" else [1, 2, 3, 4])
            present = [value for value in possible if buckets[value]]
            if (kind == "presence" and 0 in present and len(present) > 1) or (kind == "counting" and len(present) > 1):
                eligible.setdefault(kind, []).append((category, buckets, present))
    for kind in pools:
        options = eligible.get(kind, [])
        if not options:
            raise ValueError(f"No eligible {kind} grounding pairs")
        for _ in range(draws_per_type * 3):
            category, buckets, available = options[int(rng.integers(len(options)))]
            first = 0 if kind == "presence" else int(rng.choice(available))
            second = int(rng.choice([value for value in available if value != first]))
            image_a = buckets[first][int(rng.integers(len(buckets[first])))]
            image_b = buckets[second][int(rng.integers(len(buckets[second])))]
            key = (category, first, second, image_a, image_b)
            if key in seen[kind]:
                continue
            seen[kind].add(key)
            question = (f"Does this image contain {names[category]}?" if kind == "presence"
                        else f"How many {names[category]} can you see?")
            answer_a = ("no" if first == 0 else "yes") if kind == "presence" else str(first)
            answer_b = ("no" if second == 0 else "yes") if kind == "presence" else str(second)
            pools[kind].append({"pair_id": f"{kind}-{category}-{image_a}-{image_b}", "question_type": kind,
                                "examples": [{"image_id": image_a, "question": question, "answer": answer_a},
                                             {"image_id": image_b, "question": question, "answer": answer_b}]})
            if len(pools[kind]) == draws_per_type:
                break
        if len(pools[kind]) < draws_per_type:
            raise ValueError(f"Only generated {len(pools[kind])} unique {kind} pairs")
    return pools
