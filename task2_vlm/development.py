"""Teacher-forced development metrics under correct, swapped, and no-image inputs."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .prompts import build_example


@torch.no_grad()
def per_example_nll(model, tokenizer, features, examples: list[dict], image_ids: list[int | None],
                    caption: bool, batch_size: int = 8) -> list[float]:
    if len(examples) != len(image_ids) or not examples:
        raise ValueError("Evaluation needs matched nonempty examples and image conditions")
    model.eval()
    values = []
    for start in range(0, len(examples), batch_size):
        rows = examples[start:start + batch_size]
        ids = image_ids[start:start + batch_size]
        prepared = [build_example(tokenizer, row["question"], row["answer"], 64 if caption else 16,
                                  features[image] if image is not None else None,
                                  include_eot=caption) for row, image in zip(rows, ids)]
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            batch = model.batch(prepared)
            hidden = model.hidden(batch)
            labels = batch["labels"][:, 1:]
            valid = labels != -100
            counts = valid.sum(dim=1)
            if torch.any(counts == 0):
                raise RuntimeError("No development target tokens")
            states = hidden[:, :-1][valid]
            targets = labels[valid]
            owners = torch.arange(len(rows), device=labels.device).unsqueeze(1).expand_as(labels)[valid]
            sums = torch.zeros(len(rows), device=labels.device, dtype=torch.float32)
            for xs, ys, sample_ids in zip(states.split(128), targets.split(128), owners.split(128)):
                losses = F.cross_entropy(model.backbone.lm_head(xs).float(), ys, reduction="none")
                sums.scatter_add_(0, sample_ids, losses)
            values.extend((sums / counts).cpu().tolist())
    return values


def mean_nll(model, tokenizer, features, examples, image_ids, caption, batch_size=8):
    values = per_example_nll(model, tokenizer, features, examples, image_ids, caption, batch_size)
    return sum(values) / len(values)


def evaluate_development(model, tokenizer, features, data: dict, step: int) -> dict:
    captions = data["caption"]
    result = {"step": step, "caption_reference_loss": mean_nll(
        model, tokenizer, features, captions, [row["image_id"] for row in captions], caption=True)}
    for suite in ("natural", "grounding"):
        pairs = data[suite]
        examples = [item for pair in pairs for item in pair["examples"]]
        correct = [item["image_id"] for item in examples]
        swapped = [item["image_id"] for pair in pairs for item in reversed(pair["examples"])]
        scores = {"correct": mean_nll(model, tokenizer, features, examples, correct, caption=False),
                  "complementary": mean_nll(model, tokenizer, features, examples, swapped, caption=False),
                  "no_image": mean_nll(model, tokenizer, features, examples, [None] * len(examples), caption=False)}
        scores["swap_penalty"] = scores["complementary"] - scores["correct"]
        result[suite] = scores
    return result
