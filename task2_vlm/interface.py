"""Visual connector, token insertion, and example-weighted causal loss primitives."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class VisualConnector(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 1024))
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 768 or features.shape[1] not in (196, 49):
            raise ValueError("Expected batch x 196 (or 49) x 768 vision features")
        return self.layers(features.float()) * self.scale


def assemble_batch(embedding: nn.Module, connector: VisualConnector, examples: list[dict]) -> dict[str, torch.Tensor]:
    """Insert visual tokens at each example's precomputed user-message position.

    Each example supplies input_ids, labels, visual_insert_index and either a
    196x768 feature tensor or None. None removes visual tokens entirely.
    """
    device = next(embedding.parameters()).device
    visual_by_index = {}
    groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for index, row in enumerate(examples):
        if row.get("features") is not None:
            feature = torch.as_tensor(row["features"])
            if feature.ndim != 2 or feature.shape[-1] != 768 or feature.shape[0] not in (49, 196):
                raise ValueError("Each image requires 196 (or 49) x 768 features")
            groups.setdefault(feature.shape[0], []).append((index, feature))
    for group in groups.values():
        patches = torch.stack([feature for _, feature in group]).to(device=device, dtype=torch.float32)
        projected = connector(patches)
        for offset, (index, _) in enumerate(group):
            visual_by_index[index] = projected[offset]
    vectors, targets = [], []
    for row_index, row in enumerate(examples):
        ids = torch.as_tensor(row["input_ids"], dtype=torch.long, device=device)
        labels = torch.as_tensor(row["labels"], dtype=torch.long, device=device)
        index = int(row["visual_insert_index"])
        if ids.ndim != 1 or ids.shape != labels.shape or not 0 <= index <= ids.numel():
            raise ValueError("Invalid text tokens, labels, or visual insertion index")
        text = embedding(ids)
        if row_index in visual_by_index:
            visual = visual_by_index[row_index].to(text.dtype)
            vectors.append(torch.cat([text[:index], visual, text[index:]], dim=0))
            targets.append(torch.cat([labels[:index], labels.new_full((visual.shape[0],), -100), labels[index:]]))
        else:
            vectors.append(text)
            targets.append(labels)
    if not vectors:
        raise ValueError("Empty batch")
    maximum = max(item.shape[0] for item in vectors)
    inputs, labels, masks = [], [], []
    for vector, target in zip(vectors, targets):
        padding = maximum - vector.shape[0]
        inputs.append(torch.cat([vector, vector.new_zeros((padding, vector.shape[-1]))]))
        labels.append(torch.cat([target, target.new_full((padding,), -100)]))
        masks.append(torch.cat([target.new_ones(target.shape[0]), target.new_zeros(padding)]))
    mask = torch.stack(masks)
    return {"inputs_embeds": torch.stack(inputs), "labels": torch.stack(labels),
            "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp_min(0)}


def example_weighted_loss(hidden: torch.Tensor, labels: torch.Tensor, lm_head: nn.Module, chunk_size: int = 128) -> torch.Tensor:
    """Average response-token CE within each example, then across examples."""
    targets = labels[:, 1:]
    features = hidden[:, :-1, :]
    keep = targets != -100
    counts = keep.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Each example must have at least one supervised token")
    weights = (1.0 / (counts.float() * targets.shape[0])).unsqueeze(1).expand_as(targets)[keep]
    states, truth = features[keep], targets[keep]
    def part(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (F.cross_entropy(lm_head(x).float(), y, reduction="none") * w).sum()
    return sum(checkpoint(part, x, y, w, use_reentrant=False)
               for x, y, w in zip(states.split(chunk_size), truth.split(chunk_size), weights.split(chunk_size)))
