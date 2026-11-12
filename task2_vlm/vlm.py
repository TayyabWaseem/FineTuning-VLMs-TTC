"""Frozen Qwen backbone with connector and optional attention-only LoRA."""
from __future__ import annotations

import math
import torch
from torch import nn

from .interface import VisualConnector, assemble_batch, example_weighted_loss


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, inputs):
        original = self.base(inputs)
        update = self.dropout(inputs).float() @ self.lora_A.T @ self.lora_B.T
        return original + self.scaling * update.to(original.dtype)


class VisionLanguageModel(nn.Module):
    def __init__(self, backbone: nn.Module, connector: VisualConnector | None = None):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.connector = connector if connector is not None else VisualConnector()

    def enable_lora(self) -> int:
        names = [name for name, module in self.backbone.named_modules()
                 if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")]
        if not names:
            raise RuntimeError("No Qwen attention projections found")
        for name in names:
            parent_name, _, leaf = name.rpartition(".")
            parent = self.backbone.get_submodule(parent_name)
            setattr(parent, leaf, LoRALinear(getattr(parent, leaf)))
        return len(names)

    def batch(self, examples):
        return assemble_batch(self.backbone.get_input_embeddings(), self.connector, examples)

    def hidden(self, batch):
        return self.backbone.model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                                   position_ids=batch["position_ids"], use_cache=False).last_hidden_state

    def loss(self, examples):
        batch = self.batch(examples)
        hidden = self.hidden(batch)
        return example_weighted_loss(hidden, batch["labels"], self.backbone.lm_head)
