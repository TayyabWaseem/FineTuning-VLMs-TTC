"""LoRA and OPLoRA modules with factor-space projection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class ProjectedLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float,
        protection_rank: int = 0,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.protection_rank = protection_rank
        self.register_buffer("u_protected", torch.empty(base.out_features, 0, dtype=torch.float32), persistent=True)
        self.register_buffer("v_protected", torch.empty(base.in_features, 0, dtype=torch.float32), persistent=True)
        self.base.requires_grad_(False)

    @torch.no_grad()
    def initialize_protection(self, protection_rank: int) -> None:
        if protection_rank <= 0:
            return
        weight = self.base.weight.detach().float()
        max_rank = min(weight.shape)
        k = min(protection_rank, max_rank)
        start = time.perf_counter()
        u, _, vh = torch.linalg.svd(weight, full_matrices=False)
        self.u_protected = u[:, :k].contiguous()
        self.v_protected = vh[:k, :].T.contiguous()
        self.protection_rank = k
        self.protection_init_seconds = time.perf_counter() - start

    def projected_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.lora_A
        b = self.lora_B
        if self.protection_rank <= 0:
            return a, b
        u = self.u_protected.to(device=b.device, dtype=b.dtype)
        v = self.v_protected.to(device=a.device, dtype=a.dtype)
        b_projected = b - u @ (u.T @ b)
        a_projected = a - (a @ v) @ v.T
        return a_projected, b_projected

    def delta_weight(self) -> torch.Tensor:
        a, b = self.projected_factors()
        return self.scaling * (b @ a)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        a, b = self.projected_factors()
        update = self.dropout(inputs).to(dtype=a.dtype) @ a.T @ b.T
        return output + self.scaling * update.to(dtype=output.dtype)


@dataclass
class AdapterStats:
    trainable_parameters: int
    protected_basis_parameters: int
    preprocessing_seconds: float


def _replace_named_linear(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child_name, replacement)


def add_lora(
    model: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    protection_rank: int = 0,
    initialize_protection: bool = True,
) -> AdapterStats:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    paths = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_MODULES
    ]
    started = time.perf_counter()
    basis_parameters = 0
    for path in paths:
        original = model.get_submodule(path)
        replacement = ProjectedLoRALinear(
            original, rank, alpha, dropout, protection_rank=protection_rank
        )
        if protection_rank > 0:
            if initialize_protection:
                replacement.initialize_protection(protection_rank)
            else:
                k = min(protection_rank, original.out_features, original.in_features)
                replacement.u_protected = torch.empty(original.out_features, k, dtype=torch.float32, device=original.weight.device)
                replacement.v_protected = torch.empty(original.in_features, k, dtype=torch.float32, device=original.weight.device)
                replacement.protection_rank = k
            basis_parameters += replacement.u_protected.numel() + replacement.v_protected.numel()
        _replace_named_linear(model, path, replacement)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return AdapterStats(trainable, basis_parameters, time.perf_counter() - started)


def export_projected_adapters(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    started = time.perf_counter()
    exported = {}
    for name, module in model.named_modules():
        if isinstance(module, ProjectedLoRALinear):
            a, b = module.projected_factors()
            exported[name] = {"lora_A": a.detach().cpu(), "lora_B": b.detach().cpu()}
    exported["_metadata"] = {"export_seconds": time.perf_counter() - started}
    return exported
