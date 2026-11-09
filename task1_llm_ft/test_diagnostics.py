"""Small numerical check of the local weight and activation definitions."""

import unittest

import torch
from torch import nn

from .diagnostics import activation_diagnostics, weight_diagnostics


class TinyModel(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(28):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(3, 3, bias=False)
            with torch.no_grad():
                layer.mlp.down_proj.weight.copy_(factor * torch.eye(3))
            self.layers.append(layer)


class DiagnosticFormulaTest(unittest.TestCase):
    def test_shared_starting_basis_and_inputs(self) -> None:
        base = TinyModel(1.0)
        trained = TinyModel(1.1)
        cache = {
            "u_128": {layer: torch.eye(3) for layer in (0, 13, 27)},
            "z": {kind: {layer: torch.tensor([[1.0, 2.0, 3.0]]) for layer in (0, 13, 27)}
                  for kind in ("math", "non_math")},
            "probe_ids": {"gsm8k": ["0"], "hellaswag": ["0"]},
            "prompts": {"math": ["Q: test\nA:"], "non_math": ["context"]},
        }
        for row in weight_diagnostics(trained, base, cache):
            self.assertAlmostEqual(row["rho_k"], 1.0, places=6)
            self.assertAlmostEqual(row["eta"], 0.1, places=6)
        for kind in ("math", "non_math"):
            for row in activation_diagnostics(trained, base, cache, kind):
                self.assertAlmostEqual(row["correction_ratio"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
