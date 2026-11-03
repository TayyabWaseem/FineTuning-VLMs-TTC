"""CPU unit tests for training math helpers."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from .train import add_l2_sp_gradients, l2_sp_penalty


class L2SPGradientTest(unittest.TestCase):
    def test_gradient_matches_lambda_times_displacement(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        reference = {"weight": torch.tensor([[1.5, 0.5]])}
        coefficient = 0.001

        add_l2_sp_gradients(model, reference, coefficient)

        expected = coefficient * (model.weight.detach() - reference["weight"])
        self.assertTrue(torch.equal(model.weight.grad, expected))
        self.assertAlmostEqual(
            l2_sp_penalty(model, reference).item(),
            0.5 * (model.weight.detach() - reference["weight"]).square().sum().item(),
        )


if __name__ == "__main__":
    unittest.main()
