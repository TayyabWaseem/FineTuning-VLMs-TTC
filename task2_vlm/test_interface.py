import unittest

import torch
from torch import nn

from .interface import VisualConnector, assemble_batch, example_weighted_loss


class InterfaceTest(unittest.TestCase):
    def test_visual_insertion_masks_positions_and_gradients(self):
        embedding = nn.Embedding(20, 1024)
        embedding.requires_grad_(False)
        connector = VisualConnector()
        rows = [{"input_ids": [1, 2, 3], "labels": [-100, -100, 3],
                 "visual_insert_index": 1, "features": torch.randn(196, 768)},
                {"input_ids": [4, 5, 6], "labels": [-100, -100, 6],
                 "visual_insert_index": 1, "features": None}]
        batch = assemble_batch(embedding, connector, rows)
        self.assertEqual(batch["inputs_embeds"].shape, (2, 199, 1024))
        self.assertTrue((batch["labels"][0, 1:197] == -100).all())
        self.assertEqual(batch["attention_mask"][1].sum().item(), 3)
        self.assertEqual(batch["position_ids"][0, -1].item(), 198)
        # A frozen causal block still propagates gradients into the visual connector.
        block = nn.Linear(1024, 1024, bias=False).requires_grad_(False)
        head = nn.Linear(1024, 20, bias=False).requires_grad_(False)
        hidden = block(batch["inputs_embeds"].cumsum(dim=1))
        example_weighted_loss(hidden, batch["labels"], head, chunk_size=1).backward()
        self.assertGreater(connector.layers[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(block.weight.grad)

    def test_example_weighting(self):
        hidden = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]],
                               [[1., 0.], [0., 1.], [0., 0.]]], requires_grad=True)
        labels = torch.tensor([[-100, 0, -100], [-100, 0, 1]])
        head = nn.Linear(2, 2, bias=False)
        with torch.no_grad(): head.weight.copy_(torch.eye(2))
        computed = example_weighted_loss(hidden, labels, head, chunk_size=1)
        first = torch.nn.functional.cross_entropy(head(hidden[0, :1]), torch.tensor([0]))
        second = torch.nn.functional.cross_entropy(head(hidden[1, :2]), torch.tensor([0, 1]))
        self.assertTrue(torch.allclose(computed, (first + second) / 2))


if __name__ == "__main__":
    unittest.main()
