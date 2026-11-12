import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from .development import per_example_nll


class DevelopmentTest(unittest.TestCase):
    def test_batched_nll_matches_separate_example_means(self):
        class Model:
            def __init__(self):
                self.backbone = SimpleNamespace(lm_head=nn.Linear(3, 7, bias=False))
            def eval(self): return self
            def batch(self, rows):
                return {"labels": torch.tensor([row["labels"] for row in rows]),
                        "hidden": torch.stack([row["hidden"] for row in rows])}
            def hidden(self, batch): return batch["hidden"]
        torch.manual_seed(8)
        model = Model()
        rows = [{"labels": [-100, 2, -100, -100], "hidden": torch.randn(4, 3)},
                {"labels": [-100, 1, 3, 4], "hidden": torch.randn(4, 3)}]
        expected = []
        for row in rows:
            labels = torch.tensor(row["labels"][1:])
            valid = labels != -100
            expected.append(float(F.cross_entropy(model.backbone.lm_head(row["hidden"][:-1][valid]), labels[valid])))
        examples = [{"question": "q1", "answer": "a"}, {"question": "q2", "answer": "b"}]
        with patch("task2_vlm.development.build_example", side_effect=rows):
            actual = per_example_nll(model, None, {}, examples, [None, None], caption=False)
        self.assertTrue(torch.allclose(torch.tensor(actual), torch.tensor(expected), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
