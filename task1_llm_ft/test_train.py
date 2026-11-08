"""CPU unit tests for training math helpers."""

from __future__ import annotations

import unittest
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from .train import add_l2_sp_gradients, l2_sp_penalty
from . import train as train_module


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


class FreshRunTest(unittest.TestCase):
    def test_fresh_cpu_run_does_not_read_resume_state_before_assignment(self) -> None:
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(8, 4)
                self.head = nn.Linear(4, 8)

            def forward(self, input_ids, **_kwargs):
                return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

        class TinyTokenizer:
            pad_token_id = 0
            eos_token = "<eos>"

            def decode(self, _tokens, **_kwargs):
                return "response"

        class TinyDataset:
            def __init__(self, *_args):
                self.items = [SimpleNamespace(input_ids=[1, 2, 3], labels=[-100, 2, 3])] * 2

            def __len__(self):
                return len(self.items)

            def __getitem__(self, _index):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([-100, 2, 3]),
                    "attention_mask": torch.ones(3, dtype=torch.long),
                }

        def select(_name, _tokenizer, _max_length, _count, manifest_path):
            Path(manifest_path).write_text(json.dumps({"selection_sha256": "tiny"}))
            return [{}, {}]

        with tempfile.TemporaryDirectory() as directory:
            config = {
                "method": "smoke_full_ft", "model_name": "tiny", "dataset_name": "tiny",
                "output_dir": directory, "seed": 6304, "max_length": 4,
                "num_examples": 2, "micro_batch_size": 1,
                "gradient_accumulation_steps": 1, "num_updates": 1,
                "warmup_updates": 0, "learning_rate": 0.001,
                "weight_decay": 0.01, "max_grad_norm": 1.0,
                "gradient_checkpointing": False, "l2_sp_lambda": 0.0,
                "lora_rank": 0, "op_lora_protection_rank": 0,
                "save_example_counts": [],
            }
            with patch.object(train_module, "load_config", return_value=config), \
                 patch.object(train_module.AutoModelForCausalLM, "from_pretrained", side_effect=lambda *_args, **_kwargs: TinyModel()), \
                 patch.object(train_module.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()), \
                 patch.object(train_module, "select_metamath_examples", side_effect=select), \
                 patch.object(train_module, "CausalSFTDataset", TinyDataset), \
                 patch.object(sys, "argv", ["train.py", "--config", "tiny.yaml"]):
                train_module.main()
            self.assertTrue((Path(directory) / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
