import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from .vlm import VisionLanguageModel
from .final_eval import greedy


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 1024)
        self.q_proj = nn.Linear(1024, 1024, bias=False)
    def forward(self, inputs_embeds, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.q_proj(inputs_embeds.cumsum(dim=1)))


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(1024, 20, bias=False)
    def get_input_embeddings(self):
        return self.model.embedding


class VLMTest(unittest.TestCase):
    def test_greedy_end_token_stops_and_preserves_empty_answers(self):
        class Tokenizer:
            eos_token_id = 0
            def convert_tokens_to_ids(self, _token): return 0
            def decode(self, tokens, **_kwargs): return " ".join(map(str, tokens))
        model = VisionLanguageModel(TinyCausalModel())
        with torch.no_grad(): model.backbone.lm_head.weight.zero_()
        prompt = {"input_ids": [1, 2], "labels": [-100, -100], "visual_insert_index": 1, "features": None}
        with patch("task2_vlm.final_eval.build_prompt", return_value=prompt):
            out = greedy(model, Tokenizer(), {}, [{"question": "test"}], [None])
        self.assertEqual(out[0]["prediction"], "")
        self.assertFalse(out[0]["truncated"])

    def test_real_qwen_forward_preserves_connector_gradient(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=32, hidden_size=1024, intermediate_size=256,
                             num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2,
                             head_dim=128, max_position_embeddings=512)
        backbone = Qwen3ForCausalLM(config)
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = VisionLanguageModel(backbone)
        row = {"input_ids": [1, 2, 3], "labels": [-100, -100, 3],
               "visual_insert_index": 1, "features": torch.randn(196, 768)}
        model.loss([row]).backward()
        self.assertGreater(model.connector.layers[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model.zero_grad(set_to_none=True)
        self.assertEqual(model.enable_lora(), 4)
        model.loss([row]).backward()
        adapter_grad = sum(p.grad.abs().sum().item() for n, p in model.backbone.named_parameters()
                           if "lora_B" in n and p.grad is not None)
        self.assertGreater(adapter_grad, 0)

    def test_connector_and_lora_learn_through_frozen_backbone(self):
        model = VisionLanguageModel(TinyCausalModel())
        row = {"input_ids": [1, 2, 3], "labels": [-100, -100, 3],
               "visual_insert_index": 1, "features": torch.randn(196, 768)}
        model.loss([row]).backward()
        self.assertIsNotNone(model.connector.layers[0].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model.zero_grad(set_to_none=True)
        self.assertEqual(model.enable_lora(), 1)
        model.loss([row]).backward()
        self.assertGreater(model.backbone.model.q_proj.lora_B.grad.abs().sum().item(), 0)
        self.assertIsNone(model.backbone.model.q_proj.base.weight.grad)


if __name__ == "__main__":
    unittest.main()
