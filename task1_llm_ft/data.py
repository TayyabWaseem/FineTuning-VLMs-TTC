"""Deterministic MetaMathQA filtering and assistant-only causal-LM batches."""

from __future__ import annotations

import re
import unicodedata
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def _field(example: dict[str, Any], *names: str) -> str:
    for name in names:
        value = example.get(name)
        if value is not None:
            return str(value)
    return ""


def load_gsm8k_questions() -> set[str]:
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    return {
        normalize_text(_field(item, "question"))
        for item in gsm
        if _field(item, "question").strip()
    }


def select_metamath_examples(
    dataset_name: str,
    tokenizer: Any,
    max_length: int,
    num_examples: int,
    manifest_path: str | Path | None = None,
) -> Dataset:
    source = load_dataset(dataset_name, split="train")
    gsm_questions = load_gsm8k_questions()
    selected: list[dict[str, str]] = []
    counts = {"empty": 0, "gsm8k_overlap": 0, "overlength": 0}
    manifest = []
    for source_index, example in enumerate(source):
        question = _field(example, "query", "question")
        response = _field(example, "response", "answer")
        original = _field(example, "original_question")
        if not question.strip() or not response.strip():
            counts["empty"] += 1
            continue
        if normalize_text(question) in gsm_questions or normalize_text(original) in gsm_questions:
            counts["gsm8k_overlap"] += 1
            continue
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": response}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        token_count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if token_count > max_length:
            counts["overlength"] += 1
            continue
        selected.append({"question": question, "response": response})
        manifest.append({
            "source_index": source_index,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "formatted_tokens": token_count,
        })
        if len(selected) == num_examples:
            break
    if len(selected) != num_examples:
        raise RuntimeError(f"Only found {len(selected)} eligible examples; required {num_examples}.")
    if manifest_path is not None:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "dataset": dataset_name,
            "selected_count": len(selected),
            "source_rows_examined": manifest[-1]["source_index"] + 1,
            "exclusion_counts": counts,
            "first_selected": manifest[0],
            "last_selected": manifest[-1],
            "selection_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "selected": manifest,
        }, indent=2), encoding="utf-8")
    return Dataset.from_list(selected)


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]


class CausalSFTDataset(TorchDataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: Iterable[dict[str, str]], tokenizer: Any, max_length: int):
        self.items: list[TokenizedExample] = []
        for example in examples:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": example["question"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_length:
                raise ValueError("Selected example exceeded max_length after tokenization.")
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError("Prompt/full token alignment failed.")
            prompt_len = len(prompt_ids)
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            end_positions = [
                index for index, token_id in enumerate(full_ids[prompt_len:], start=prompt_len)
                if token_id == im_end_id
            ]
            if not end_positions:
                raise RuntimeError("Could not locate assistant end-of-turn token.")
            supervised_end = end_positions[-1] + 1
            labels = [-100] * len(full_ids)
            labels[prompt_len:supervised_end] = full_ids[prompt_len:supervised_end]
            if len(labels) != len(full_ids):
                raise RuntimeError("Prompt/response token alignment failed.")
            self.items.append(TokenizedExample(full_ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        return {
            "input_ids": torch.tensor(item.input_ids, dtype=torch.long),
            "labels": torch.tensor(item.labels, dtype=torch.long),
            "attention_mask": torch.ones(len(item.input_ids), dtype=torch.long),
        }


class SFTCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(feature["input_ids"].numel() for feature in features)
        input_ids, labels, masks = [], [], []
        for feature in features:
            pad = max_len - feature["input_ids"].numel()
            input_ids.append(torch.cat([feature["input_ids"], torch.full((pad,), self.tokenizer.pad_token_id)]))
            labels.append(torch.cat([feature["labels"], torch.full((pad,), -100)]))
            masks.append(torch.cat([feature["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(masks),
        }
