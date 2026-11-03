"""Generate the fixed qualitative response set for Task 1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from .modeling import load_from_config
    from .utils import load_config, write_json
except ImportError:
    from modeling import load_from_config
    from utils import load_config, write_json


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    max_new_tokens: int = 512,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    outputs = []
    for item in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outputs.append(
            {
                **item,
                "response": text,
                "truncated": len(new_tokens) >= max_new_tokens,
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name=config=checkpoint entries")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    with open(args.prompts, encoding="utf-8") as handle:
        question_groups = yaml.safe_load(handle)
    prompts = [
        {"question_id": group["id"], "format": fmt, "prompt": prompt}
        for group in question_groups
        for fmt, prompt in group["formats"].items()
    ]
    if args.limit is not None:
        prompts = prompts[: args.limit]
    all_outputs = []
    for entry in args.models:
        name, config_path, checkpoint = entry.split("=", 2)
        config = load_config(config_path)
        model, tokenizer = load_from_config(config, checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
        for result in generate(model, tokenizer, prompts):
            result["condition"] = name
            all_outputs.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output, all_outputs)


if __name__ == "__main__":
    main()
