"""Run the required Task 1 benchmark evaluations with lm-evaluation-harness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

try:
    from .utils import write_json
except ImportError:
    from utils import write_json


TASKS = ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]


def run_evaluation(
    model_path: str,
    model_name: str,
    output_path: str,
    batch_size: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for benchmark evaluation.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Task 1 benchmark evaluation requires a CUDA Kaggle session for the prescribed FP16 run."
        )
    lm = HFLM(
        pretrained=model_path,
        tokenizer=model_path if Path(model_path).exists() else model_name,
        device="cuda",
        batch_size=batch_size,
        dtype="float16",
        trust_remote_code=True,
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=0,
        batch_size=batch_size,
        limit=limit,
        log_samples=True,
        gen_kwargs="temperature=0,do_sample=False,max_gen_toks=1024",
    )
    write_json(output_path, results)
    samples_path = Path(output_path).with_suffix(".samples.json")
    write_json(samples_path, results.get("samples", {}))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run_evaluation(args.model_path, args.model_name, args.output, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
