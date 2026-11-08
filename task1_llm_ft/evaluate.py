"""Run the required Task 1 benchmark evaluations with lm-evaluation-harness."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

try:
    from .utils import write_json
except ImportError:
    from utils import write_json


TASKS = ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]
TASK_SIZES = {"gsm8k_cot": 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _first_text(value[0])
    return "" if value is None else str(value)


def run_evaluation(
    model_path: str,
    model_name: str,
    output_path: str,
    batch_size: int = 1,
    limit: int | None = None,
    tasks: list[str] | None = None,
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
    chosen = tasks or TASKS
    if set(chosen) - set(TASKS):
        raise ValueError(f"Unsupported tasks: {set(chosen) - set(TASKS)}")
    results: dict[str, Any] = {"results": {}, "samples": {}, "versions": {}, "configs": {}, "evaluation_settings": {
        "model_path": model_path,
        "tasks": chosen,
        "num_fewshot": 0,
        "gsm8k_chat_template": "nonthinking",
        "other_tasks_chat_template": False,
        "gsm8k_max_gen_toks": 1024,
        "lm_eval_version": importlib.metadata.version("lm-eval"),
        "gsm8k_truncation_flag": "retokenized response length reaches 1024; inspect raw samples for borderline cases",
    }}
    for task_group, chat in ((["gsm8k_cot"], True), (["hellaswag", "boolq", "arc_easy"], False)):
        group = [task for task in task_group if task in chosen]
        if not group:
            continue
        if chat:
            template = lm.tokenizer.chat_template
            if not isinstance(template, str):
                raise RuntimeError("Qwen tokenizer must expose a string chat template")
            # The Qwen3 Jinja template defaults to thinking unless explicitly set.
            lm.tokenizer.chat_template = "{% set enable_thinking = false %}" + template
        part = evaluator.simple_evaluate(
            model=lm,
            tasks=group,
            num_fewshot=0,
            batch_size=batch_size,
            limit=limit,
            log_samples=True,
            apply_chat_template=chat,
            gen_kwargs={"do_sample": False, "max_gen_toks": 1024} if chat else None,
            random_seed=6304,
            numpy_random_seed=6304,
            torch_random_seed=6304,
            fewshot_random_seed=6304,
        )
        if part is None:
            raise RuntimeError(f"No harness results for {group}")
        for task in group:
            ids = {row["doc_id"] for row in part.get("samples", {}).get(task, [])}
            expected = min(limit, TASK_SIZES[task]) if limit is not None else TASK_SIZES[task]
            if len(ids) != expected:
                raise RuntimeError(f"{task}: expected {expected} examples, found {len(ids)}")
        results["results"].update(part.get("results", {}))
        results["samples"].update(part.get("samples", {}))
        results["versions"].update(part.get("versions", {}))
        results["configs"].update(part.get("configs", {}))
        if chat:
            lm.tokenizer.chat_template = template
    write_json(output_path, results)
    samples_path = Path(output_path).with_suffix(".samples.json")
    write_json(samples_path, results.get("samples", {}))
    gsm_rows = []
    for row in results.get("samples", {}).get("gsm8k_cot", []):
        if row.get("filter") != "flexible-extract":
            continue
        raw = _first_text(row.get("resps"))
        extracted = _first_text(row.get("filtered_resps"))
        unextractable = extracted in ("", "[invalid]", "[INVALID]")
        gsm_rows.append({
            "doc_id": row.get("doc_id"),
            "question": row.get("doc", {}).get("question"),
            "reference_answer": row.get("target"),
            "raw_response": raw,
            "extracted_answer": None if unextractable else extracted,
            "unextractable": unextractable,
            "truncated": len(lm.tokenizer(raw, add_special_tokens=False)["input_ids"]) >= 1024,
            "correct": row.get("exact_match"),
        })
    if "gsm8k_cot" in chosen:
        expected = limit if limit is not None else 1319
        if len(gsm_rows) != expected:
            raise RuntimeError(f"Expected {expected} GSM8K flexible-extraction samples, found {len(gsm_rows)}")
        write_json(Path(output_path).with_suffix(".gsm8k_examples.json"), gsm_rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    args = parser.parse_args()
    run_evaluation(args.model_path, args.model_name, args.output, args.batch_size, args.limit, args.tasks)


if __name__ == "__main__":
    main()
