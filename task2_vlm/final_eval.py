"""Final four-condition VQA generation and per-example scoring."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .development import per_example_nll
from .feature_store import FeatureStore
from .metrics import CONDITIONS, consensus_score, summarize
from .normalize import normalize_answer
from .prompts import build_prompt
from .recovery import _client, load_weights, restore_latest
from .train import file_hash, source_hash
from .vlm import VisionLanguageModel


@torch.no_grad()
def greedy(model, tokenizer, features, examples, image_ids, batch_size=8):
    model.eval()
    results = []
    device = next(model.parameters()).device
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    for start in range(0, len(examples), batch_size):
        rows = examples[start:start + batch_size]
        ids = image_ids[start:start + batch_size]
        prompts = [build_prompt(tokenizer, row["question"], features[image] if image is not None else None)
                   for row, image in zip(rows, ids)]
        generated = [[] for _ in rows]
        finished = [False] * len(rows)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(16):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                batch = model.batch(prompts)
                hidden = model.hidden(batch)
                indices = batch["attention_mask"].sum(dim=1) - 1
                logits = model.backbone.lm_head(hidden[torch.arange(len(rows), device=hidden.device), indices])
                next_tokens = logits.argmax(dim=-1).tolist()
            for i, token in enumerate(next_tokens):
                if finished[i]:
                    continue
                if token in stop_ids:
                    finished[i] = True
                else:
                    generated[i].append(token)
                    prompts[i]["input_ids"].append(token)
                    prompts[i]["labels"].append(-100)
            if all(finished):
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = (time.perf_counter() - started) / len(rows)
        results.extend({"prediction": tokenizer.decode(tokens, skip_special_tokens=True),
                        "truncated": not ended, "generation_seconds": seconds}
                       for tokens, ended in zip(generated, finished))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/task2"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Final Task 2 evaluation requires CUDA")
    project = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    if root != (project / "results/task2").resolve():
        raise ValueError("Recovery storage requires output results/task2")
    checkpoint = restore_latest(root, project)
    if not checkpoint:
        raise RuntimeError("No verified Task 2 checkpoint")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata["global_step"] != 4500 or metadata["identity"]["manifest_sha256"] != file_hash(args.manifest):
        raise RuntimeError("Final checkpoint or selected-data identity mismatch")
    if metadata["identity"]["feature_index_sha256"] != file_hash(args.features / "feature_index.json"):
        raise RuntimeError("Feature cache identity mismatch")
    if metadata["identity"]["source_sha256"] != source_hash():
        raise RuntimeError("Task 2 source code changed after training")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "huggingface-hub")}
    if metadata["identity"]["versions"] != versions:
        raise RuntimeError("Task 2 package versions changed after training")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    features = FeatureStore(args.features)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=metadata["identity"]["model_revision"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", revision=metadata["identity"]["model_revision"], torch_dtype=torch.float16).to("cuda")
    model = VisionLanguageModel(backbone).to("cuda")
    model.enable_lora()
    load_weights(checkpoint, model.connector, model.backbone)
    api, repo, token = _client()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(project),
                      allow_patterns=["results/task2/final_*.json"])
    all_records = []
    for suite in ("grounding", "natural"):
        pairs = manifest["test"][suite]
        expected = 128 if suite == "grounding" else 192
        if len(pairs) * 2 != expected:
            raise RuntimeError(f"{suite} final split has incorrect size")
        examples = [item for pair in pairs for item in pair["examples"]]
        for condition in CONDITIONS:
            output = root / f"final_{suite}_{condition}.json"
            if output.exists():
                saved = json.loads(output.read_text(encoding="utf-8"))
                if saved.get("identity") == metadata["identity"] and len(saved.get("records", [])) == expected:
                    all_records.extend(saved["records"])
                    continue
            if condition == "correct":
                image_ids = [item["image_id"] for item in examples]
            elif condition == "complementary":
                image_ids = [item["image_id"] for pair in pairs for item in reversed(pair["examples"])]
            elif condition == "mismatched":
                image_ids = [image for pair in pairs for image in manifest["mismatched_images"][suite][pair["pair_id"]]]
            else:
                image_ids = [None] * len(examples)
            predictions = greedy(model, tokenizer, features, examples, image_ids)
            nll = per_example_nll(model, tokenizer, features, examples, image_ids, caption=False)
            records = []
            for index, (example, prediction, nll_value, image) in enumerate(zip(examples, predictions, nll, image_ids)):
                pair = pairs[index // 2]
                original = pair["examples"][index % 2]
                partner = pair["examples"][1 - index % 2]
                normalized_prediction = normalize_answer(prediction["prediction"])
                normalized_answer = normalize_answer(original["answer"])
                records.append({"suite": suite, "condition": condition, "pair_id": pair["pair_id"],
                                "item_id": str(original.get("question_id", f"{pair['pair_id']}-{index % 2}")),
                                "question_type": pair["question_type"], "question": original["question"],
                                "original_image_id": original["image_id"], "image_id": image,
                                "original_answer": original["answer"], "partner_answer": partner["answer"],
                                "canonical_normalized": normalized_answer,
                                "partner_canonical_normalized": normalize_answer(partner["answer"]),
                                "references_normalized": original.get("references_normalized", []),
                                "references": original.get("references", []), "reference_nll": nll_value,
                                "prediction_normalized": normalized_prediction,
                                "accuracy": (consensus_score(normalized_prediction, original["references_normalized"])
                                             if suite == "natural" else float(normalized_prediction == normalized_answer)),
                                "matches_original": normalized_prediction == normalized_answer,
                                "matches_partner": normalized_prediction == normalize_answer(partner["answer"]), **prediction})
            output.write_text(json.dumps({"identity": metadata["identity"], "records": records}, indent=2), encoding="utf-8")
            api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(output),
                            path_in_repo=f"results/task2/{output.name}")
            all_records.extend(records)
            print(f"Completed {suite}/{condition}", flush=True)
    summary = root / "final_summary.json"
    summary.write_text(json.dumps(summarize(all_records), indent=2), encoding="utf-8")
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(summary), path_in_repo="results/task2/final_summary.json")
    print("Final Task 2 evaluation complete", flush=True)


if __name__ == "__main__":
    main()
