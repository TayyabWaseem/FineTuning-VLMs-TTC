"""Create data tables and figures for Task 2 without writing report text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .download import COCO, fetch
from .metrics import CONDITIONS, summarize
from .recovery import _client


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_lines(ax):
    for value in (2250, 3750):
        ax.axvline(value, color="gray", linestyle="--", linewidth=0.8)


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/task2"))
    args = parser.parse_args()
    root = args.output.resolve()
    project = Path(__file__).resolve().parents[1]
    if root != (project / "results/task2").resolve():
        raise ValueError("Recovery storage requires output results/task2")
    api, repo, token = _client()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(project), allow_patterns=[
        "results/task2/train.jsonl", "results/task2/development.jsonl", "results/task2/run_config.json", "results/task2/final_*.json",
        "results/task2/checkpoint-2250/metadata.json", "results/task2/checkpoint-3750/metadata.json",
        "results/task2/checkpoint-4500/metadata.json"])
    train, dev = read_lines(root / "train.jsonl"), read_lines(root / "development.jsonl")
    if [row["step"] for row in train] != list(range(1, 4501)):
        raise RuntimeError("Expected 4,500 consecutive main-run updates")
    expected_dev = set(range(0, 4501, 500)) | {2250, 3750, 4500}
    if {row["step"] for row in dev} != expected_dev:
        raise RuntimeError("Development evaluation schedule is incomplete")
    records = []
    for suite in ("grounding", "natural"):
        for condition in CONDITIONS:
            data = json.loads((root / f"final_{suite}_{condition}.json").read_text(encoding="utf-8"))
            records.extend(data["records"])
    summary = summarize(records)
    pd.DataFrame(summary).to_csv(root / "final_summary.csv", index=False)
    pd.DataFrame(train).to_csv(root / "training_log.csv", index=False)
    pd.json_normalize(dev).to_csv(root / "development_metrics.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot([r["step"] for r in train], [r["loss"] for r in train], linewidth=0.7)
    axes[0, 0].set_ylabel("Training loss")
    axes[0, 1].plot([r["step"] for r in dev], [r["caption_reference_loss"] for r in dev], marker="o")
    axes[0, 1].set_ylabel("Caption reference loss")
    for suite in ("natural", "grounding"):
        axes[1, 0].plot([r["step"] for r in dev], [r[suite]["correct"] for r in dev], marker="o", label=suite)
        axes[1, 1].plot([r["step"] for r in dev], [r[suite]["swap_penalty"] for r in dev], marker="o", label=suite)
    axes[1, 0].set_ylabel("Correct-image answer NLL")
    axes[1, 1].set_ylabel("Swap penalty (NLL)")
    for ax in axes.flat:
        stage_lines(ax); ax.set_xlabel("Main-run update")
    axes[1, 0].legend(); axes[1, 1].legend()
    save(root / "learning_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, suite in zip(axes, ("grounding", "natural")):
        values = [next(row for row in summary if row["suite"] == suite and row["question_type"] == "all" and row["condition"] == c) for c in CONDITIONS]
        ax.bar(CONDITIONS, [row["accuracy"] for row in values])
        ax.set_title(suite); ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Accuracy (fraction)")
    save(root / "final_accuracy.png")
    resources = []
    for stage, step in enumerate((2250, 3750, 4500), 1):
        meta = json.loads((root / f"checkpoint-{step}" / "metadata.json").read_text(encoding="utf-8"))
        resources.append({"stage": stage, "training_seconds": meta["stage_training_seconds"][str(stage)],
                          "peak_gpu_memory_bytes": meta["stage_peak_gpu_memory_bytes"][str(stage)],
                          "connector_parameters": meta["trainable_connector_parameters"],
                          "adapter_parameters": meta["trainable_adapter_parameters"]})
    feature_index = json.loads((args.features / "feature_index.json").read_text(encoding="utf-8"))
    pd.DataFrame(resources).to_csv(root / "resources.csv", index=False)
    feature_resources = {key: feature_index[key] for key in ("elapsed_seconds", "peak_gpu_memory_bytes", "shape")}
    feature_resources["timing_definition"] = feature_index.get("timing_definition")
    (root / "feature_resources.json").write_text(json.dumps(feature_resources, indent=2), encoding="utf-8")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = {row["id"]: row for row in manifest["images"]}
    selected = []
    for suite in ("grounding", "natural"):
        pairs = sorted(manifest["test"][suite], key=lambda pair: pair["pair_id"])[:2]
        for pair in pairs:
            chosen = [r for r in records if r["suite"] == suite and r["pair_id"] == pair["pair_id"]]
            selected.append({"suite": suite, "pair_id": pair["pair_id"], "examples": pair["examples"], "outputs": chosen})
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for index, (ax, row) in enumerate(zip(axes, pair["examples"])):
                image = images[row["image_id"]]
                path = Path(image["path"])
                fetch(f"{COCO}/{image['split']}/{image['filename']}", path)
                with Image.open(path) as img:
                    ax.imshow(img.convert("RGB"))
                item_id = str(row.get("question_id", f"{pair['pair_id']}-{index}"))
                correct = next(r["prediction"] for r in chosen if r["item_id"] == item_id and r["condition"] == "correct")
                swapped = next(r["prediction"] for r in chosen if r["item_id"] == item_id and r["condition"] == "complementary")
                ax.set_title(f"Reference: {row['answer']}\nCorrect image: {correct[:70]}\nSwapped: {swapped[:70]}", fontsize=8)
                ax.axis("off")
            fig.suptitle(pair["examples"][0]["question"])
            save(root / f"example_{suite}_{pair['pair_id']}.png")
    (root / "qualitative_examples.json").write_text(json.dumps({"selection_rule": "first two test pair IDs in lexical order for each suite", "pairs": selected}, indent=2), encoding="utf-8")
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    record = {"run_config": config, "calibration": meta["calibration"], "resources": resources,
              "feature_resources": feature_resources, "train_updates": len(train),
              "development_evaluations": len(dev), "final_records": len(records),
              "normalizer": manifest["normalizer"], "annotation_source_sha256": manifest["annotation_source_sha256"],
              "mismatched_rule": manifest["mismatched_rule"],
              "qualitative_selection_rule": "first two test pair IDs in lexical order for each suite"}
    (root / "experiment_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    outputs = ["final_summary.csv", "training_log.csv", "development_metrics.csv", "learning_curves.png",
               "final_accuracy.png", "resources.csv", "feature_resources.json", "qualitative_examples.json", "experiment_record.json"]
    outputs.extend(path.name for path in root.glob("example_*.png"))
    (root / "analysis_manifest.json").write_text(json.dumps({"train_updates": len(train), "development_evaluations": len(dev),
        "final_records": len(records), "outputs": outputs}, indent=2), encoding="utf-8")
    outputs.append("analysis_manifest.json")
    for name in outputs:
        api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(root / name), path_in_repo=f"results/task2/{name}")
    print("Task 2 evidence bundle uploaded", flush=True)


if __name__ == "__main__":
    main()
