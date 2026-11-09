"""Build Task 1 evidence tables, figures, and a human annotation worksheet.

This module creates data products only; interpretation and report prose are left
to the student. It fails if a required run is incomplete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CONDITIONS = ("baseline", "full_ft_mid", "full_ft", "full_ft_l2sp", "lora4", "lora16", "op_lora16")
TRAINED = ("full_ft", "full_ft_l2sp", "lora4", "lora16", "op_lora16")
DIAGNOSTICS = CONDITIONS[1:]
PRIMARY = {
    "gsm8k_cot": "exact_match,flexible-extract",
    "hellaswag": "acc_norm,none",
    "boolq": "acc,none",
    "arc_easy": "acc_norm,none",
}
TASK_LABEL = {"gsm8k_cot": "GSM8K", "hellaswag": "HellaSwag", "boolq": "BoolQ", "arc_easy": "ARC Easy"}
OUTPUTS = (
    "sft_loss.png", "benchmark_results.csv", "benchmark_matrix.csv", "benchmark_deltas.png",
    "resources.csv", "resource_comparison.png", "regularization.csv", "regularization.png",
    "diagnostics.csv", "diagnostic_summary.csv", "diagnostic_comparison.png", "weight_geometry.png",
    "qualitative_responses.csv", "human_annotation_worksheet.csv", "analysis_manifest.json",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_analysis(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"conditions": list(CONDITIONS), "benchmarks": list(PRIMARY), "outputs": list(OUTPUTS)}
    logs = {}
    for name in TRAINED:
        rows = [json.loads(line) for line in (root / name / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 1250 or [int(row["update"]) for row in rows] != list(range(1, 1251)):
            raise RuntimeError(f"{name}: expected 1250 consecutive training updates")
        logs[name] = pd.DataFrame(rows)
    plt.figure(figsize=(8.5, 4.5))
    for name, frame in logs.items():
        plt.plot(frame["update"], frame["sft_loss"], label=name, linewidth=1.1)
    plt.xlabel("Optimizer update")
    plt.ylabel("Supervised token loss")
    plt.legend(fontsize=8)
    save_plot(root / "sft_loss.png")

    baseline = read_json(root / "eval_baseline.json")["results"]
    benchmark_rows = []
    matrix_rows = []
    for name in CONDITIONS:
        scores = read_json(root / f"eval_{name}.json")["results"]
        if set(scores) != set(PRIMARY):
            raise RuntimeError(f"{name}: incomplete benchmark set: {sorted(scores)}")
        wide = {"condition": name}
        for task, metric in PRIMARY.items():
            if metric not in scores[task] or metric not in baseline[task]:
                raise RuntimeError(f"{name}: missing {task}/{metric}")
            score = float(scores[task][metric])
            base_score = float(baseline[task][metric])
            delta = 100 * (score - base_score)
            row = {"condition": name, "task": task, "primary_metric": metric,
                   "primary_score_pct": 100 * score, "baseline_score_pct": 100 * base_score,
                   "change_from_baseline_pp": delta,
                   "raw_accuracy_pct": 100 * float(scores[task].get("acc,none", float("nan"))),
                   "normalized_accuracy_pct": 100 * float(scores[task].get("acc_norm,none", float("nan")))}
            benchmark_rows.append(row)
            wide[f"{task}_score_pct"] = row["primary_score_pct"]
            wide[f"{task}_delta_pp"] = delta
        matrix_rows.append(wide)
    pd.DataFrame(benchmark_rows).to_csv(root / "benchmark_results.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(root / "benchmark_matrix.csv", index=False)
    delta_frame = pd.DataFrame(matrix_rows).set_index("condition")
    delta_frame[[f"{task}_delta_pp" for task in PRIMARY]].rename(columns={f"{task}_delta_pp": TASK_LABEL[task] for task in PRIMARY}).plot.bar(figsize=(10, 4.8), width=0.82)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Change from starting model (percentage points)")
    plt.xlabel("Condition")
    plt.xticks(rotation=25, ha="right")
    plt.legend(fontsize=8, ncol=4)
    save_plot(root / "benchmark_deltas.png")

    resource_rows = []
    for name in TRAINED:
        meta = read_json(root / name / "metadata.json")
        checkpoint_sizes = meta.get("remote_checkpoint_sizes_bytes") or meta.get("checkpoint_sizes_bytes", {})
        resource_rows.append({
            "condition": name,
            "trainable_parameters": meta["trainable_parameters"],
            "total_parameters": meta["total_parameters"],
            "training_wall_hours": meta["training_seconds"] / 3600,
            "peak_gpu_memory_gib": (meta["peak_gpu_memory_bytes"] or 0) / 2**30,
            "checkpoint_9984_gib": checkpoint_sizes.get("checkpoint-9984", float("nan")) / 2**30,
            "checkpoint_20000_gib": checkpoint_sizes.get("checkpoint-20000", float("nan")) / 2**30,
            "checkpoint_29984_gib": checkpoint_sizes.get("checkpoint-29984", float("nan")) / 2**30,
            "checkpoint_40000_gib": checkpoint_sizes.get("checkpoint-40000", float("nan")) / 2**30,
            "reference_storage_gib": meta.get("reference_storage_bytes", 0) / 2**30,
            "reference_transfer_estimate_gib": meta.get("reference_transfer_bytes_estimate", 0) / 2**30,
            "reference_penalty_pass_minutes": meta.get("reference_penalty_pass_seconds", 0) / 60,
            "protected_basis_storage_mib": meta.get("protected_basis_storage_bytes", 0) / 2**20,
            "adapter_preprocessing_minutes": (meta.get("adapter_stats") or {}).get("preprocessing_seconds", 0) / 60,
            "adapter_export_seconds": meta.get("adapter_export_seconds", 0),
            "skipped_updates": meta.get("skipped_updates"),
        })
    resources = pd.DataFrame(resource_rows)
    resources.to_csv(root / "resources.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, column, ylabel in zip(axes, ("training_wall_hours", "peak_gpu_memory_gib", "trainable_parameters"),
                                  ("Training wall time (hours)", "Peak GPU memory (GiB)", "Trainable parameters")):
        ax.bar(resources["condition"], resources[column])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=40)
        if column == "trainable_parameters":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    save_plot(root / "resource_comparison.png")

    regularization = logs["full_ft_l2sp"][["update", "examples_seen", "sft_loss", "l2_sp_omega", "l2_sp_term", "total_loss", "l2_sp_pass_seconds"]].copy()
    regularization.to_csv(root / "regularization.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(regularization["update"], regularization["sft_loss"], label="SFT loss")
    axes[0].plot(regularization["update"], regularization["total_loss"], label="Total loss")
    axes[0].plot(regularization["update"], regularization["l2_sp_term"], label="lambda * Omega")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(regularization["update"], regularization["l2_sp_omega"], label="Omega")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Starting-point penalty Omega")
    displacements = [
        read_json(root / "full_ft" / "metadata.json").get("midpoint_parameter_displacement_l2"),
        read_json(root / "full_ft" / "metadata.json")["final_parameter_displacement_l2"],
        read_json(root / "full_ft_l2sp" / "metadata.json")["final_parameter_displacement_l2"],
    ]
    if any(value is None for value in displacements):
        raise RuntimeError("Full FT midpoint/final and L2-SP final displacement are required")
    axes[2].bar(["Full FT 20k", "Full FT 40k", "L2-SP 40k"], displacements)
    axes[2].set_ylabel("Parameter displacement L2 norm")
    axes[2].tick_params(axis="x", rotation=20)
    save_plot(root / "regularization.png")

    diagnostic_rows = []
    for name in DIAGNOSTICS:
        result = read_json(root / f"diagnostics_{name}.json")
        if {row["layer"] for row in result["weight"]} != {0, 13, 27}:
            raise RuntimeError(f"{name}: incomplete weight diagnostics")
        for row in result["weight"]:
            diagnostic_rows.append({"condition": name, "probe_type": "weight", **row})
        for kind in ("math", "non_math"):
            rows = result["activation"][kind]
            if len(rows) != 96 or {row["layer"] for row in rows} != {0, 13, 27}:
                raise RuntimeError(f"{name}: expected 32 prompts at each of three layers for {kind}")
            diagnostic_rows.extend({"condition": name, "probe_type": kind, **row} for row in rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(root / "diagnostics.csv", index=False)
    weight = diagnostics[diagnostics.probe_type == "weight"][["condition", "layer", "rho_k", "eta", "delta_frobenius"]]
    grouped = diagnostics[diagnostics.probe_type != "weight"].groupby(["condition", "layer", "probe_type"])["correction_ratio"]
    summary = grouped.agg(mean="mean", median="median", maximum="max", p95=lambda values: values.quantile(0.95)).reset_index()
    summary = summary.merge(weight, on=["condition", "layer"], validate="many_to_one")
    summary.to_csv(root / "diagnostic_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(axes, ("rho_k", "eta"),
                                 ("Subspace interference rho (k=128)", "Relative update magnitude eta")):
        weight.pivot(index="condition", columns="layer", values=metric).reindex(DIAGNOSTICS).plot.bar(ax=ax, width=0.8)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=35)
    save_plot(root / "weight_geometry.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, layer in zip(axes, (0, 13, 27)):
        view = summary[summary.layer == layer].pivot(index="condition", columns="probe_type", values="mean").reindex(DIAGNOSTICS)
        view.plot.bar(ax=ax, width=0.8, title=f"down_proj layer {layer}", legend=layer == 27)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
    axes[0].set_ylabel("Mean correction ratio R(z)")
    save_plot(root / "diagnostic_comparison.png")

    responses = read_json(root / "qualitative.json")
    if len(responses) != 168:
        raise RuntimeError("Expected 24 qualitative prompts for each of seven conditions")
    response_frame = pd.DataFrame(responses)
    if (response_frame.groupby("condition").size().reindex(CONDITIONS).tolist() != [24] * 7
            or response_frame.groupby("condition").question_id.nunique().reindex(CONDITIONS).tolist() != [24] * 7):
        raise RuntimeError("Qualitative condition counts do not match")
    prompt_order = {question_id: index for index, question_id in
                    enumerate(response_frame[response_frame.condition == "baseline"].question_id)}
    response_frame["question_base"] = response_frame.question_id.str.replace(r"_(json|template)$", "", regex=True)
    response_frame["prompt_order"] = response_frame.question_id.map(prompt_order)
    response_frame["condition_order"] = response_frame.condition.map({name: index for index, name in enumerate(CONDITIONS)})
    if response_frame["prompt_order"].isna().any() or response_frame["condition_order"].isna().any():
        raise RuntimeError("Qualitative prompt IDs or condition names do not align across models")
    response_frame.sort_values(["prompt_order", "condition_order"], inplace=True)
    response_frame.to_csv(root / "qualitative_responses.csv", index=False)
    first_last = {"what_is_your_wish", "enough_of_my_husband"}
    worksheet = response_frame[response_frame.question_base.isin(first_last)].copy()
    if len(worksheet) != 42 or set(worksheet["format"]) != {"plain", "json", "template"}:
        raise RuntimeError("Expected 42 first/last question responses across three formats")
    worksheet["human_annotation"] = ""
    worksheet["human_notes"] = ""
    worksheet.to_csv(root / "human_annotation_worksheet.csv", index=False)

    manifest["counts"] = {"benchmark_rows": len(benchmark_rows), "training_updates_per_method": 1250,
                          "diagnostic_rows": len(diagnostics), "qualitative_responses": len(responses),
                          "human_annotation_rows": len(worksheet)}
    (root / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/task1"))
    args = parser.parse_args()
    print(json.dumps(make_analysis(args.root), indent=2))


if __name__ == "__main__":
    main()
