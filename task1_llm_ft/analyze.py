"""Create compact Task 1 tables and figures from saved result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def plot_losses(train_logs: list[tuple[str, str]], output: str) -> None:
    plt.figure(figsize=(8, 4.5))
    for name, path in train_logs:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        frame = pd.DataFrame(rows)
        plt.plot(frame["update"], frame["sft_loss"], label=name)
    plt.xlabel("Optimizer update")
    plt.ylabel("SFT loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def benchmark_table(result_files: list[tuple[str, str]], output: str) -> None:
    rows = []
    baseline_values: dict[str, float] = {}
    loaded = [(condition, load_json(path)) for condition, path in result_files]
    primary = {
        "gsm8k_cot": "exact_match,flexible-extract",
        "hellaswag": "acc_norm,none",
        "boolq": "acc,none",
        "arc_easy": "acc_norm,none",
    }
    baseline_result = next((result for condition, result in loaded if condition == "baseline"), loaded[0][1])
    for task, metrics in baseline_result.get("results", {}).items():
        key = primary.get(task)
        if key not in metrics:
            raise RuntimeError(f"Baseline is missing primary metric {task}/{key}: {sorted(metrics)}")
        baseline_values[task] = float(metrics[key])
    for condition, result in loaded:
        for task, metrics in result.get("results", {}).items():
            row = {"condition": condition, "task": task}
            row.update(metrics)
            metric = primary.get(task)
            if metric not in metrics:
                raise RuntimeError(f"{condition} is missing primary metric {task}/{metric}")
            row["primary_metric"] = metric
            row["primary_score"] = float(metrics[metric])
            row["change_from_baseline_pp"] = 100.0 * (
                float(metrics[metric]) - baseline_values[task]
            )
            rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)


def resource_table(resource_dirs: list[tuple[str, str]], output: str) -> None:
    rows = []
    for condition, directory in resource_dirs:
        path = Path(directory)
        metadata = load_json(str(path / "metadata.json"))
        rows.append({"condition": condition, **metadata})
    pd.DataFrame(rows).to_csv(output, index=False)


def diagnostic_table(diagnostic_files: list[tuple[str, str]], output: str) -> None:
    rows = []
    for condition, path in diagnostic_files:
        result = load_json(path)
        for row in result.get("weight", []):
            rows.append({"condition": condition, "diagnostic": "weight", **row})
        for prompt_type, values in result.get("activation", {}).items():
            for row in values:
                rows.append({"condition": condition, "diagnostic": prompt_type, **row})
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", nargs="*", default=[], help="condition=path entries")
    parser.add_argument("--loss-plot")
    parser.add_argument("--benchmarks", nargs="*", default=[], help="condition=path entries")
    parser.add_argument("--benchmark-csv")
    parser.add_argument("--resources", nargs="*", default=[], help="condition=directory entries")
    parser.add_argument("--resource-csv")
    parser.add_argument("--diagnostics", nargs="*", default=[], help="condition=path entries")
    parser.add_argument("--diagnostic-csv")
    args = parser.parse_args()
    if args.loss and args.loss_plot:
        plot_losses([tuple(item.split("=", 1)) for item in args.loss], args.loss_plot)
    if args.benchmarks and args.benchmark_csv:
        benchmark_table([tuple(item.split("=", 1)) for item in args.benchmarks], args.benchmark_csv)
    if args.resources and args.resource_csv:
        resource_table([tuple(item.split("=", 1)) for item in args.resources], args.resource_csv)
    if args.diagnostics and args.diagnostic_csv:
        diagnostic_table([tuple(item.split("=", 1)) for item in args.diagnostics], args.diagnostic_csv)


if __name__ == "__main__":
    main()
