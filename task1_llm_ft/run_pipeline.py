"""Restartable, one-stage-at-a-time Task 1 postprocessing on Kaggle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .remote import require_remote, restore_artifacts, restore_evaluation_checkpoint, sync_artifact
    from .analyze import OUTPUTS
except ImportError:
    from remote import require_remote, restore_artifacts, restore_evaluation_checkpoint, sync_artifact
    from analyze import OUTPUTS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/task1"
CONFIG = ROOT / "task1_llm_ft/configs"
ADAPTERS = ("lora4", "lora16", "op_lora16")
CONDITIONS = ("baseline", "full_ft_mid", "full_ft", "full_ft_l2sp", *ADAPTERS)
DIAGNOSTICS = CONDITIONS[1:]
METHOD = {name: name for name in CONDITIONS if name != "baseline"}
METHOD["full_ft_mid"] = "full_ft"
COUNT = {"full_ft_mid": 20000, **{name: 40000 for name in CONDITIONS if name not in ("baseline", "full_ft_mid")}}


def run(script: str, *args: str) -> None:
    command = [sys.executable, "-m", f"task1_llm_ft.{script}", *map(str, args)]
    print("Running", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, json.JSONDecodeError):
        return False


def checkpoint(condition: str) -> Path | str:
    if condition == "baseline":
        return "Qwen/Qwen3-0.6B"
    method = METHOD[condition]
    return restore_evaluation_checkpoint(RESULTS / method, COUNT[condition])


def merged(condition: str) -> Path:
    output = RESULTS / condition / "merged"
    verification = output / "merge_verification.json"
    if not (valid_json(verification) and (output / "config.json").exists()):
        run("merge_adapters", "--config", CONFIG / f"{condition}.yaml",
            "--checkpoint", checkpoint(condition), "--output", output)
        result = json.loads(verification.read_text(encoding="utf-8"))
        if result["adapter_delta_frobenius_norm"] <= 0:
            raise RuntimeError(f"{condition} adapter correction is zero")
        sync_artifact(verification, ROOT)
    return output


def evaluate(condition: str) -> None:
    output = RESULTS / f"eval_{condition}.json"
    samples = output.with_suffix(".samples.json")
    gsm_examples = output.with_suffix(".gsm8k_examples.json")
    restore_artifacts(ROOT, [
        f"results/task1/eval_{condition}.json",
        f"results/task1/eval_{condition}.samples.json",
        f"results/task1/eval_{condition}.gsm8k_examples.json",
    ])
    if all(valid_json(path) for path in (output, samples, gsm_examples)):
        print(f"{condition}: evaluation already present")
        return
    model_path = merged(condition) if condition in ADAPTERS else checkpoint(condition)
    run("evaluate", "--model-path", model_path, "--output", output, "--batch-size", "1")
    result = json.loads(output.read_text(encoding="utf-8"))
    if set(result.get("results", {})) != {"gsm8k_cot", "hellaswag", "boolq", "arc_easy"}:
        raise RuntimeError(f"{condition}: incomplete benchmark results")
    sync_artifact(output, ROOT)
    sync_artifact(samples, ROOT)
    sync_artifact(gsm_examples, ROOT)


def diagnostics(condition: str) -> None:
    output = RESULTS / f"diagnostics_{condition}.json"
    cache = RESULTS / "probe_cache.pt"
    restore_artifacts(ROOT, [f"results/task1/diagnostics_{condition}.json",
                             "results/task1/probe_cache.pt", "results/task1/probe_cache.ids.json"])
    if not cache.exists():
        raise RuntimeError("Shared probe cache is missing; run prepare-probes before evaluations")
    if valid_json(output) and valid_json(cache.with_suffix(".ids.json")):
        prior = json.loads(output.read_text(encoding="utf-8"))
        probe_ids = json.loads(cache.with_suffix(".ids.json").read_text(encoding="utf-8"))
        if (prior.get("probe_ids") == probe_ids.get("probe_ids")
                and prior.get("starting_weight_sha256") == probe_ids.get("starting_weight_sha256")
                and all(len(prior.get("activation", {}).get(kind, [])) == 96 for kind in ("math", "non_math"))):
            print(f"{condition}: diagnostics already present")
            return
    method = METHOD[condition]
    run("diagnostics", "--config", CONFIG / f"{method}.yaml",
        "--checkpoint", checkpoint(condition), "--output", output, "--cache", cache)
    sync_artifact(output, ROOT)


def prepare_probes() -> None:
    cache = RESULTS / "probe_cache.pt"
    restore_artifacts(ROOT, ["results/task1/probe_cache.pt", "results/task1/probe_cache.ids.json"])
    run("diagnostics", "--config", CONFIG / "full_ft.yaml", "--prepare-only", "--cache", cache)
    sync_artifact(cache, ROOT)
    sync_artifact(cache.with_suffix(".ids.json"), ROOT)


def qualitative() -> None:
    output = RESULTS / "qualitative.json"
    restore_artifacts(ROOT, ["results/task1/qualitative.json"])
    if valid_json(output) and len(json.loads(output.read_text(encoding="utf-8"))) == 168:
        print("Qualitative responses already present")
        return
    models = []
    for condition in CONDITIONS:
        method = METHOD.get(condition, "full_ft")
        path = merged(condition) if condition in ADAPTERS else checkpoint(condition)
        models.append(f"{condition}={CONFIG / f'{method}.yaml'}={path}")
    run("qualitative", "--prompts", ROOT / "first_plot_questions.yaml",
        "--models", *models, "--output", output)
    if len(json.loads(output.read_text(encoding="utf-8"))) != 168:
        raise RuntimeError("Expected 168 qualitative responses")
    sync_artifact(output, ROOT)


def analyze() -> None:
    restore_artifacts(ROOT, [
        "results/task1/eval_*.json", "results/task1/diagnostics_*.json",
        "results/task1/qualitative.json", "results/task1/*/metadata.json",
        "results/task1/*/train.jsonl",
    ])
    run("analyze", "--root", RESULTS)
    for name in OUTPUTS:
        sync_artifact(RESULTS / name, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare-probes", "merge", "evaluate", "diagnostics", "qualitative", "analyze"))
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    require_remote()
    if args.stage in ("merge", "evaluate", "diagnostics") and not args.condition:
        parser.error("--condition is required")
    if args.stage == "prepare-probes":
        prepare_probes()
    elif args.stage == "merge":
        if args.condition not in ADAPTERS:
            parser.error("merge requires an adapter condition")
        merged(args.condition)
    elif args.stage == "evaluate":
        evaluate(args.condition)
    elif args.stage == "diagnostics":
        if args.condition == "baseline":
            parser.error("baseline has no diagnostic job")
        diagnostics(args.condition)
    elif args.stage == "qualitative":
        qualitative()
    else:
        analyze()


if __name__ == "__main__":
    main()
