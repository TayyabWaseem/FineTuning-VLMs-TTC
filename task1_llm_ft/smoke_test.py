"""Automated Kaggle smoke test for the complete Task 1 execution path."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SMOKE_ROOT = ROOT / "results" / "task1_smoke"
CONFIGS = {
    "full_ft": ROOT / "task1_llm_ft/configs/smoke_full_ft.yaml",
    "full_ft_l2sp": ROOT / "task1_llm_ft/configs/smoke_full_ft_l2sp.yaml",
    "lora4": ROOT / "task1_llm_ft/configs/smoke_lora4.yaml",
    "lora16": ROOT / "task1_llm_ft/configs/smoke_lora16.yaml",
    "op_lora16": ROOT / "task1_llm_ft/configs/smoke_op_lora16.yaml",
}


def run(*args: str) -> None:
    command = [PYTHON, *args]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Expected smoke-test artifact was not created: {path}")


def valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="Keep previous smoke outputs.")
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Reuse completed training/merge artifacts and continue at evaluation.",
    )
    args = parser.parse_args()
    if args.continue_run and not SMOKE_ROOT.exists():
        raise RuntimeError(f"Cannot continue; smoke output directory is missing: {SMOKE_ROOT}")
    if not args.keep and not args.continue_run and SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)

    run("-m", "task1_llm_ft.preflight", "--smoke")
    if not args.continue_run:
        for config in CONFIGS.values():
            run("task1_llm_ft/train.py", "--config", str(config))

        run(
            "task1_llm_ft/verify_artifacts.py",
            "--root",
            str(SMOKE_ROOT),
            "--checkpoint-counts",
            "4,8",
        )

        # Resume from the first optimizer checkpoint and verify that the second
        # update can be recovered without changing the configured output location.
        run(
            "task1_llm_ft/train.py",
            "--config",
            str(CONFIGS["full_ft"]),
            "--resume",
            str(SMOKE_ROOT / "full_ft/checkpoint-4"),
        )
        require(SMOKE_ROOT / "full_ft/checkpoint-8/training_state.pt")
        for name in ("lora16", "op_lora16"):
            run(
                "task1_llm_ft/train.py",
                "--config",
                str(CONFIGS[name]),
                "--resume",
                str(SMOKE_ROOT / name / "checkpoint-4"),
            )
            require(SMOKE_ROOT / name / "checkpoint-8/training_state.pt")

        for name in ("lora4", "lora16", "op_lora16"):
            config = CONFIGS[name]
            checkpoint = SMOKE_ROOT / name / "checkpoint-8"
            merged = SMOKE_ROOT / name / "merged"
            run(
                "task1_llm_ft/merge_adapters.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(merged),
            )
            require(merged / "merge_verification.json")
            verification = json.loads(
                (merged / "merge_verification.json").read_text(encoding="utf-8")
            )
            if verification.get("adapter_delta_frobenius_norm", 0.0) <= 0.0:
                raise RuntimeError(
                    f"{name} adapter update is zero; smoke training did not move the adapter."
                )
    else:
        run(
            "task1_llm_ft/verify_artifacts.py",
            "--root",
            str(SMOKE_ROOT),
            "--checkpoint-counts",
            "8",
        )
        require(SMOKE_ROOT / "full_ft/checkpoint-8/training_state.pt")
        for name in ("lora4", "lora16", "op_lora16"):
            require(SMOKE_ROOT / name / "merged/merge_verification.json")
            verification = json.loads(
                (SMOKE_ROOT / name / "merged/merge_verification.json").read_text(
                    encoding="utf-8"
                )
            )
            if verification.get("adapter_delta_frobenius_norm", 0.0) <= 0.0:
                raise RuntimeError(
                    f"{name} adapter update is zero; rerun smoke training before continuing."
                )

    evaluation_paths = {
        "baseline": "Qwen/Qwen3-0.6B",
        "full_ft": str(SMOKE_ROOT / "full_ft/checkpoint-8"),
        "full_ft_l2sp": str(SMOKE_ROOT / "full_ft_l2sp/checkpoint-8"),
        "lora4": str(SMOKE_ROOT / "lora4/merged"),
        "lora16": str(SMOKE_ROOT / "lora16/merged"),
        "op_lora16": str(SMOKE_ROOT / "op_lora16/merged"),
    }
    for name, model_path in evaluation_paths.items():
        output = SMOKE_ROOT / f"eval_{name}.json"
        samples = SMOKE_ROOT / f"eval_{name}.samples.json"
        if not (valid_json(output) and valid_json(samples)):
            run(
                "task1_llm_ft/evaluate.py",
                "--model-path",
                model_path,
                "--output",
                str(output),
                "--batch-size",
                "1",
                "--limit",
                "1",
            )
        require(output)
        require(samples)

    qualitative_models = [
        f"baseline={CONFIGS['full_ft']}=Qwen/Qwen3-0.6B",
        f"full_ft={CONFIGS['full_ft']}={SMOKE_ROOT / 'full_ft/checkpoint-8'}",
        f"full_ft_l2sp={CONFIGS['full_ft_l2sp']}={SMOKE_ROOT / 'full_ft_l2sp/checkpoint-8'}",
        f"lora4={CONFIGS['lora4']}={SMOKE_ROOT / 'lora4/merged'}",
        f"lora16={CONFIGS['lora16']}={SMOKE_ROOT / 'lora16/merged'}",
        f"op_lora16={CONFIGS['op_lora16']}={SMOKE_ROOT / 'op_lora16/merged'}",
    ]
    run(
        "task1_llm_ft/qualitative.py",
        "--prompts",
        str(ROOT / "first_plot_questions.yaml"),
        "--models",
        *qualitative_models,
        "--limit",
        "1",
        "--output",
        str(SMOKE_ROOT / "qualitative.json"),
    )
    require(SMOKE_ROOT / "qualitative.json")

    summary = {
        "status": "PASS",
        "methods_trained": list(CONFIGS),
        "resume_checkpoint": str(SMOKE_ROOT / "full_ft/checkpoint-4"),
        "limited_evaluations": len(evaluation_paths),
        "qualitative_responses": len(qualitative_models),
        "merge_verifications": {
            name: json.loads((SMOKE_ROOT / name / "merged/merge_verification.json").read_text())
            for name in ("lora4", "lora16", "op_lora16")
        },
        "artifacts_root": str(SMOKE_ROOT),
    }
    (SMOKE_ROOT / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
