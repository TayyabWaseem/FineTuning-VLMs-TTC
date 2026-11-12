import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from . import analysis
from .metrics import CONDITIONS


class AnalysisTest(unittest.TestCase):
    def test_builds_evidence_from_complete_synthetic_records(self):
        class Api:
            def upload_file(self, **_kwargs): pass
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            root = project / "results/task2"
            features = root / "features"
            features.mkdir(parents=True)
            (features / "feature_index.json").write_text(json.dumps({"elapsed_seconds": 1, "peak_gpu_memory_bytes": 1, "shape": [1, 196, 768]}))
            (root / "train.jsonl").write_text(''.join(json.dumps({"step": i, "loss": 1})+'\n' for i in range(1, 4501)))
            steps = sorted(set(range(0, 4501, 500)) | {2250, 3750, 4500})
            dev = [{"step": i, "caption_reference_loss": 1, "natural": {"correct": 1, "swap_penalty": 0.1},
                    "grounding": {"correct": 1, "swap_penalty": 0.2}} for i in steps]
            (root / "development.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in dev))
            for step in (2250, 3750, 4500):
                folder = root / f"checkpoint-{step}"
                folder.mkdir()
                (folder / "metadata.json").write_text(json.dumps({"stage_training_seconds": {"1": 1, "2": 2, "3": 3},
                    "stage_peak_gpu_memory_bytes": {"1": 1, "2": 2, "3": 3},
                    "trainable_connector_parameters": 1, "trainable_adapter_parameters": 1, "calibration": {"scale": 1}}))
            (root / "run_config.json").write_text('{}')
            manifest = {"images": [], "test": {"grounding": [], "natural": []}, "normalizer": "test",
                        "annotation_source_sha256": {}, "mismatched_rule": "test"}
            for suite in ("grounding", "natural"):
                for p in range(2):
                    pair = {"pair_id": f"{suite}-{p}", "question_type": "presence" if suite == "grounding" else "yes/no", "examples": []}
                    for i, answer in enumerate(("yes", "no")):
                        image_id = 100 * p + i + (1000 if suite == "natural" else 0)
                        path = root / f"image-{image_id}.png"
                        Image.new("RGB", (8, 8)).save(path)
                        manifest["images"].append({"id": image_id, "path": str(path), "split": "val2014", "filename": path.name})
                        pair["examples"].append({"image_id": image_id, "question": "Question?", "answer": answer, "question_id": image_id})
                    manifest["test"][suite].append(pair)
                for condition in CONDITIONS:
                    records = []
                    for pair in manifest["test"][suite]:
                        for item in pair["examples"]:
                            records.append({"suite": suite, "pair_id": pair["pair_id"], "item_id": str(item["question_id"]),
                                "question_type": pair["question_type"], "condition": condition, "prediction": item["answer"],
                                "prediction_normalized": item["answer"], "canonical_normalized": item["answer"],
                                "references_normalized": [item["answer"]] * 10 if suite == "natural" else [],
                                "reference_nll": 1, "generation_seconds": 0.1})
                    (root / f"final_{suite}_{condition}.json").write_text(json.dumps({"records": records}))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(analysis, "__file__", str(project / "task2_vlm/analysis.py")), \
                 patch.object(analysis, "_client", return_value=(Api(), "user/private", "token")), \
                 patch("huggingface_hub.snapshot_download"), \
                 patch("sys.argv", ["analysis", "--manifest", str(manifest_path), "--features", str(features), "--output", str(root)]):
                analysis.main()
            self.assertTrue((root / "learning_curves.png").is_file())
            self.assertTrue((root / "final_summary.csv").is_file())
            self.assertEqual(json.loads((root / "analysis_manifest.json").read_text())["final_records"], 32)


if __name__ == "__main__":
    unittest.main()
