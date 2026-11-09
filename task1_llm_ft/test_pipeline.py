"""Check that separately saved benchmark tasks assemble into one condition."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import run_pipeline


class EvaluationRestartTest(unittest.TestCase):
    def test_reuses_complete_task_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results" / "task1"
            results.mkdir(parents=True)
            for task in run_pipeline.TASKS:
                part = results / f"eval_baseline.{task}.json"
                part.write_text(json.dumps({
                    "results": {task: {"score": 0.5}},
                    "samples": {task: [{"doc_id": 0}]},
                    "versions": {task: 1}, "configs": {task: {}},
                    "evaluation_settings": {"tasks": [task], "model_path": "starting-model"},
                }), encoding="utf-8")
                part.with_suffix(".samples.json").write_text(json.dumps({task: [{"doc_id": 0}]}), encoding="utf-8")
            gsm = results / "eval_baseline.gsm8k_cot.gsm8k_examples.json"
            gsm.write_text("[]", encoding="utf-8")
            with patch.object(run_pipeline, "ROOT", root), \
                 patch.object(run_pipeline, "RESULTS", results), \
                 patch.object(run_pipeline, "restore_artifacts"), \
                 patch.object(run_pipeline, "sync_artifact"), \
                 patch.object(run_pipeline, "run") as execute:
                run_pipeline.evaluate("baseline")
                execute.assert_not_called()
            combined = json.loads((results / "eval_baseline.json").read_text(encoding="utf-8"))
            self.assertEqual(set(combined["results"]), set(run_pipeline.TASKS))
            self.assertEqual(combined["evaluation_settings"]["tasks"], list(run_pipeline.TASKS))
            self.assertTrue((results / "eval_baseline.gsm8k_examples.json").is_file())


if __name__ == "__main__":
    unittest.main()
