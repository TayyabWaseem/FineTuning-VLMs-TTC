"""End-to-end shape check for report evidence without a GPU or model download."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from .analyze import CONDITIONS, DIAGNOSTICS, OUTPUTS, PRIMARY, TRAINED, make_analysis


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class AnalysisOutputTest(unittest.TestCase):
    def test_complete_evidence_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in TRAINED:
                folder = root / name
                folder.mkdir()
                rows = [dict(update=i, examples_seen=32*i, sft_loss=1.0 / i,
                             l2_sp_omega=0.1, l2_sp_term=0.0001,
                             total_loss=1.0/i + 0.0001, l2_sp_pass_seconds=0.01)
                        for i in range(1, 1251)]
                (folder / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                write_json(folder / "metadata.json", {
                    "trainable_parameters": 1000, "total_parameters": 2000,
                    "training_seconds": 3600, "peak_gpu_memory_bytes": 2**30,
                    "remote_checkpoint_sizes_bytes": {f"checkpoint-{i}": 2**30 for i in (9984, 20000, 29984, 40000)},
                    "midpoint_parameter_displacement_l2": 1.0,
                    "final_parameter_displacement_l2": 2.0,
                    "skipped_updates": 0,
                })
            for name in CONDITIONS:
                write_json(root / f"eval_{name}.json", {"results": {
                    task: {metric: 0.5, "acc,none": 0.4, "acc_norm,none": 0.5}
                    for task, metric in PRIMARY.items()
                }})
            for name in DIAGNOSTICS:
                write_json(root / f"diagnostics_{name}.json", {
                    "weight": [dict(layer=layer, rho_k=0.1, eta=0.01, delta_frobenius=1.0) for layer in (0, 13, 27)],
                    "activation": {kind: [dict(layer=layer, probe_id=str(i), correction_ratio=0.2)
                                          for layer in (0, 13, 27) for i in range(32)]
                                   for kind in ("math", "non_math")},
                })
            responses = [dict(condition=name, question_id=question + suffix,
                              format=fmt, prompt="test", response="response", truncated=False)
                         for name in CONDITIONS
                         for fmt, suffix in (("plain", ""), ("json", "_json"), ("template", "_template"))
                         for question in ("what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
                                          "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband")]
            write_json(root / "qualitative.json", responses)
            manifest = make_analysis(root)
            self.assertEqual(manifest["counts"]["benchmark_rows"], 28)
            self.assertEqual(manifest["counts"]["diagnostic_rows"], 1170)
            self.assertEqual(manifest["counts"]["human_annotation_rows"], 42)
            self.assertTrue(all((root / name).is_file() for name in OUTPUTS))
            self.assertEqual(len(pd.read_csv(root / "benchmark_matrix.csv")), 7)
            self.assertEqual(len(pd.read_csv(root / "diagnostic_summary.csv")), 36)


if __name__ == "__main__":
    unittest.main()
