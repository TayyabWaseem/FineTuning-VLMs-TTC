import unittest

from .metrics import CONDITIONS, consensus_score, summarize


class PairMetricsTest(unittest.TestCase):
    def test_consensus_uses_leave_one_out(self):
        self.assertAlmostEqual(consensus_score("yes", ["yes"] * 3 + ["no"] * 7), 0.9)

    def test_paired_advantage_and_original_references(self):
        rows = []
        for pair in range(3):
            for condition in CONDITIONS:
                for index, canonical in enumerate(("yes", "no")):
                    correct = condition == "correct"
                    rows.append({"suite": "grounding", "pair_id": pair, "item_id": index,
                                 "condition": condition, "question_type": "presence",
                                 "prediction_normalized": canonical if correct else "",
                                 "canonical_normalized": canonical, "reference_nll": 1 if correct else 2,
                                 "generation_seconds": 0.5})
        result = summarize(rows, resamples=40)
        swap = next(row for row in result if row["question_type"] == "all" and row["condition"] == "complementary")
        self.assertEqual(swap["correct_advantage_accuracy"], 1)
        self.assertEqual(swap["correct_advantage_reference_nll"], 1)
        self.assertEqual(swap["correct_advantage_accuracy_ci_low"], 1)
        rows[-1]["canonical_normalized"] = "changed"
        with self.assertRaises(ValueError):
            summarize(rows, resamples=40)


if __name__ == "__main__":
    unittest.main()
