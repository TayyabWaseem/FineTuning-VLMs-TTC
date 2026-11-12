import unittest

import numpy as np

from .selection import _draw, assert_split_isolation, natural_candidates, grounding_pools, select_captions
from .normalize import normalize_answer


class SelectionTest(unittest.TestCase):
    def test_normalization_examples(self):
        self.assertEqual(normalize_answer("The TWO cats!"), "2 cats")
        self.assertEqual(normalize_answer("1,234.50"), "1234.50")
        self.assertEqual(normalize_answer("Don't"), normalize_answer("dont"))

    def test_consensus_and_different_answers(self):
        questions = [{"question_id": 1, "image_id": 10, "question": "Is it red?"},
                     {"question_id": 2, "image_id": 20, "question": "Is it red?"}]
        annotations = [{"question_id": 1, "answer_type": "yes/no", "answers": [{"answer": "Yes"}] * 8 + [{"answer": "no"}] * 2},
                       {"question_id": 2, "answer_type": "yes/no", "answers": [{"answer": "No"}] * 10}]
        candidates = natural_candidates(questions, annotations, [[1, 2]], str.lower)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["examples"][0]["agreement"], 8)
        annotations[0]["answers"] = [{"answer": "Yes"}] * 7 + [{"answer": "no"}] * 3
        self.assertEqual(natural_candidates(questions, annotations, [[1, 2]], str.lower), [])

    def test_disjoint_pair_selection(self):
        candidates = [{"pair_id": str(i), "question_type": "other", "examples": [
            {"image_id": i * 2, "question_id": i * 2}, {"image_id": i * 2 + 1, "question_id": i * 2 + 1}]} for i in range(10)]
        chosen = _draw(candidates, {"other": 3}, np.random.default_rng(6304), {0, 1}, set())
        self.assertEqual(len(chosen), 3)
        self.assertTrue(all(e["image_id"] not in {0, 1} for pair in chosen for e in pair["examples"]))
        with self.assertRaises(ValueError):
            assert_split_isolation({"dev": {1, 2}, "test": {2, 3}})

    def test_caption_and_grounding_generation(self):
        captions = [{"id": image * 10 + i, "image_id": image, "caption": f"caption {i}"}
                    for image in range(1, 8) for i in range(3)]
        selected = select_captions(captions, set(range(1, 8)), 4, 3, np.random.default_rng(1))
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row["image_id"] for row in selected}), 4)
        instances = [{"image_id": 1, "category_id": 1}, {"image_id": 2, "category_id": 1},
                     {"image_id": 2, "category_id": 1}, {"image_id": 3, "category_id": 1, "iscrowd": 1},
                     {"image_id": 4, "category_id": 1}, {"image_id": 5, "category_id": 1},
                     {"image_id": 5, "category_id": 1}]
        pools = grounding_pools(set(range(1, 8)), instances, [{"id": 1, "name": "cat"}], np.random.default_rng(2), 4)
        for pair in pools["presence"] + pools["counting"]:
            self.assertNotIn(3, [example["image_id"] for example in pair["examples"]])
            self.assertNotEqual(pair["examples"][0]["answer"], pair["examples"][1]["answer"])


if __name__ == "__main__":
    unittest.main()
