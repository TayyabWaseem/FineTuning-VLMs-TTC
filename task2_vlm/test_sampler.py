import unittest

from .sampler import CurriculumSampler


class SamplerTest(unittest.TestCase):
    def test_two_caption_passes_and_resumed_pair_draws(self):
        pools = {"caption": list(range(18000)),
                 "presence": [{"examples": ["p1", "p2"]}],
                 "counting": [{"examples": ["c1", "c2"]}],
                 "natural": [{"examples": ["n1", "n2"]}]}
        sampler = CurriculumSampler(pools)
        first, second = [], []
        for i in range(2250):
            examples, _ = sampler.next_update()
            (first if i < 1125 else second).extend(examples)
        self.assertEqual(set(first), set(range(18000)))
        self.assertEqual(set(second), set(range(18000)))
        sampler.next_stage()
        state = sampler.state_dict()
        expected = sampler.next_update()
        restored = CurriculumSampler(pools)
        restored.load_state_dict(state)
        self.assertEqual(restored.next_update(), expected)
        self.assertEqual(len(expected[0]), 16)
        self.assertEqual(sum(expected[1].values()), 8)


if __name__ == "__main__":
    unittest.main()
