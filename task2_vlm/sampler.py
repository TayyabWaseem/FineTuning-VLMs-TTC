"""Reproducible curriculum draws; pair draws remain together across microbatches."""
from __future__ import annotations

import copy
import numpy as np


STAGE_UPDATES = (2250, 1500, 750)
MIXTURES = {2: (("presence", "counting", "caption"), (0.50, 0.25, 0.25)),
            3: (("natural", "presence", "counting", "caption"), (0.70, 0.10, 0.10, 0.10))}


class CurriculumSampler:
    def __init__(self, pools: dict[str, list], seed: int = 6304):
        self.pools = pools
        if len(pools.get("caption", [])) != 18000:
            raise ValueError("Caption alignment requires exactly 18,000 examples")
        self.rng = np.random.default_rng(seed)
        self.stage = 1
        self.update = 0
        self.epoch = 0
        self.caption_order = self.rng.permutation(18000).tolist()

    def next_update(self) -> tuple[list, dict[str, int]]:
        if self.stage not in (1, 2, 3) or self.update >= STAGE_UPDATES[self.stage - 1]:
            raise StopIteration("Stage complete; enter the next stage explicitly")
        if self.stage == 1:
            start = (self.update * 16) % 18000
            if start == 0 and self.update:
                self.epoch += 1
                self.caption_order = self.rng.permutation(18000).tolist()
            indices = self.caption_order[start:start + 16]
            if len(indices) != 16:
                raise RuntimeError("Caption batch straddles an epoch unexpectedly")
            examples = [self.pools["caption"][i] for i in indices]
            counts = {"caption": 16}
        else:
            names, probabilities = MIXTURES[self.stage]
            examples = []
            counts = {name: 0 for name in names}
            for _ in range(8):
                name = str(self.rng.choice(names, p=probabilities))
                pool = self.pools.get(name, [])
                if not pool:
                    raise ValueError(f"Missing training pool: {name}")
                if name == "caption":
                    examples.extend(pool[int(self.rng.integers(len(pool)))] for _ in range(2))
                else:
                    pair = pool[int(self.rng.integers(len(pool)))]
                    if len(pair["examples"]) != 2:
                        raise ValueError(f"{name} training draw must have two examples")
                    examples.extend(pair["examples"])
                counts[name] += 1
        self.update += 1
        return examples, counts

    def next_stage(self) -> None:
        if self.update != STAGE_UPDATES[self.stage - 1] or self.stage == 3:
            raise ValueError("Current stage is incomplete or final")
        self.stage += 1
        self.update = 0

    def state_dict(self) -> dict:
        return {"stage": self.stage, "update": self.update, "epoch": self.epoch,
                "caption_order": self.caption_order, "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict) -> None:
        self.stage, self.update, self.epoch = state["stage"], state["update"], state["epoch"]
        self.caption_order = list(state["caption_order"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
