"""Read cached SigLIP patches by COCO image ID."""
import json
from pathlib import Path

import numpy as np


class FeatureStore:
    def __init__(self, path: Path):
        self.path = path
        self.index = json.loads((path / "feature_index.json").read_text(encoding="utf-8"))
        self.rows = {str(image): row for row, image in enumerate(self.index["image_ids"])}
        self.array = np.load(path / "features.npy", mmap_mode="r")
        if tuple(self.array.shape) != tuple(self.index["shape"]) or self.array.dtype != np.float16:
            raise RuntimeError("Feature cache shape or dtype does not match index")
        if len(self.rows) != len(self.index["image_ids"]):
            raise RuntimeError("Feature cache has duplicate image IDs")

    def __getitem__(self, image_id: int):
        return np.array(self.array[self.rows[str(image_id)]], copy=True)
