import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from . import feature_cache


class FakeApi:
    def __init__(self): self.files = {}
    def model_info(self, _name): return SimpleNamespace(sha="fixed-model-revision")
    def list_repo_files(self, *_args, **_kwargs): return sorted(self.files)
    def upload_file(self, path_or_fileobj, path_in_repo, **_kwargs):
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()


class FakePixels:
    def __init__(self, count): self.count = count
    def to(self, *_args, **_kwargs): return self


class FakeProcessor:
    @staticmethod
    def from_pretrained(*_args, **_kwargs): return FakeProcessor()
    def __call__(self, images, **_kwargs): return {"pixel_values": FakePixels(len(images))}


class FakeEncoder:
    calls = 0
    @staticmethod
    def from_pretrained(*_args, **_kwargs): return FakeEncoder()
    def to(self, *_args, **_kwargs): return self
    def eval(self): return self
    def requires_grad_(self, *_args): return self
    def __call__(self, pixel_values):
        FakeEncoder.calls += 1
        return SimpleNamespace(last_hidden_state=torch.ones(pixel_values.count, 196, 768, dtype=torch.float16))


class FeatureCacheTest(unittest.TestCase):
    def test_uploaded_shards_are_restored_without_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            output = project / "results/task2/features"
            fake_code = project / "task2_vlm"
            fake_code.mkdir()
            original_code = Path(feature_cache.__file__).resolve().parent
            for name in ("feature_cache.py", "download.py"):
                (fake_code / name).write_bytes((original_code / name).read_bytes())
            images = []
            for i in range(3):
                path = project / f"image-{i}.png"
                Image.new("RGB", (4, 4)).save(path)
                images.append({"id": i, "path": str(path), "split": "train2014", "filename": path.name})
            api = FakeApi()
            def download(*_args, local_dir, allow_patterns, **_kwargs):
                for path in allow_patterns:
                    target = Path(local_dir) / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(api.files[path])
            FakeEncoder.calls = 0
            with patch.object(feature_cache, "__file__", str(project / "task2_vlm/feature_cache.py")), \
                 patch.object(feature_cache, "_client", return_value=(api, "user/private", "token")), \
                 patch.object(feature_cache, "AutoProcessor", FakeProcessor), \
                 patch.object(feature_cache, "SiglipVisionModel", FakeEncoder), \
                 patch.object(feature_cache, "snapshot_download", side_effect=download), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.get_device_name", return_value="test-gpu"), \
                 patch("torch.cuda.reset_peak_memory_stats"), \
                 patch("torch.cuda.max_memory_allocated", return_value=1024):
                feature_cache.cache_features(images, output, batch_size=2, shard_size=2)
                count = FakeEncoder.calls
                feature_cache.cache_features(images, output, batch_size=2, shard_size=2)
                self.assertEqual(FakeEncoder.calls, count)
            self.assertEqual(np.load(output / "features.npy").shape, (3, 196, 768))


if __name__ == "__main__":
    unittest.main()
