import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.amp import GradScaler

from .recovery import save_checkpoint


class FakeApi:
    def __init__(self, omit=None):
        self.files = set()
        self.omit = omit
        self.uploaded = []
    def upload_folder(self, folder_path, path_in_repo, **_kwargs):
        self.files.update(f"{path_in_repo}/{p.name}" for p in Path(folder_path).iterdir() if p.name != self.omit)
    def list_repo_files(self, *_args, **_kwargs):
        return sorted(self.files)
    def upload_file(self, path_in_repo, **_kwargs):
        self.uploaded.append(path_in_repo)
        self.files.add(path_in_repo)


class RecoveryTest(unittest.TestCase):
    def test_marker_written_only_after_complete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text('{}\n')
            connector, model = nn.Linear(2, 2), nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(list(connector.parameters()) + list(model.parameters()))
            scaler = GradScaler("cuda", enabled=False)
            sampler = type("Sampler", (), {"state_dict": lambda self: {"stage": 1, "update": 1}})()
            api = FakeApi(omit="adapter.safetensors")
            with patch("task2_vlm.recovery._client", return_value=(api, "user/private", "token")):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(root, 250, connector, model, optimizer, scaler, sampler, {"step": 250})
            self.assertNotIn("results/task2/latest_verified.json", api.uploaded)
            api = FakeApi()
            with patch("task2_vlm.recovery._client", return_value=(api, "user/private", "token")):
                save_checkpoint(root, 250, connector, model, optimizer, scaler, sampler, {"step": 250})
            self.assertEqual(api.uploaded[-1], "results/task2/latest_verified.json")


if __name__ == "__main__":
    unittest.main()
