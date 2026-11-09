"""Offline regression checks for checkpoint upload and restore paths."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import remote


class FakeApi:
    def __init__(self) -> None:
        self.files: set[str] = set()

    def upload_folder(self, *, folder_path: str, path_in_repo: str, **_kwargs) -> None:
        self.files.update(
            f"{path_in_repo}/{path.name}"
            for path in Path(folder_path).iterdir() if path.is_file()
        )

    def upload_file(self, *, path_in_repo: str, **_kwargs) -> None:
        self.files.add(path_in_repo)

    def list_repo_files(self, *_args, **_kwargs) -> list[str]:
        return sorted(self.files)


class RemotePathTest(unittest.TestCase):
    def test_checkpoint_and_progress_use_one_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/task1/full_ft"
            checkpoint = output / "checkpoint-9984"
            checkpoint.mkdir(parents=True)
            (checkpoint / "training_state.pt").write_bytes(b"state")
            (checkpoint / "run_config.json").write_text("{}")
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            (output / "train.jsonl").write_text("{}\n")
            api = FakeApi()
            with patch.object(remote, "PROJECT_ROOT", root), patch.object(
                remote, "_client", return_value=(api, "user/private", "token")
            ):
                remote.sync_checkpoint(checkpoint, output)
            prefix = "results/task1/full_ft/"
            self.assertIn(prefix + "checkpoint-9984/training_state.pt", api.files)
            self.assertIn(prefix + "train.jsonl", api.files)
            self.assertIn(prefix + "remote_checkpoints.json", api.files)
            index = json.loads((output / "remote_checkpoints.json").read_text())
            self.assertEqual(index["checkpoint-9984"]["path"], prefix + "checkpoint-9984")
            with patch.object(remote, "PROJECT_ROOT", root), patch.object(
                remote, "_client", return_value=(api, "user/private", "token")
            ), patch("huggingface_hub.snapshot_download") as download:
                self.assertEqual(remote.restore_latest(output), checkpoint)
            self.assertIn(prefix + "checkpoint-9984/*", download.call_args.kwargs["allow_patterns"])


if __name__ == "__main__":
    unittest.main()
