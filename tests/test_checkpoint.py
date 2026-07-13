from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.messages import Message
from harness.runtime.checkpoint import RunCheckpoint, RunCheckpointStore


class CheckpointStoreTests(unittest.TestCase):
    def test_checkpoint_store_saves_and_loads_latest_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunCheckpointStore(Path(tmp))
            checkpoint = RunCheckpoint(
                run_id="run-1",
                task="Inspect this project",
                step=2,
                status="running",
                messages=[Message("user", "Inspect this project")],
                plan={"items": [], "counts": {}},
            )

            path = store.save(checkpoint)
            loaded = store.load("run-1")
            loaded_state = store.load_state("run-1")

            self.assertEqual(path.name, "run-1.json")
            self.assertEqual(loaded["run_id"], "run-1")
            self.assertEqual(loaded["step"], 2)
            self.assertEqual(loaded["messages"][0], {"role": "user", "content": "Inspect this project"})
            self.assertEqual(loaded_state.task, "Inspect this project")
            self.assertEqual(loaded_state.messages[0].content, "Inspect this project")
            self.assertEqual(loaded_state.plan.snapshot()["items"], [])

    def test_checkpoint_store_rejects_path_like_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunCheckpointStore(Path(tmp))

            with self.assertRaises(ValueError):
                store.load("../bad")


if __name__ == "__main__":
    unittest.main()
