from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.runtime.task_queue import TaskRecord
from harness.runtime.task_store import TaskStore


class TaskStoreTests(unittest.TestCase):
    def test_load_latest_keeps_last_record_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.jsonl")
            record = TaskRecord(task_id="task-1", run_id="run-1", task="demo", stream=False)
            store.append(record)
            record.status = "completed"
            record.result = {"ok": True}
            store.append(record)

            loaded = store.load_latest()

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].task_id, "task-1")
            self.assertEqual(loaded[0].status, "completed")
            self.assertEqual(loaded[0].result, {"ok": True})


if __name__ == "__main__":
    unittest.main()
