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
            record.started_at = "2026-07-15T00:00:00+00:00"
            record.finished_at = "2026-07-15T00:00:01.500000+00:00"
            record.duration_seconds = 1.5
            record.result = {"ok": True}
            store.append(record)

            loaded = store.load_latest()

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].task_id, "task-1")
            self.assertEqual(loaded[0].status, "completed")
            self.assertEqual(loaded[0].result, {"ok": True})
            self.assertEqual(loaded[0].started_at, "2026-07-15T00:00:00+00:00")
            self.assertEqual(loaded[0].finished_at, "2026-07-15T00:00:01.500000+00:00")
            self.assertEqual(loaded[0].duration_seconds, 1.5)


    def test_load_latest_accepts_old_records_without_timing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(
                '{"task_id":"task-1","run_id":"run-1","task":"demo","stream":false,"status":"completed"}\n',
                encoding="utf-8",
            )
            store = TaskStore(path)

            loaded = store.load_latest()

            self.assertEqual(len(loaded), 1)
            self.assertIsNone(loaded[0].started_at)
            self.assertIsNone(loaded[0].finished_at)
            self.assertIsNone(loaded[0].duration_seconds)

if __name__ == "__main__":
    unittest.main()
