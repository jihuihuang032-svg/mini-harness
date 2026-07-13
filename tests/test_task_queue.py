from __future__ import annotations

import time
import unittest
import tempfile
from pathlib import Path

from harness.runtime.task_queue import TaskQueue
from harness.runtime.task_store import TaskStore


class TaskQueueTests(unittest.TestCase):
    def test_submit_runs_task_in_background(self) -> None:
        queue = TaskQueue(lambda task, stream, run_id, metadata: {"task": task, "stream": stream, "run_id": run_id, "metadata": metadata})

        record = queue.submit("demo", stream=True, mode="model", provider="deepseek", metadata={"provider": "deepseek"})

        self.assertIn(record.status, {"queued", "running", "completed"})
        final = _wait_for_status(queue, record.task_id, "completed")
        self.assertEqual(final.mode, "model")
        self.assertEqual(final.provider, "deepseek")
        self.assertEqual(final.result, {"task": "demo", "stream": True, "run_id": record.run_id, "metadata": {"provider": "deepseek"}})

    def test_queue_persists_task_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.jsonl")
            queue = TaskQueue(lambda task, stream, run_id, metadata: {"ok": True, "run_id": run_id}, store=store)

            record = queue.submit("demo")
            final = _wait_for_status(queue, record.task_id, "completed")
            restored = TaskQueue(lambda task, stream, run_id, metadata: {}, store=store)

            restored_record = restored.get(record.task_id)
            self.assertEqual(final.status, "completed")
            self.assertEqual(restored_record.status, "completed")
            self.assertEqual(restored_record.result, {"ok": True, "run_id": record.run_id})


def _wait_for_status(queue: TaskQueue, task_id: str, status: str) -> object:
    deadline = time.time() + 5
    while time.time() < deadline:
        record = queue.get(task_id)
        if record.status == status:
            return record
        time.sleep(0.01)
    raise AssertionError(f"Task did not reach status {status}")


if __name__ == "__main__":
    unittest.main()
