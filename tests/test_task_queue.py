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
        self.assertIsNotNone(final.started_at)
        self.assertIsNotNone(final.finished_at)
        self.assertIsNotNone(final.duration_seconds)
        self.assertGreaterEqual(final.duration_seconds or 0, 0)

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
            self.assertEqual(restored_record.started_at, final.started_at)
            self.assertEqual(restored_record.finished_at, final.finished_at)
            self.assertEqual(restored_record.duration_seconds, final.duration_seconds)


    def test_failed_task_records_timing(self) -> None:
        def fail(task: str, stream: bool, run_id: str, metadata: dict[str, object]) -> dict[str, object]:
            raise RuntimeError("boom")

        queue = TaskQueue(fail)

        record = queue.submit("demo")
        final = _wait_for_status(queue, record.task_id, "failed")

        self.assertEqual(final.error, "boom")
        self.assertIsNotNone(final.started_at)
        self.assertIsNotNone(final.finished_at)
        self.assertIsNotNone(final.duration_seconds)
        self.assertGreaterEqual(final.duration_seconds or 0, 0)

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
