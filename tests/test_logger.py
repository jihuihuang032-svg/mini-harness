from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.runtime.logger import RunLogger
from harness.runtime.run_store import RunStore


class RunLoggerTests(unittest.TestCase):
    def test_logger_writes_ordered_trace_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(Path(tmp))
            logger.event("example", {"ok": True})
            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(records[0]["kind"], "run_started")
            self.assertEqual(records[0]["seq"], 1)
            self.assertEqual(records[1]["kind"], "example")
            self.assertEqual(records[1]["seq"], 2)
            self.assertEqual(records[0]["schema_version"], 1)
            self.assertEqual(records[0]["run_id"], records[1]["run_id"])

    def test_run_store_lists_and_loads_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            changes_dir = Path(tmp).parent / "changes"
            changes_dir.mkdir(parents=True, exist_ok=True)
            logger = RunLogger(logs_dir)
            logger.event("workspace_changes", {"changed_count": 1, "added": ["x.txt"], "modified": [], "deleted": []})
            logger.event(
                "model_response_metadata",
                {"usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
            )
            logger.event(
                "model_response_metadata",
                {"usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}},
            )
            logger.event(
                "tool_result",
                {
                    "action": {"type": "tool_call", "tool": "read_file", "args": {"path": "README.md"}},
                    "result": {"ok": True},
                },
            )
            logger.event(
                "tool_result",
                {
                    "action": {"type": "tool_call", "tool": "run_command", "args": {"command": "bad"}},
                    "result": {"ok": False, "error": "denied"},
                },
            )
            logger.event("final", {"content": "done"})
            logger.event("run_finished", {"status": "completed"})
            (changes_dir / f"{logger.run_id}.json").write_text(
                json.dumps({"changed_count": 2, "added": ["artifact.txt"], "modified": [], "deleted": ["old.txt"]}),
                encoding="utf-8",
            )

            store = RunStore(logs_dir)
            summaries = store.list_runs()
            records = store.load_run(logger.run_id)
            changes = store.load_changes(logger.run_id)

            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].run_id, logger.run_id)
            self.assertEqual(summaries[0].status, "completed")
            self.assertEqual(summaries[0].final, "done")
            self.assertEqual(summaries[0].changes["changed_count"], 1)
            self.assertEqual(summaries[0].usage["response_count"], 2)
            self.assertEqual(summaries[0].usage["prompt_tokens"], 6)
            self.assertEqual(summaries[0].usage["completion_tokens"], 6)
            self.assertEqual(summaries[0].usage["total_tokens"], 12)
            self.assertEqual(summaries[0].tools["total_calls"], 2)
            self.assertEqual(summaries[0].tools["failed_calls"], 1)
            self.assertEqual(summaries[0].tools["tool_names"], ["read_file", "run_command"])
            self.assertEqual(summaries[0].tools["by_tool"]["read_file"], {"calls": 1, "failures": 0})
            self.assertEqual(summaries[0].tools["by_tool"]["run_command"], {"calls": 1, "failures": 1})
            self.assertIn("final", {record["kind"] for record in records})
            self.assertEqual(changes["changed_count"], 2)
            self.assertEqual(changes["added"], ["artifact.txt"])

    def test_run_store_uses_run_finished_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(Path(tmp))
            logger.event("run_finished", {"status": "stopped", "reason": "max_steps"})

            summary = RunStore(Path(tmp)).summarize_run(logger.run_id)

            self.assertEqual(summary.status, "stopped")

    def test_run_store_loads_changes_from_trace_when_artifact_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(Path(tmp))
            logger.event("workspace_changes", {"changed_count": 1, "added": ["x.txt"], "modified": [], "deleted": []})

            changes = RunStore(Path(tmp)).load_changes(logger.run_id)

            self.assertEqual(changes["changed_count"], 1)
            self.assertEqual(changes["added"], ["x.txt"])

    def test_run_store_rejects_path_like_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp))
            with self.assertRaises(ValueError):
                store.load_run("../bad")


if __name__ == "__main__":
    unittest.main()
