from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.context import ContextBudget, ContextManager
from harness.messages import Message
from harness.runtime.workspace import Workspace


class ContextManagerTests(unittest.TestCase):
    def test_repo_summary_samples_files_and_ignores_harness_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
            (root / ".harness" / "logs").mkdir(parents=True)
            (root / ".harness" / "logs" / "run.jsonl").write_text("{}", encoding="utf-8")
            manager = ContextManager(Workspace(root), ContextBudget(max_summary_files=20))

            self.assertIn("src", manager.repo_summary)
            self.assertIn("main.py", manager.repo_summary)
            self.assertNotIn("run.jsonl", manager.repo_summary)

    def test_prepare_for_model_keeps_head_and_recent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextManager(Workspace(Path(tmp)), ContextBudget(max_message_chars=120))
            messages = [
                Message("system", "s" * 10),
                Message("user", "repo" * 10),
                Message("user", "task"),
                Message("assistant", "old" * 100),
                Message("user", "new" * 10),
            ]

            trimmed = manager.prepare_for_model(messages)

            self.assertEqual(trimmed[0].role, "system")
            self.assertEqual(trimmed[1].role, "user")
            self.assertEqual(trimmed[2].content, "task")
            self.assertLessEqual(sum(len(message.content) for message in trimmed), 180)
            self.assertIn("new", trimmed[-1].content)

    def test_compress_tool_result_truncates_large_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextManager(Workspace(Path(tmp)), ContextBudget(max_tool_result_chars=80))
            result = manager.compress_tool_result({"ok": True, "content": "x" * 500})

            self.assertIs(result["ok"], False)
            self.assertIs(result["truncated"], True)
            self.assertIn("original_chars", result)


if __name__ == "__main__":
    unittest.main()
