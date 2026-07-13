from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.init_project import init_workspace


class InitProjectTests(unittest.TestCase):
    def test_init_writes_harness_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = init_workspace(tmp)

            path = Path(tmp) / "harness.json"
            self.assertEqual(results[0].status, "written")
            self.assertTrue(path.exists())
            self.assertIn('"tool_profile": "full"', path.read_text(encoding="utf-8"))
            self.assertIn('"command_profile": "default"', path.read_text(encoding="utf-8"))
            self.assertIn('"native_tools": false', path.read_text(encoding="utf-8"))
            self.assertIn('"json_mode": false', path.read_text(encoding="utf-8"))
            self.assertIn('"trace_messages": false', path.read_text(encoding="utf-8"))
            self.assertIn('"trace_model_responses": false', path.read_text(encoding="utf-8"))
            self.assertIn('"model_max_retries": 2', path.read_text(encoding="utf-8"))
            self.assertIn('"max_run_tokens": 0', path.read_text(encoding="utf-8"))

    def test_init_can_write_env_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = init_workspace(tmp, include_env=True)

            self.assertEqual([result.path.name for result in results], ["harness.json", ".env"])
            self.assertTrue((Path(tmp) / ".env").exists())
            self.assertIn("HARNESS_NATIVE_TOOLS=false", (Path(tmp) / ".env").read_text(encoding="utf-8"))
            self.assertIn("HARNESS_JSON_MODE=false", (Path(tmp) / ".env").read_text(encoding="utf-8"))
            self.assertIn("HARNESS_TRACE_MESSAGES=false", (Path(tmp) / ".env").read_text(encoding="utf-8"))
            self.assertIn("HARNESS_TRACE_MODEL_RESPONSES=false", (Path(tmp) / ".env").read_text(encoding="utf-8"))
            self.assertIn("HARNESS_MODEL_MAX_RETRIES=2", (Path(tmp) / ".env").read_text(encoding="utf-8"))
            self.assertIn("HARNESS_MAX_RUN_TOKENS=0", (Path(tmp) / ".env").read_text(encoding="utf-8"))

    def test_init_skips_existing_files_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "harness.json"
            path.write_text("custom", encoding="utf-8")

            results = init_workspace(tmp)

            self.assertEqual(results[0].status, "skipped")
            self.assertEqual(path.read_text(encoding="utf-8"), "custom")

    def test_init_force_overwrites_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "harness.json"
            path.write_text("custom", encoding="utf-8")

            results = init_workspace(tmp, force=True)

            self.assertEqual(results[0].status, "written")
            self.assertIn('"provider": "deepseek"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
