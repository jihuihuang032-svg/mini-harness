from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from harness.server import HarnessServer


class HarnessServerTests(unittest.TestCase):
    def test_health_and_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = HarnessServer(tmp)

            self.assertTrue(server.health()["ok"])
            self.assertEqual(Path(server.health()["workspace"]), Path(tmp).resolve())
            provider_names = {provider["name"] for provider in server.providers()}
            self.assertIn("deepseek", provider_names)
            self.assertIn("qwen", provider_names)

    def test_console_html_contains_api_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = HarnessServer(tmp)

            html = server.console_html()

            self.assertIn("Mini Harness Console", html)
            self.assertIn("/tasks", html)
            self.assertIn("/runs?limit=20", html)
            self.assertIn("EventSource", html)
            self.assertIn("Real provider", html)
            self.assertIn(str(Path(tmp).resolve()), html)

    def test_run_mock_task_creates_run_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = HarnessServer(tmp)

            result = server.run_mock_task("Inspect this project", stream=True)

            self.assertIn("run_id", result)
            self.assertIn("Offline mock run completed", result["content"])
            self.assertEqual(result["changes"]["changed_count"], 0)
            self.assertGreater(len(result["stream_chunks"]), 0)
            runs = server.list_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["changes"]["changed_count"], 0)
            changes = server.load_run_changes(str(result["run_id"]))
            self.assertEqual(changes["changed_count"], 0)
            records = server.load_run(str(result["run_id"]))
            self.assertIn("final", {record["kind"] for record in records})
            self.assertIn("workspace_changes", {record["kind"] for record in records})
            run_config = [record for record in records if record["kind"] == "run_config"][0]["payload"]
            self.assertEqual(run_config["mode"], "mock")
            self.assertTrue(run_config["stream"])
            self.assertGreater(run_config["tool_count"], 0)
            self.assertEqual(len(run_config["tool_schema_sha256"]), 64)
            self.assertEqual(len(run_config["system_prompt_sha256"]), 64)
            self.assertNotIn("api_key", run_config)
            checkpoint = server.load_run_checkpoint(str(result["run_id"]))
            self.assertEqual(checkpoint["status"], "completed")
            self.assertGreater(len(checkpoint["messages"]), 1)

    def test_submit_mock_task_runs_asynchronously(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = HarnessServer(tmp)

            task = server.submit_mock_task("Inspect this project", stream=True)

            self.assertIn("task_id", task)
            self.assertIn("run_id", task)
            completed = _wait_for_task(server, str(task["task_id"]))
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["run_id"], task["run_id"])
            self.assertIn("Offline mock run completed", completed["result"]["content"])
            records = server.load_run(str(task["run_id"]))
            self.assertIn("final", {record["kind"] for record in records})

    def test_tasks_survive_server_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = HarnessServer(tmp)
            task = server.submit_mock_task("Inspect this project", stream=False)
            _wait_for_task(server, str(task["task_id"]))

            restored = HarnessServer(tmp)
            restored_task = restored.load_task(str(task["task_id"]))

            self.assertEqual(restored_task["status"], "completed")
            self.assertEqual(restored_task["run_id"], task["run_id"])

    def test_submit_model_task_requires_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = HarnessServer(tmp)

            with self.assertRaises(ValueError):
                server.submit_model_task("Inspect this project", provider=None)

    def test_server_reads_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"max_steps": 1}', encoding="utf-8")
            server = HarnessServer(tmp, config_path=str(config_path))

            result = server.run_mock_task("Inspect this project")

            self.assertEqual(result["steps"], 1)
            self.assertIn("Stopped after reaching max_steps=1.", result["content"])

    def test_server_can_resume_stopped_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"max_steps": 2}', encoding="utf-8")
            server = HarnessServer(tmp, config_path=str(config_path))

            stopped = server.run_mock_task("Inspect this project")
            resumed = server.run_mock_task("Inspect this project", resume_from=str(stopped["run_id"]))
            records = server.load_run(str(resumed["run_id"]))

            self.assertIn("Stopped after reaching max_steps=2.", stopped["content"])
            self.assertIn("Offline mock run completed", resumed["content"])
            self.assertIn("resumed_from", {record["kind"] for record in records})
            run_config = [record for record in records if record["kind"] == "run_config"][0]["payload"]
            self.assertEqual(run_config["resume_from"], stopped["run_id"])

    def test_server_can_submit_resume_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"max_steps": 2}', encoding="utf-8")
            server = HarnessServer(tmp, config_path=str(config_path))
            stopped = server.run_mock_task("Inspect this project")

            task = server.submit_resume_task(str(stopped["run_id"]), mock=True, stream=False)
            completed = _wait_for_task(server, str(task["task_id"]))

            self.assertEqual(completed["status"], "completed")
            self.assertIn("Offline mock run completed", completed["result"]["content"])

    def test_server_uses_tool_profile_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"tool_profile": "read-only"}', encoding="utf-8")
            server = HarnessServer(tmp, config_path=str(config_path))

            result = server.run_mock_task("Inspect this project")
            records = server.load_run(str(result["run_id"]))

            tool_profile_events = [record for record in records if record["kind"] == "tool_profile"]
            self.assertEqual(tool_profile_events[-1]["payload"], {"profile": "read-only"})

    def test_server_uses_command_profile_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"command_profile": "strict"}', encoding="utf-8")
            server = HarnessServer(tmp, config_path=str(config_path))

            result = server.run_mock_task("Inspect this project")
            records = server.load_run(str(result["run_id"]))

            events = [record for record in records if record["kind"] == "command_profile"]
            self.assertEqual(events[-1]["payload"], {"profile": "strict"})

    def test_task_events_include_trace_and_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = HarnessServer(tmp)
            task = server.submit_mock_task("Inspect this project", stream=True)

            events = server.task_events(str(task["task_id"]), poll_interval=0.01, timeout_seconds=5)

            names = [event["event"] for event in events]
            self.assertIn("task", names)
            self.assertIn("trace", names)
            self.assertEqual(names[-1], "done")
            trace_kinds = {
                event["data"]["kind"]
                for event in events
                if event["event"] == "trace" and isinstance(event["data"], dict)
            }
            self.assertIn("final", trace_kinds)


def _wait_for_task(server: HarnessServer, task_id: str) -> dict[str, object]:
    deadline = time.time() + 5
    while time.time() < deadline:
        task = server.load_task(task_id)
        if task["status"] in {"completed", "failed"}:
            return task
        time.sleep(0.01)
    raise AssertionError("Task did not finish")


if __name__ == "__main__":
    unittest.main()
