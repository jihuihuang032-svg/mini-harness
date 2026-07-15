from __future__ import annotations

import json
import tempfile
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import time
import unittest
from pathlib import Path

from harness.server import HarnessServer, create_handler


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

    def test_preview_run_returns_non_secret_runtime_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"tool_profile": "read-only", "command_profile": "strict"}', encoding="utf-8")
            server = HarnessServer(tmp, config_path=str(config_path))

            preview = server.preview_run(stream=True)

            self.assertEqual(preview["mode"], "mock")
            self.assertTrue(preview["stream"])
            self.assertEqual(preview["workspace"], str(Path(tmp).resolve()))
            self.assertEqual(preview["tool_profile"], "read-only")
            self.assertEqual(preview["command_profile"], "strict")
            self.assertEqual(preview["approval"], "never")
            self.assertEqual(preview["tool_count"], 5)
            self.assertEqual(preview["tool_names"], ["git_diff", "git_status", "list_files", "read_file", "search_text"])
            self.assertEqual(len(preview["tool_schema_sha256"]), 64)
            self.assertEqual(len(preview["system_prompt_sha256"]), 64)
            self.assertIn("tools", preview)
            self.assertIn("args_schema", preview["tools"][0])
            self.assertNotIn("api_key", preview)
            self.assertNotIn("system_prompt", preview)

    def test_preview_run_matches_mock_run_trace_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"tool_profile": "read-only", "command_profile": "strict"}', encoding="utf-8")
            server = HarnessServer(tmp, config_path=str(config_path))

            preview = server.preview_run(stream=True)
            result = server.run_mock_task("Inspect this project", stream=True)
            records = server.load_run(str(result["run_id"]))
            traced = [record for record in records if record["kind"] == "run_config"][0]["payload"]

            for key in (
                "mode",
                "stream",
                "tool_profile",
                "command_profile",
                "approval",
                "workspace",
                "max_steps",
                "tool_count",
                "tool_names",
                "tool_schema_sha256",
                "system_prompt_sha256",
            ):
                self.assertEqual(preview[key], traced[key])

    def test_http_preview_run_route_returns_runtime_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"tool_profile": "read-only", "command_profile": "strict"}', encoding="utf-8")
            server = HarnessServer(tmp, config_path=str(config_path))

            status, payload = _request_json(server, "/preview-run?mock=true&stream=true")

            self.assertEqual(status, 200)
            self.assertEqual(payload["mode"], "mock")
            self.assertTrue(payload["stream"])
            self.assertEqual(payload["tool_profile"], "read-only")
            self.assertEqual(payload["command_profile"], "strict")
            self.assertEqual(payload["tool_count"], 5)
            self.assertIn("tools", payload)
            self.assertNotIn("api_key", payload)
            self.assertNotIn("system_prompt", payload)

    def test_http_preview_run_rejects_invalid_bool_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = HarnessServer(tmp)

            status, payload = _request_json(server, "/preview-run?mock=maybe")

            self.assertEqual(status, 400)
            self.assertIn("mock", payload["error"])
            self.assertIn("boolean", payload["error"])

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


def _request_json(app: HarnessServer, path: str) -> tuple[int, dict[str, object]]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        conn = HTTPConnection(host, port, timeout=5)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise AssertionError("Expected JSON object response")
            return response.status, payload
        finally:
            conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

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
