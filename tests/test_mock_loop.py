from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from harness.agent import Agent
from harness.config import HarnessConfig
from harness.messages import Message
from harness.models.mock import MockModelClient
from harness.runtime.executor import CommandExecutor
from harness.runtime.logger import RunLogger
from harness.runtime.policy import CommandPolicy
from harness.runtime.workspace import Workspace
from harness.tools import build_default_router


class MockLoopTests(unittest.TestCase):
    def test_mock_model_runs_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("demo", encoding="utf-8")
            config = HarnessConfig.offline(str(root))
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)

            result = Agent(config, MockModelClient(), router, logger, workspace=workspace).run("offline smoke test")

            self.assertEqual(result.steps, 4)
            self.assertIn("Offline mock run completed", result.content)
            self.assertTrue(logger.path.exists())
            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            kinds = {record["kind"] for record in records}
            self.assertIn("context_summary", kinds)
            self.assertIn("plan_updated", kinds)
            self.assertIn("run_finished", kinds)
            request = [record for record in records if record["kind"] == "model_request"][0]["payload"]
            self.assertNotIn("messages", request)
            self.assertEqual(len(request["messages_sha256"]), 64)
            self.assertEqual(request["trace_messages"], False)
            response = [record for record in records if record["kind"] == "model_response"][0]["payload"]
            self.assertNotIn("content", response)
            self.assertGreater(response["content_chars"], 0)
            self.assertEqual(len(response["content_sha256"]), 64)
            self.assertEqual(response["trace_model_responses"], False)
            final = [record for record in records if record["kind"] == "final"][0]
            self.assertEqual(final["payload"]["plan"]["counts"]["completed"], 2)
            finished = [record for record in records if record["kind"] == "run_finished"][-1]
            self.assertEqual(finished["payload"]["status"], "completed")

    def test_agent_logs_stopped_when_max_steps_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig.offline(str(root))
            config = replace(config, max_steps=1)
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)

            result = Agent(config, MockModelClient(), router, logger, workspace=workspace).run("offline smoke test")

            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            finished = [record for record in records if record["kind"] == "run_finished"][-1]
            self.assertEqual(result.content, "Stopped after reaching max_steps=1.")
            self.assertEqual(finished["payload"]["status"], "stopped")
            self.assertEqual(finished["payload"]["reason"], "max_steps")

    def test_agent_can_trace_full_model_request_messages_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(HarnessConfig.offline(str(root)), trace_messages=True)
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)

            Agent(config, MockModelClient(), router, logger, workspace=workspace).run("offline smoke test")

            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            request = [record for record in records if record["kind"] == "model_request"][0]["payload"]
            self.assertIn("messages", request)
            self.assertEqual(request["trace_messages"], True)
            self.assertEqual(request["messages"][2]["content"], "offline smoke test")

    def test_agent_can_trace_full_model_response_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(HarnessConfig.offline(str(root)), trace_model_responses=True)
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)

            Agent(config, MetadataModel(), router, logger, workspace=workspace).run("metadata smoke test")

            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            response = [record for record in records if record["kind"] == "model_response"][0]["payload"]
            self.assertIn("content", response)
            self.assertEqual(response["trace_model_responses"], True)
            self.assertEqual(json.loads(response["content"])["content"], "done")

    def test_agent_logs_failed_when_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig.offline(str(root))
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)

            with self.assertRaises(RuntimeError):
                Agent(config, RaisingModel(), router, logger, workspace=workspace).run("offline smoke test")

            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            finished = [record for record in records if record["kind"] == "run_finished"][-1]
            self.assertEqual(finished["payload"]["status"], "failed")
            self.assertIn("model unavailable", finished["payload"]["error"])

    def test_agent_logs_model_response_metadata_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig.offline(str(root))
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)

            result = Agent(config, MetadataModel(), router, logger, workspace=workspace).run("metadata smoke test")

            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            metadata = [record for record in records if record["kind"] == "model_response_metadata"][0]["payload"]
            self.assertEqual(result.content, "done")
            self.assertEqual(metadata["step"], 1)
            self.assertEqual(metadata["usage"]["total_tokens"], 3)

    def test_agent_stops_when_token_budget_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(HarnessConfig.offline(str(root)), max_run_tokens=2)
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)

            result = Agent(config, MetadataModel(), router, logger, workspace=workspace).run("metadata smoke test")

            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            kinds = {record["kind"] for record in records}
            finished = [record for record in records if record["kind"] == "run_finished"][-1]
            self.assertIn("token_budget_exceeded", kinds)
            self.assertEqual(result.steps, 1)
            self.assertIn("max_run_tokens=2", result.content)
            self.assertEqual(finished["payload"]["reason"], "token_budget")
            self.assertEqual(finished["payload"]["total_tokens"], 3)

    def test_agent_logs_command_audit_for_denied_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig.offline(str(root))
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)

            Agent(config, DeniedCommandModel(), router, logger, workspace=workspace).run("try denied command")

            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            audit = [record for record in records if record["kind"] == "command_audit"][0]["payload"]
            self.assertEqual(audit["tool"], "run_command")
            self.assertEqual(audit["command"], "git reset --hard HEAD")
            self.assertEqual(audit["risk"], {"level": "denied", "reason": "destructive git reset"})
            self.assertEqual(audit["approval"]["approved"], False)
            self.assertIs(audit["ok"], False)


class RaisingModel:
    def complete(self, messages: list[Message]) -> str:
        raise RuntimeError("model unavailable")

    def stream_complete(self, messages: list[Message]) -> Iterator[str]:
        raise RuntimeError("model unavailable")
        yield ""


class MetadataModel:
    def __init__(self) -> None:
        self.last_response_metadata: dict[str, object] = {}

    def complete(self, messages: list[Message]) -> str:
        self.last_response_metadata = {"usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
        return json.dumps({"type": "final", "content": "done"})

    def stream_complete(self, messages: list[Message]) -> Iterator[str]:
        yield self.complete(messages)


class DeniedCommandModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message]) -> str:
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "type": "tool_call",
                    "tool": "run_command",
                    "args": {"command": "git reset --hard HEAD"},
                }
            )
        return json.dumps({"type": "final", "content": "done"})

    def stream_complete(self, messages: list[Message]) -> Iterator[str]:
        yield self.complete(messages)


if __name__ == "__main__":
    unittest.main()
