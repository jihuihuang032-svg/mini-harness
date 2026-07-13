from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.runtime.approval import ApprovalController
from harness.runtime.executor import CommandExecutor
from harness.runtime.policy import CommandPolicy
from harness.runtime.workspace import Workspace
from harness.tools import build_default_router


class WorkspaceAndToolTests(unittest.TestCase):
    def test_workspace_blocks_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            with self.assertRaises(PermissionError):
                workspace.resolve("../outside.txt")

    def test_read_write_tool_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            written = router.call("write_file", {"path": "notes/demo.txt", "content": "hello"})
            self.assertIs(written["ok"], True)

            read = router.call("read_file", {"path": "notes/demo.txt"})
            self.assertIs(read["ok"], True)
            self.assertEqual(read["content"], "hello")

    def test_read_file_can_select_line_range_with_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo.txt").write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
            workspace = Workspace(root)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call(
                "read_file",
                {"path": "demo.txt", "start_line": 2, "max_lines": 2, "line_numbers": True},
            )

            self.assertIs(result["ok"], True)
            self.assertEqual(result["content"], "2: two\n3: three")
            self.assertEqual(result["start_line"], 2)
            self.assertEqual(result["end_line"], 3)
            self.assertEqual(result["total_lines"], 5)
            self.assertEqual(result["truncated"], True)

    def test_read_file_rejects_invalid_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo.txt").write_text("one\n", encoding="utf-8")
            workspace = Workspace(root)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call("read_file", {"path": "demo.txt", "start_line": 0})

            self.assertIs(result["ok"], False)
            self.assertIn("start_line must be >= 1", result["error"])

    def test_router_exposes_tool_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            specs = router.specs()
            names = {spec["name"] for spec in specs}
            self.assertIn("read_file", names)
            self.assertIn("run_command", names)
            self.assertIn("git_status", names)
            self.assertTrue(all("args_schema" in spec for spec in specs))

    def test_router_validates_required_tool_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call("read_file", {})

            self.assertIs(result["ok"], False)
            self.assertIn("Invalid args", result["error"])
            self.assertIn("Missing required arg: path", result["error"])

    def test_router_validates_arg_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call("list_files", {"limit": "not-an-int"})

            self.assertIs(result["ok"], False)
            self.assertIn("Arg 'limit' must be integer", result["error"])

    def test_router_applies_schema_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("demo", encoding="utf-8")
            workspace = Workspace(root)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call("list_files", {})

            self.assertIs(result["ok"], True)
            self.assertIn("README.md", result["files"])

    def test_search_text_supports_case_insensitive_literal_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo.txt").write_text("Alpha\nbeta\n", encoding="utf-8")
            workspace = Workspace(root)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call("search_text", {"query": "alpha", "case_sensitive": False})

            self.assertIs(result["ok"], True)
            self.assertEqual(result["match_count"], 1)
            self.assertEqual(result["matches"][0]["line"], 1)
            self.assertEqual(result["matches"][0]["text"], "Alpha")

    def test_search_text_supports_regex_context_and_match_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo.txt").write_text("before\nfoo_1\nmiddle\nfoo_2\nafter\n", encoding="utf-8")
            workspace = Workspace(root)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call(
                "search_text",
                {
                    "query": r"foo_\d",
                    "regex": True,
                    "context_lines": 1,
                    "max_matches": 1,
                },
            )

            self.assertIs(result["ok"], True)
            self.assertEqual(result["match_count"], 1)
            self.assertEqual(result["truncated"], True)
            self.assertEqual(result["matches"][0]["line"], 2)
            self.assertEqual(
                result["matches"][0]["context"],
                [
                    {"line": 1, "text": "before"},
                    {"line": 2, "text": "foo_1"},
                    {"line": 3, "text": "middle"},
                ],
            )

    def test_search_text_rejects_invalid_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo.txt").write_text("demo\n", encoding="utf-8")
            workspace = Workspace(root)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call("search_text", {"query": "[", "regex": True})

            self.assertIs(result["ok"], False)
            self.assertIn("Invalid regex", result["error"])

    def test_read_only_profile_hides_and_denies_write_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000, tool_profile="read-only")

            names = {spec["name"] for spec in router.specs()}
            denied = router.call("write_file", {"path": "demo.txt", "content": "nope"})

            self.assertIn("read_file", names)
            self.assertIn("git_status", names)
            self.assertNotIn("write_file", names)
            self.assertNotIn("run_command", names)
            self.assertIs(denied["ok"], False)
            self.assertIn("not allowed by profile read-only", denied["error"])

    def test_review_profile_allows_commands_but_denies_file_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000, tool_profile="review")

            names = {spec["name"] for spec in router.specs()}

            self.assertIn("run_command", names)
            self.assertIn("git_status", names)
            self.assertIn("git_diff", names)
            self.assertNotIn("write_file", names)
            self.assertNotIn("apply_patch", names)

    def test_git_status_tool_runs_concise_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = Workspace(root)
            executor = RecordingExecutor(workspace)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call("git_status", {})

            self.assertIs(result["ok"], True)
            self.assertEqual(executor.commands[-1], "git status --short --branch")

    def test_git_status_tool_can_run_long_status_without_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = Workspace(root)
            executor = RecordingExecutor(workspace)
            router = build_default_router(workspace, executor, max_output_chars=1000)

            result = router.call("git_status", {"short": False, "branch": False})

            self.assertIs(result["ok"], True)
            self.assertEqual(executor.commands[-1], "git status")

    def test_command_policy_denies_destructive_command(self) -> None:
        policy = CommandPolicy.default()
        with self.assertRaises(PermissionError):
            policy.validate("git reset --hard HEAD")

    def test_command_policy_marks_dependency_install_for_approval(self) -> None:
        policy = CommandPolicy.default()
        assessment = policy.assess("pip install requests")
        self.assertEqual(assessment.level, "approval_required")
        self.assertEqual(policy.validate("pip install requests").level, "approval_required")

    def test_strict_command_policy_allows_read_only_commands(self) -> None:
        policy = CommandPolicy.default("strict")

        self.assertEqual(policy.validate("git diff").level, "low")
        self.assertEqual(policy.validate("python -m unittest discover -s tests").level, "low")

    def test_strict_command_policy_denies_unlisted_commands(self) -> None:
        policy = CommandPolicy.default("strict")

        with self.assertRaises(PermissionError):
            policy.validate("python script.py")

    def test_executor_refuses_approval_required_command_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            result = executor.run("pip install requests")
            self.assertIs(result["ok"], False)
            self.assertEqual(result["command"], "pip install requests")
            self.assertEqual(result["risk"], {"level": "approval_required", "reason": "installs dependencies"})
            self.assertEqual(result["approval"]["mode"], "never")

    def test_executor_returns_structured_denied_command_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)

            result = executor.run("git reset --hard HEAD")

            self.assertIs(result["ok"], False)
            self.assertEqual(result["command"], "git reset --hard HEAD")
            self.assertEqual(result["risk"], {"level": "denied", "reason": "destructive git reset"})
            self.assertEqual(result["approval"]["approved"], False)
            self.assertIn("denied by policy", result["approval"]["reason"])

    def test_approval_controller_on_request_accepts_yes(self) -> None:
        controller = ApprovalController("on-request", prompt=lambda _: "yes")
        assessment = CommandPolicy.default().assess("pip install requests")
        decision = controller.decide("pip install requests", assessment)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.mode, "on-request")

    def test_approval_controller_auto_approves(self) -> None:
        controller = ApprovalController("auto")
        assessment = CommandPolicy.default().assess("pip install requests")
        decision = controller.decide("pip install requests", assessment)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.mode, "auto")


class RecordingExecutor:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.commands: list[str] = []

    def run(self, command: str) -> dict[str, object]:
        self.commands.append(command)
        return {"ok": True, "command": command, "stdout": "", "stderr": "", "returncode": 0}


if __name__ == "__main__":
    unittest.main()
