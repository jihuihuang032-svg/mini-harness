from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from harness import __version__
from harness.cli import main


class CliTests(unittest.TestCase):
    def test_legacy_run_invocation_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--workspace", tmp, "--mock", "Inspect this project"])

            self.assertEqual(code, 0)
            self.assertIn("Offline mock run completed", stdout.getvalue())
            self.assertIn("changes:", stdout.getvalue())
            self.assertIn("run_id:", stdout.getvalue())
            self.assertTrue((Path(tmp) / ".harness" / "logs").exists())

    def test_list_and_show_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_stdout = io.StringIO()
            with redirect_stdout(run_stdout):
                self.assertEqual(main(["run", "--workspace", tmp, "--mock", "Inspect this project"]), 0)
            run_id = [line for line in run_stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            list_stdout = io.StringIO()
            with redirect_stdout(list_stdout):
                self.assertEqual(main(["list-runs", "--workspace", tmp]), 0)
            self.assertIn(run_id, list_stdout.getvalue())
            self.assertIn("changes", list_stdout.getvalue())
            self.assertIn("tools", list_stdout.getvalue())
            self.assertIn("tokens", list_stdout.getvalue())

            list_json_stdout = io.StringIO()
            with redirect_stdout(list_json_stdout):
                self.assertEqual(main(["list-runs", "--workspace", tmp, "--json"]), 0)
            self.assertIn('"usage": null', list_json_stdout.getvalue())
            self.assertIn('"tools"', list_json_stdout.getvalue())
            self.assertIn('"tool_names"', list_json_stdout.getvalue())

            show_stdout = io.StringIO()
            with redirect_stdout(show_stdout):
                self.assertEqual(main(["show-run", "--workspace", tmp, run_id]), 0)
            self.assertIn("run_started", show_stdout.getvalue())
            self.assertIn("final", show_stdout.getvalue())

            changes_stdout = io.StringIO()
            with redirect_stdout(changes_stdout):
                self.assertEqual(main(["show-changes", "--workspace", tmp, run_id]), 0)
            self.assertIn("changed_count:", changes_stdout.getvalue())

            summary_stdout = io.StringIO()
            with redirect_stdout(summary_stdout):
                self.assertEqual(main(["show-run", "--workspace", tmp, "--summary", run_id]), 0)
            self.assertIn("status: completed", summary_stdout.getvalue())
            self.assertIn("tool_calls: 1", summary_stdout.getvalue())
            self.assertIn("failed_tools: 0", summary_stdout.getvalue())

            summary_json_stdout = io.StringIO()
            with redirect_stdout(summary_json_stdout):
                self.assertEqual(main(["show-run", "--workspace", tmp, "--summary", "--json", run_id]), 0)
            summary_payload = json.loads(summary_json_stdout.getvalue())
            self.assertEqual(summary_payload["run_id"], run_id)
            self.assertEqual(summary_payload["status"], "completed")
            self.assertEqual(summary_payload["failed_tools"], 0)
            changes_json_stdout = io.StringIO()
            with redirect_stdout(changes_json_stdout):
                self.assertEqual(main(["show-changes", "--workspace", tmp, "--json", run_id]), 0)
            self.assertIn('"changed_count"', changes_json_stdout.getvalue())

    def test_list_providers(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["list-providers"]), 0)

        output = stdout.getvalue()
        self.assertIn("deepseek", output)
        self.assertIn("qwen", output)
        self.assertIn("kimi", output)

    def test_list_providers_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["list-providers", "--json"]), 0)

        self.assertIn('"name": "deepseek"', stdout.getvalue())

    def test_list_profiles_text_and_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["list-profiles"]), 0)

        self.assertIn("tool_profiles:", stdout.getvalue())
        self.assertIn("read-only", stdout.getvalue())
        self.assertIn("command_profiles:", stdout.getvalue())
        self.assertIn("strict", stdout.getvalue())

        json_stdout = io.StringIO()
        with redirect_stdout(json_stdout):
            self.assertEqual(main(["list-profiles", "--json"]), 0)

        payload = json.loads(json_stdout.getvalue())
        self.assertEqual(payload["tool_profiles"][0]["name"], "full")
        self.assertEqual(payload["command_profiles"][1]["name"], "strict")

    def test_top_level_help_lists_server(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["--help"]), 0)

        self.assertIn("doctor", stdout.getvalue())
        self.assertIn("eval", stdout.getvalue())
        self.assertIn("init", stdout.getvalue())
        self.assertIn("list-evals", stdout.getvalue())
        self.assertIn("list-profiles", stdout.getvalue())
        self.assertIn("list-tools", stdout.getvalue())
        self.assertIn("resume", stdout.getvalue())
        self.assertIn("server", stdout.getvalue())
        self.assertIn("show-eval", stdout.getvalue())
        self.assertIn("show-changes", stdout.getvalue())
        self.assertIn("show-checkpoint", stdout.getvalue())

    def test_top_level_version(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), f"mini-harness {__version__}")

    def test_package_module_entrypoint_runs_cli(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "harness", "--help"],
            text=True,
            capture_output=True,
            timeout=10,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("mini-harness", completed.stdout)
        self.assertIn("run", completed.stdout)

    def test_package_module_entrypoint_prints_version(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "harness", "--version"],
            text=True,
            capture_output=True,
            timeout=10,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.strip(), f"mini-harness {__version__}")

    def test_init_command_writes_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["init", "--workspace", tmp, "--env"])

            self.assertEqual(code, 0)
            self.assertIn("written:", stdout.getvalue())
            self.assertTrue((Path(tmp) / "harness.json").exists())
            self.assertTrue((Path(tmp) / ".env").exists())

    def test_list_tools_text_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["list-tools", "--workspace", tmp, "--tool-profile", "read-only"]), 0)

            output = stdout.getvalue()
            self.assertIn("tool_profile: read-only", output)
            self.assertIn("read_file", output)
            self.assertIn("git_status", output)
            self.assertNotIn("write_file", output)

            json_stdout = io.StringIO()
            with redirect_stdout(json_stdout):
                self.assertEqual(main(["list-tools", "--workspace", tmp, "--tool-profile", "read-only", "--json"]), 0)

            payload = json.loads(json_stdout.getvalue())
            names = {tool["name"] for tool in payload["tools"]}
            self.assertEqual(payload["tool_profile"], "read-only")
            self.assertIn("read_file", names)
            self.assertIn("search_text", names)
            self.assertNotIn("write_file", names)
            self.assertIn("args_schema", payload["tools"][0])

    def test_init_command_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["init", "--workspace", tmp, "--json"])

            self.assertEqual(code, 0)
            self.assertIn('"status": "written"', stdout.getvalue())

    def test_doctor_mock_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["doctor", "--workspace", tmp, "--mock"])

            self.assertEqual(code, 0)
            self.assertIn("ok\tworkspace", stdout.getvalue())
            self.assertIn("ok\tapi_key\tnot required for mock mode", stdout.getvalue())

    def test_doctor_json_reports_failure_for_missing_provider_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["doctor", "--workspace", tmp, "--provider", "deepseek", "--json"])

            self.assertEqual(code, 1)
            self.assertIn('"ok": false', stdout.getvalue())
            self.assertIn('"name": "api_key"', stdout.getvalue())

    def test_eval_mock_json_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.jsonl"
            cases.write_text(
                '{"id":"smoke","task":"Inspect this project","expect_contains":"Offline mock run completed"}\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["eval", "--workspace", tmp, "--mock", "--json", str(cases)])

            self.assertEqual(code, 0)
            self.assertIn('"passed": 1', stdout.getvalue())
            self.assertIn('"id": "smoke"', stdout.getvalue())
            self.assertIn('"eval_id"', stdout.getvalue())

    def test_eval_report_can_be_listed_and_shown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.jsonl"
            cases.write_text(
                '{"id":"smoke","task":"Inspect this project","expect_contains":"Offline mock run completed"}\n',
                encoding="utf-8",
            )
            eval_stdout = io.StringIO()
            with redirect_stdout(eval_stdout):
                self.assertEqual(main(["eval", "--workspace", tmp, "--mock", str(cases)]), 0)
            eval_id = [line for line in eval_stdout.getvalue().splitlines() if line.startswith("eval_id:")][0].split(":", 1)[1].strip()

            list_stdout = io.StringIO()
            with redirect_stdout(list_stdout):
                self.assertEqual(main(["list-evals", "--workspace", tmp]), 0)

            show_stdout = io.StringIO()
            with redirect_stdout(show_stdout):
                self.assertEqual(main(["show-eval", "--workspace", tmp, eval_id]), 0)

            self.assertIn(eval_id, list_stdout.getvalue())
            self.assertIn("status: passed", show_stdout.getvalue())
            self.assertIn("smoke", show_stdout.getvalue())

    def test_eval_mock_returns_nonzero_on_failed_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.jsonl"
            cases.write_text('{"id":"fail","task":"Inspect this project","expect_contains":"not present"}\n', encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["eval", "--workspace", tmp, "--mock", str(cases)])

            self.assertEqual(code, 1)
            self.assertIn("eval: 0/1 passed", stdout.getvalue())
            self.assertIn("missing: not present", stdout.getvalue())

    def test_streaming_mock_run_prints_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["run", "--workspace", tmp, "--mock", "--stream", "Inspect this project"])

            self.assertEqual(code, 0)
            output = stdout.getvalue()
            self.assertIn('"type": "plan"', output)
            self.assertIn("Offline mock run completed", output)
            self.assertIn("changes:", output)
            self.assertIn("run_id:", output)

    def test_run_max_steps_cli_override_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["run", "--workspace", tmp, "--mock", "--max-steps", "1", "Inspect this project"])

            self.assertEqual(code, 0)
            self.assertIn("Stopped after reaching max_steps=1.", stdout.getvalue())

    def test_run_rejects_invalid_budget_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(["run", "--workspace", tmp, "--mock", "--max-run-tokens", "-1", "Inspect this project"])

            self.assertEqual(code, 1)
            self.assertIn("--max-run-tokens must be >= 0", stderr.getvalue())
    def test_run_reads_explicit_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"max_steps": 1, "approval": "auto"}', encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["run", "--workspace", tmp, "--config", str(config_path), "--mock", "Inspect this project"])

            self.assertEqual(code, 0)
            self.assertIn("Stopped after reaching max_steps=1.", stdout.getvalue())

    def test_run_can_read_task_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "task.md"
            task_path.write_text("Inspect this project\n", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["run", "--workspace", tmp, "--mock", "--json", "--task-file", str(task_path)])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertIn("Offline mock run completed", payload["content"])
            self.assertIn("run_id", payload)

    def test_run_rejects_task_and_task_file_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "task.md"
            task_path.write_text("Inspect this project\n", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(["run", "--workspace", tmp, "--mock", "--task-file", str(task_path), "Inspect this project"])

            self.assertEqual(code, 1)
            self.assertIn("either a task argument or --task-file", stderr.getvalue())
    def test_run_json_outputs_machine_readable_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["run", "--workspace", tmp, "--mock", "--json", "Inspect this project"])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertIn("Offline mock run completed", payload["content"])
            self.assertEqual(payload["steps"], 4)
            self.assertIn("run_id", payload)
            self.assertEqual(payload["changes"]["changed_count"], 0)

    def test_run_json_rejects_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(["run", "--workspace", tmp, "--mock", "--stream", "--json", "Inspect this project"])

            self.assertEqual(code, 1)
            self.assertIn("--stream cannot be combined with --json", stderr.getvalue())

    def test_cli_tool_profile_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["run", "--workspace", tmp, "--mock", "--tool-profile", "read-only", "Inspect this project"])
            run_id = [line for line in stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            show_stdout = io.StringIO()
            with redirect_stdout(show_stdout):
                self.assertEqual(main(["show-run", "--workspace", tmp, run_id]), 0)

            self.assertEqual(code, 0)
            self.assertIn("tool_profile", show_stdout.getvalue())
            self.assertIn("read-only", show_stdout.getvalue())

    def test_cli_run_config_snapshot_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "run",
                        "--workspace",
                        tmp,
                        "--mock",
                        "--stream",
                        "--tool-profile",
                        "read-only",
                        "--command-profile",
                        "strict",
                        "Inspect this project",
                    ]
                )
            run_id = [line for line in stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            show_stdout = io.StringIO()
            with redirect_stdout(show_stdout):
                self.assertEqual(main(["show-run", "--workspace", tmp, "--json", run_id]), 0)
            records = json.loads(show_stdout.getvalue())
            run_config = [record for record in records if record["kind"] == "run_config"][0]["payload"]

            self.assertEqual(code, 0)
            self.assertEqual(run_config["mode"], "mock")
            self.assertTrue(run_config["stream"])
            self.assertEqual(run_config["tool_profile"], "read-only")
            self.assertEqual(run_config["command_profile"], "strict")
            self.assertEqual(run_config["tool_count"], 5)
            self.assertEqual(run_config["tool_names"], ["git_diff", "git_status", "list_files", "read_file", "search_text"])
            self.assertEqual(len(run_config["tool_schema_sha256"]), 64)
            self.assertEqual(len(run_config["system_prompt_sha256"]), 64)
            self.assertGreater(run_config["system_prompt_chars"], 0)
            self.assertNotIn("api_key", run_config)
            self.assertNotIn("system_prompt", run_config)

    def test_cli_command_profile_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["run", "--workspace", tmp, "--mock", "--command-profile", "strict", "Inspect this project"])
            run_id = [line for line in stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            show_stdout = io.StringIO()
            with redirect_stdout(show_stdout):
                self.assertEqual(main(["show-run", "--workspace", tmp, run_id]), 0)

            self.assertEqual(code, 0)
            self.assertIn("command_profile", show_stdout.getvalue())
            self.assertIn("strict", show_stdout.getvalue())

    def test_show_checkpoint_after_mock_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["run", "--workspace", tmp, "--mock", "Inspect this project"]), 0)
            run_id = [line for line in stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            show_stdout = io.StringIO()
            with redirect_stdout(show_stdout):
                self.assertEqual(main(["show-checkpoint", "--workspace", tmp, run_id]), 0)

            show_json_stdout = io.StringIO()
            with redirect_stdout(show_json_stdout):
                self.assertEqual(main(["show-checkpoint", "--workspace", tmp, "--json", run_id]), 0)

            self.assertIn("status: completed", show_stdout.getvalue())
            self.assertIn("messages:", show_stdout.getvalue())
            self.assertIn('"status": "completed"', show_json_stdout.getvalue())
            self.assertIn('"messages"', show_json_stdout.getvalue())

    def test_resume_continues_from_stopped_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"max_steps": 2}', encoding="utf-8")
            first_stdout = io.StringIO()
            with redirect_stdout(first_stdout):
                self.assertEqual(
                    main(["run", "--workspace", tmp, "--config", str(config_path), "--mock", "Inspect this project"]),
                    0,
                )
            first_run_id = [line for line in first_stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            resume_stdout = io.StringIO()
            with redirect_stdout(resume_stdout):
                self.assertEqual(
                    main(["resume", "--workspace", tmp, "--config", str(config_path), "--mock", first_run_id]),
                    0,
                )
            resumed_run_id = [line for line in resume_stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            show_stdout = io.StringIO()
            with redirect_stdout(show_stdout):
                self.assertEqual(main(["show-run", "--workspace", tmp, resumed_run_id]), 0)

            checkpoint_stdout = io.StringIO()
            with redirect_stdout(checkpoint_stdout):
                self.assertEqual(main(["show-checkpoint", "--workspace", tmp, resumed_run_id]), 0)

            self.assertNotEqual(first_run_id, resumed_run_id)
            self.assertIn("Offline mock run completed", resume_stdout.getvalue())
            self.assertIn(f"resumed_from: {first_run_id}", resume_stdout.getvalue())
            self.assertIn("resumed_from", show_stdout.getvalue())
            self.assertIn("status: completed", checkpoint_stdout.getvalue())

            show_json_stdout = io.StringIO()
            with redirect_stdout(show_json_stdout):
                self.assertEqual(main(["show-run", "--workspace", tmp, "--json", resumed_run_id]), 0)
            records = json.loads(show_json_stdout.getvalue())
            run_config = [record for record in records if record["kind"] == "run_config"][0]["payload"]
            self.assertEqual(run_config["resume_from"], first_run_id)

    def test_resume_json_outputs_machine_readable_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"max_steps": 2}', encoding="utf-8")
            first_stdout = io.StringIO()
            with redirect_stdout(first_stdout):
                self.assertEqual(
                    main(["run", "--workspace", tmp, "--config", str(config_path), "--mock", "Inspect this project"]),
                    0,
                )
            first_run_id = [line for line in first_stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            resume_stdout = io.StringIO()
            with redirect_stdout(resume_stdout):
                code = main(["resume", "--workspace", tmp, "--config", str(config_path), "--mock", "--json", first_run_id])

            payload = json.loads(resume_stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["resumed_from"], first_run_id)
            self.assertIn("Offline mock run completed", payload["content"])
            self.assertNotEqual(payload["run_id"], first_run_id)

    def test_resume_rejects_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_stdout = io.StringIO()
            with redirect_stdout(first_stdout):
                self.assertEqual(main(["run", "--workspace", tmp, "--mock", "Inspect this project"]), 0)
            first_run_id = [line for line in first_stdout.getvalue().splitlines() if line.startswith("run_id:")][0].split(":", 1)[1].strip()

            resume_stdout = io.StringIO()
            resume_stderr = io.StringIO()
            with redirect_stdout(resume_stdout), redirect_stderr(resume_stderr):
                code = main(["resume", "--workspace", tmp, "--mock", first_run_id])

            self.assertEqual(code, 1)
            self.assertIn("Checkpoint is already completed", resume_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

