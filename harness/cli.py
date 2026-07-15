"""Command-line entry point for Mini Harness.

This module owns argument parsing and dispatch for the CLI subcommands: run,
resume, eval, history inspection, provider/profile listing, initialization,
doctor checks, and the lightweight server.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from harness import __version__
from harness.agent import Agent, render_system_prompt
from harness.config import HarnessConfig
from harness.doctor import run_doctor
from harness.eval_runner import evaluate_case, load_eval_cases
from harness.init_project import init_workspace
from harness.models.mock import MockModelClient
from harness.models.openai_compatible import OpenAICompatibleClient
from harness.models.providers import PROVIDER_PRESETS, provider_names
from harness.runtime.approval import ApprovalController
from harness.runtime.change_tracker import WorkspaceChangeTracker
from harness.runtime.checkpoint import RunCheckpointStore
from harness.runtime.eval_store import EvalStore
from harness.runtime.executor import CommandExecutor
from harness.runtime.logger import RunLogger
from harness.runtime.policy import CommandPolicy
from harness.runtime.run_config import run_config_snapshot
from harness.runtime.run_store import RunStore
from harness.runtime.workspace import Workspace
from harness.server import serve
from harness.tools import build_default_router


# Supported subcommands. Used to detect legacy shorthand invocations.
COMMANDS = {
    "doctor",
    "eval",
    "init",
    "list-evals",
    "list-profiles",
    "list-runs",
    "list-tools",
    "resume",
    "run",
    "show-eval",
    "show-run",
    "show-changes",
    "show-checkpoint",
    "list-providers",
    "server",
}


PROFILE_DESCRIPTIONS: dict[str, list[dict[str, object]]] = {
    "tool_profiles": [
        {"name": "full", "description": "Expose every built-in tool."},
        {
            "name": "review",
            "description": "Allow read/search tools, git inspection, and run_command, but block file writes and patches.",
        },
        {"name": "read-only", "description": "Allow only list_files, read_file, search_text, git_status, and git_diff."},
    ],
    "command_profiles": [
        {
            "name": "default",
            "description": "Deny destructive commands and require approval for dependency installs, commits, pushes, and piped downloads.",
        },
        {
            "name": "strict",
            "description": "Apply default risk checks, then allow only a narrow read-only/check command set.",
        },
    ],
}


def _configure_stdio() -> None:
    """Configure UTF-8 stdio where supported."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and return a process exit code."""
    _configure_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    # Shorthand: harness "do something" is treated as harness run "do something".
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv = ["run", *argv]
    # Options before a subcommand are treated as run options, except top-level help/version.
    if argv and argv[0].startswith("-") and argv[0] not in {"-h", "--help", "--version"}:
        argv = ["run", *argv]

    parser = argparse.ArgumentParser(prog="mini-harness")
    parser.add_argument("--version", action="version", version=f"mini-harness {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize harness config files in a workspace.")
    init_parser.add_argument("--workspace", help="Workspace directory. Defaults to current directory.")
    init_parser.add_argument("--env", action="store_true", help="Also create a .env template.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing generated files.")
    init_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    doctor_parser = subparsers.add_parser("doctor", help="Check workspace and provider configuration.")
    _add_common_workspace_arg(doctor_parser)
    doctor_parser.add_argument("--mock", action="store_true", help="Skip real provider API-key checks.")
    doctor_parser.add_argument("--provider", choices=provider_names(), help="Check a built-in provider preset.")
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    eval_parser = subparsers.add_parser("eval", help="Run JSONL evaluation cases through the harness.")
    _add_common_workspace_arg(eval_parser)
    eval_parser.add_argument("cases", help="Path to JSONL eval cases.")
    eval_parser.add_argument("--mock", action="store_true", help="Use deterministic offline model.")
    eval_parser.add_argument("--provider", choices=provider_names(), help="Use a built-in OpenAI-compatible provider preset.")
    eval_parser.add_argument("--tool-profile", choices=["full", "review", "read-only"], help="Override tool profile.")
    eval_parser.add_argument("--command-profile", choices=["default", "strict"], help="Override command profile.")
    eval_parser.add_argument("--approval", choices=["never", "on-request", "auto"], help="Override approval mode.")
    eval_parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed eval case.")
    eval_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    _add_budget_override_args(eval_parser)

    list_evals_parser = subparsers.add_parser("list-evals", help="List saved eval reports.")
    _add_common_workspace_arg(list_evals_parser)
    list_evals_parser.add_argument("--limit", type=int, default=20, help="Maximum number of eval reports to show.")
    list_evals_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    run_parser = subparsers.add_parser("run", help="Run a coding-agent task.")
    _add_common_workspace_arg(run_parser)
    run_parser.add_argument("task", nargs="?", help="Task for the coding agent. Omit when using --task-file.")
    run_parser.add_argument("--task-file", help="Read the task prompt from a UTF-8 text file.")
    run_parser.add_argument("--mock", action="store_true", help="Use a deterministic offline model instead of calling an API.")
    run_parser.add_argument("--provider", choices=provider_names(), help="Use a built-in OpenAI-compatible provider preset.")
    run_parser.add_argument("--stream", action="store_true", help="Stream model output chunks while still parsing the final JSON action.")
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    _add_budget_override_args(run_parser)
    run_parser.add_argument(
        "--tool-profile",
        choices=["full", "review", "read-only"],
        help="Limit the tools exposed to the model. Overrides config/env.",
    )
    run_parser.add_argument(
        "--command-profile",
        choices=["default", "strict"],
        help="Limit shell commands run_command may execute. Overrides config/env.",
    )
    run_parser.add_argument(
        "--approval",
        choices=["never", "on-request", "auto"],
        help="How to handle commands classified as approval_required. Overrides config/env.",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume a run from its latest checkpoint.")
    _add_common_workspace_arg(resume_parser)
    resume_parser.add_argument("run_id", help="Run id whose checkpoint should be resumed.")
    resume_parser.add_argument("--mock", action="store_true", help="Use a deterministic offline model instead of calling an API.")
    resume_parser.add_argument("--provider", choices=provider_names(), help="Use a built-in OpenAI-compatible provider preset.")
    resume_parser.add_argument("--stream", action="store_true", help="Stream model output chunks while still parsing the final JSON action.")
    resume_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    _add_budget_override_args(resume_parser)
    resume_parser.add_argument(
        "--tool-profile",
        choices=["full", "review", "read-only"],
        help="Limit the tools exposed to the model. Overrides config/env.",
    )
    resume_parser.add_argument(
        "--command-profile",
        choices=["default", "strict"],
        help="Limit shell commands run_command may execute. Overrides config/env.",
    )
    resume_parser.add_argument(
        "--approval",
        choices=["never", "on-request", "auto"],
        help="How to handle commands classified as approval_required. Overrides config/env.",
    )

    list_parser = subparsers.add_parser("list-runs", help="List recent run traces.")
    _add_common_workspace_arg(list_parser)
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of runs to show.")
    list_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    tools_parser = subparsers.add_parser("list-tools", help="List tool schemas for the active tool profile.")
    _add_common_workspace_arg(tools_parser)
    tools_parser.add_argument("--tool-profile", choices=["full", "review", "read-only"], help="Override tool profile.")
    tools_parser.add_argument("--command-profile", choices=["default", "strict"], help="Override command profile.")
    tools_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    show_parser = subparsers.add_parser("show-run", help="Show one run trace.")
    _add_common_workspace_arg(show_parser)
    show_parser.add_argument("run_id", help="Run id to load.")
    show_parser.add_argument("--json", action="store_true", help="Print raw JSON records.")
    show_parser.add_argument("--summary", action="store_true", help="Print a compact diagnostic summary instead of the full trace.")

    changes_parser = subparsers.add_parser("show-changes", help="Show workspace changes for one run.")
    _add_common_workspace_arg(changes_parser)
    changes_parser.add_argument("run_id", help="Run id to load changes for.")
    changes_parser.add_argument("--json", action="store_true", help="Print raw JSON changes.")

    checkpoint_parser = subparsers.add_parser("show-checkpoint", help="Show the latest checkpoint for one run.")
    _add_common_workspace_arg(checkpoint_parser)
    checkpoint_parser.add_argument("run_id", help="Run id to load checkpoint for.")
    checkpoint_parser.add_argument("--json", action="store_true", help="Print raw JSON checkpoint.")

    show_eval_parser = subparsers.add_parser("show-eval", help="Show one saved eval report.")
    _add_common_workspace_arg(show_eval_parser)
    show_eval_parser.add_argument("eval_id", help="Eval report id to load.")
    show_eval_parser.add_argument("--json", action="store_true", help="Print raw JSON report.")

    provider_parser = subparsers.add_parser("list-providers", help="List built-in model provider presets.")
    provider_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    profiles_parser = subparsers.add_parser("list-profiles", help="List built-in tool and command profiles.")
    profiles_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    server_parser = subparsers.add_parser("server", help="Start the lightweight HTTP API server.")
    _add_common_workspace_arg(server_parser)
    server_parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Defaults to 127.0.0.1.")
    server_parser.add_argument("--port", type=int, default=8765, help="Port to bind. Defaults to 8765.")

    if argv in (["-h"], ["--help"]):
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "run":
            return _run_task(args)
        if args.command == "resume":
            return _resume_task(args)
        if args.command == "eval":
            return _eval(args)
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "init":
            return _init(args)
        if args.command == "list-evals":
            return _list_evals(args)
        if args.command == "list-runs":
            return _list_runs(args)
        if args.command == "list-tools":
            return _list_tools(args)
        if args.command == "show-eval":
            return _show_eval(args)
        if args.command == "show-run":
            return _show_run(args)
        if args.command == "show-changes":
            return _show_changes(args)
        if args.command == "show-checkpoint":
            return _show_checkpoint(args)
        if args.command == "list-providers":
            return _list_providers(args)
        if args.command == "list-profiles":
            return _list_profiles(args)
        if args.command == "server":
            return _server(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


def _add_common_workspace_arg(parser: argparse.ArgumentParser) -> None:
    """Add common workspace and config arguments."""
    parser.add_argument("--workspace", help="Workspace directory. Defaults to HARNESS_WORKSPACE or current directory.")
    parser.add_argument("--config", help="Path to harness.json. Defaults to <workspace>/harness.json or ./harness.json.")


def _add_budget_override_args(parser: argparse.ArgumentParser) -> None:
    """Add per-command run budget override arguments."""
    parser.add_argument("--max-steps", type=int, help="Override the maximum agent loop steps for this command.")
    parser.add_argument("--max-run-tokens", type=int, help="Override the provider-reported token budget for this command; 0 disables it.")


def _init(args: argparse.Namespace) -> int:
    """Handle the init subcommand."""
    results = init_workspace(args.workspace, include_env=args.env, force=args.force)
    if args.json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
        return 0
    for result in results:
        print(f"{result.status}: {result.path}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """Handle the doctor subcommand."""
    report = run_doctor(args.workspace, args.config, args.provider, mock=args.mock)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        for check in report.checks:
            status = "ok" if check.ok else "fail"
            print(f"{status}\t{check.name}\t{check.message}")
    return 0 if report.ok else 1


def _run_task(args: argparse.Namespace) -> int:
    """Handle the run subcommand."""
    if args.stream and args.json:
        raise ValueError("--stream cannot be combined with --json because streamed chunks would corrupt JSON output.")
    task = _resolve_task_input(args.task, args.task_file)
    result = _execute_agent_task(
        task=task,
        workspace_arg=args.workspace,
        config_arg=args.config,
        mock=args.mock,
        provider=args.provider,
        stream=args.stream,
        approval_override=args.approval,
        tool_profile_override=args.tool_profile,
        command_profile_override=args.command_profile,
        max_steps_override=args.max_steps,
        max_run_tokens_override=args.max_run_tokens,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.stream:
        print()
    print(result["content"])
    print(f"changes: {result['changes']['changed_count']}")
    print(f"run_id: {result['run_id']}")
    return 0


def _resolve_task_input(task: str | None, task_file: str | None) -> str:
    if task and task_file:
        raise ValueError("Provide either a task argument or --task-file, not both.")
    if task_file:
        path = Path(task_file)
        if not path.exists():
            raise ValueError(f"Task file not found: {task_file}")
        if not path.is_file():
            raise ValueError(f"Task file is not a file: {task_file}")
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Task file is empty: {task_file}")
        return content
    if task:
        return task
    raise ValueError("Task is required. Provide a task argument or --task-file.")


def _resume_task(args: argparse.Namespace) -> int:
    """Handle the resume subcommand."""
    if args.stream and args.json:
        raise ValueError("--stream cannot be combined with --json because streamed chunks would corrupt JSON output.")
    result = _execute_agent_task(
        task="",
        workspace_arg=args.workspace,
        config_arg=args.config,
        mock=args.mock,
        provider=args.provider,
        stream=args.stream,
        approval_override=args.approval,
        tool_profile_override=args.tool_profile,
        command_profile_override=args.command_profile,
        max_steps_override=args.max_steps,
        max_run_tokens_override=args.max_run_tokens,
        resume_from=args.run_id,
    )
    if args.json:
        print(json.dumps({**result, "resumed_from": args.run_id}, ensure_ascii=False, indent=2))
        return 0
    if args.stream:
        print()
    print(result["content"])
    print(f"changes: {result['changes']['changed_count']}")
    print(f"run_id: {result['run_id']}")
    print(f"resumed_from: {args.run_id}")
    return 0


def _execute_agent_task(
    task: str,
    workspace_arg: str | None,
    config_arg: str | None,
    mock: bool,
    provider: str | None,
    stream: bool,
    approval_override: str | None,
    tool_profile_override: str | None,
    command_profile_override: str | None,
    max_steps_override: int | None = None,
    max_run_tokens_override: int | None = None,
    resume_from: str | None = None,
) -> dict[str, object]:
    """Build runtime dependencies, execute the agent, and return run metadata."""
    config = (
        HarnessConfig.offline(workspace_arg, config_arg)
        if mock
        else HarnessConfig.from_env(workspace_arg, provider, config_arg)
    )
    config = _apply_budget_overrides(config, max_steps_override, max_run_tokens_override)
    workspace = Workspace(config.workspace)
    command_profile = command_profile_override or config.command_profile
    policy = CommandPolicy.default(command_profile)
    approval = ApprovalController(approval_override or config.approval)
    executor = CommandExecutor(workspace, policy, config.timeout_seconds, config.max_tool_output_chars, approval)
    tool_profile = tool_profile_override or config.tool_profile
    router = build_default_router(workspace, executor, config.max_tool_output_chars, tool_profile=tool_profile)
    tool_specs = router.specs()
    system_prompt = render_system_prompt(router)
    checkpoint_store = RunCheckpointStore(workspace.checkpoints_dir)
    loaded_checkpoint = checkpoint_store.load_state(resume_from) if resume_from is not None else None
    if loaded_checkpoint is not None and loaded_checkpoint.status == "completed":
        raise ValueError(f"Checkpoint is already completed: {resume_from}")
    logger = RunLogger(workspace.logs_dir)
    mode = "mock" if mock else "model"
    logger.event(
        "run_config",
        run_config_snapshot(
            config,
            mode=mode,
            stream=stream,
            tool_profile=tool_profile,
            command_profile=command_profile,
            approval=approval.mode,
            tool_specs=tool_specs,
            system_prompt=system_prompt,
            resume_from=resume_from,
        ),
    )
    logger.event("tool_profile", {"profile": tool_profile})
    logger.event("command_profile", {"profile": command_profile})
    tracker = WorkspaceChangeTracker(workspace)
    before = tracker.capture()
    logger.event("workspace_snapshot", {"phase": "before", "file_count": len(before.files)})
    completed_steps = loaded_checkpoint.step if loaded_checkpoint is not None else 0
    model = MockModelClient(calls=completed_steps) if mock else OpenAICompatibleClient(config, tool_specs)
    callback = _print_stream_chunk if stream else None
    try:
        agent = Agent(
            config=config,
            model=model,
            tools=router,
            logger=logger,
            workspace=workspace,
            plan=loaded_checkpoint.plan if loaded_checkpoint is not None else None,
            checkpoint_store=checkpoint_store,
            stream=stream,
            stream_callback=callback,
        )
        if loaded_checkpoint is not None:
            result = agent.resume(
                task=loaded_checkpoint.task,
                messages=loaded_checkpoint.messages,
                completed_steps=loaded_checkpoint.step,
                source_run_id=loaded_checkpoint.run_id,
            )
        else:
            result = agent.run(task)
    finally:
        after = tracker.capture()
        changes = tracker.compare(before, after)
        changes_path = tracker.save(logger.run_id, changes)
        logger.event(
            "workspace_changes",
            {
                **changes.to_dict(),
                "path": workspace.relative(changes_path),
            },
        )
    return {
        "run_id": logger.run_id,
        "content": result.content,
        "steps": result.steps,
        "changes": changes.to_dict(),
    }


def _apply_budget_overrides(
    config: HarnessConfig,
    max_steps: int | None,
    max_run_tokens: int | None,
) -> HarnessConfig:
    updates: dict[str, int] = {}
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be >= 1.")
        updates["max_steps"] = max_steps
    if max_run_tokens is not None:
        if max_run_tokens < 0:
            raise ValueError("--max-run-tokens must be >= 0.")
        updates["max_run_tokens"] = max_run_tokens
    return replace(config, **updates) if updates else config
def _eval(args: argparse.Namespace) -> int:
    """Handle the eval subcommand."""
    cases = load_eval_cases(Path(args.cases))
    eval_workspace = Workspace(HarnessConfig.offline(args.workspace, args.config).workspace)
    eval_store = EvalStore(eval_workspace.evals_dir)
    results = []
    for case in cases:
        try:
            run = _execute_agent_task(
                task=case.task,
                workspace_arg=args.workspace,
                config_arg=args.config,
                mock=args.mock,
                provider=args.provider,
                stream=False,
                approval_override=args.approval,
                tool_profile_override=args.tool_profile,
                command_profile_override=args.command_profile,
                max_steps_override=args.max_steps,
                max_run_tokens_override=args.max_run_tokens,
            )
            result = evaluate_case(
                case,
                content=str(run["content"]),
                run_id=str(run["run_id"]),
                steps=int(run["steps"]),
            )
        except Exception as exc:
            result = evaluate_case(case, content="", run_id=None, steps=None, error=str(exc))
        results.append(result)
        if args.fail_fast and not result.ok:
            break
    passed = sum(1 for result in results if result.ok)
    report = {
        "ok": passed == len(cases),
        "case_file": str(Path(args.cases).resolve()),
        "total": len(cases),
        "evaluated": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": [result.to_dict() for result in results],
    }
    stored_report = eval_store.save(report)
    if args.json:
        print(json.dumps(stored_report, ensure_ascii=False, indent=2))
    else:
        print(f"eval_id: {stored_report['eval_id']}")
        print(f"eval: {passed}/{len(cases)} passed")
        for result in results:
            status = "pass" if result.ok else "fail"
            print(f"{status}\t{result.id}\t{result.run_id or '-'}\tsteps={result.steps if result.steps is not None else '-'}")
            if result.error:
                print(f"  error: {result.error}")
            if result.missing:
                print(f"  missing: {', '.join(result.missing)}")
    return 0 if stored_report["ok"] else 1


def _eval_store_for_workspace(workspace_arg: str | None, config_arg: str | None) -> EvalStore:
    config = HarnessConfig.offline(workspace_arg, config_arg)
    workspace = Workspace(config.workspace)
    return EvalStore(workspace.evals_dir)


def _list_evals(args: argparse.Namespace) -> int:
    store = _eval_store_for_workspace(args.workspace, args.config)
    summaries = store.list_evals(limit=args.limit)
    if args.json:
        print(json.dumps([summary.to_dict() for summary in summaries], ensure_ascii=False, indent=2))
        return 0
    if not summaries:
        print("No eval reports found.")
        return 0
    for summary in summaries:
        status = "passed" if summary.ok else "failed"
        print(f"{summary.eval_id}\t{status}\t{summary.passed}/{summary.total}\t{summary.created_at}")
    return 0


def _show_eval(args: argparse.Namespace) -> int:
    store = _eval_store_for_workspace(args.workspace, args.config)
    report = store.load(args.eval_id)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    status = "passed" if report.get("ok") else "failed"
    print(f"eval_id: {report.get('eval_id', args.eval_id)}")
    print(f"status: {status}")
    print(f"cases: {report.get('passed', 0)}/{report.get('total', 0)} passed")
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        result_status = "pass" if result.get("ok") else "fail"
        print(f"{result_status}\t{result.get('id', '')}\t{result.get('run_id') or '-'}")
    return 0



def _print_stream_chunk(chunk: str) -> None:
    """Print one streamed model chunk."""
    print(chunk, end="", flush=True)


def _store_for_workspace(workspace_arg: str | None, config_arg: str | None) -> RunStore:
    """Build a RunStore for workspace history commands."""
    config = HarnessConfig.offline(workspace_arg, config_arg)
    workspace = Workspace(config.workspace)
    return RunStore(workspace.logs_dir)


def _list_runs(args: argparse.Namespace) -> int:
    store = _store_for_workspace(args.workspace, args.config)
    summaries = store.list_runs(limit=args.limit)
    if args.json:
        print(json.dumps([summary.to_dict() for summary in summaries], ensure_ascii=False, indent=2))
        return 0
    if not summaries:
        print("No runs found.")
        return 0
    for summary in summaries:
        final_preview = (summary.final or "").replace("\n", " ")[:80]
        changed_count = 0
        if summary.changes is not None:
            raw_count = summary.changes.get("changed_count")
            changed_count = raw_count if isinstance(raw_count, int) else 0
        token_count = "-"
        if summary.usage is not None:
            raw_tokens = summary.usage.get("total_tokens")
            token_count = str(raw_tokens) if isinstance(raw_tokens, int) else "-"
        tool_count = "-"
        if summary.tools is not None:
            raw_tool_count = summary.tools.get("total_calls")
            raw_failed_count = summary.tools.get("failed_calls")
            if isinstance(raw_tool_count, int):
                tool_count = str(raw_tool_count)
                if isinstance(raw_failed_count, int) and raw_failed_count:
                    tool_count = f"{tool_count}/{raw_failed_count} failed"
        print(
            f"{summary.run_id}\t{summary.status}\t{summary.event_count} events\t"
            f"{changed_count} changes\t{tool_count} tools\t{token_count} tokens\t{summary.last_event_at}\t{final_preview}"
        )
    return 0


def _list_tools(args: argparse.Namespace) -> int:
    config = HarnessConfig.offline(args.workspace, args.config)
    workspace = Workspace(config.workspace)
    command_profile = args.command_profile or config.command_profile
    policy = CommandPolicy.default(command_profile)
    executor = CommandExecutor(workspace, policy, config.timeout_seconds, config.max_tool_output_chars)
    tool_profile = args.tool_profile or config.tool_profile
    router = build_default_router(workspace, executor, config.max_tool_output_chars, tool_profile=tool_profile)
    specs = router.specs()
    payload = {"tool_profile": tool_profile, "command_profile": command_profile, "tools": specs}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"tool_profile: {tool_profile}")
    print(f"command_profile: {command_profile}")
    for spec in specs:
        print(f"{spec.get('name', '')}\t{spec.get('description', '')}")
    return 0


def _show_run(args: argparse.Namespace) -> int:
    store = _store_for_workspace(args.workspace, args.config)
    if args.summary:
        summary = _run_diagnostic_summary(store, args.run_id)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            _print_run_diagnostic_summary(summary)
        return 0
    records = store.load_run(args.run_id)
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0
    for record in records:
        seq = record.get("seq", "?")
        kind = record.get("kind", "unknown")
        ts = record.get("ts", "")
        payload = record.get("payload", {})
        preview = json.dumps(payload, ensure_ascii=False)
        if len(preview) > 180:
            preview = preview[:177] + "..."
        print(f"#{seq} {ts} {kind}: {preview}")
    return 0


def _run_diagnostic_summary(store: RunStore, run_id: str) -> dict[str, object]:
    summary = store.summarize_run(run_id)
    records = store.load_run(run_id)
    last_error = _last_event_payload(records, {"action_error", "run_finished"}, require_error=True)
    last_failed_tool = _last_failed_tool(records)
    tools = summary.tools if isinstance(summary.tools, dict) else {}
    return {
        "run_id": summary.run_id,
        "status": summary.status,
        "steps": summary.step_count,
        "events": summary.event_count,
        "started_at": summary.started_at,
        "last_event_at": summary.last_event_at,
        "final": summary.final,
        "error": summary.error,
        "changes": summary.changes,
        "usage": summary.usage,
        "tool_calls": tools.get("total_calls", 0),
        "failed_tools": tools.get("failed_calls", 0),
        "tool_names": tools.get("tool_names", []),
        "last_error": last_error,
        "last_failed_tool": last_failed_tool,
    }


def _last_event_payload(records: list[dict[str, object]], kinds: set[str], require_error: bool = False) -> dict[str, object] | None:
    for record in reversed(records):
        if record.get("kind") not in kinds:
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if require_error and not payload.get("error"):
            continue
        return payload
    return None


def _last_failed_tool(records: list[dict[str, object]]) -> dict[str, object] | None:
    for record in reversed(records):
        if record.get("kind") != "tool_result":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if not isinstance(result, dict) or result.get("ok") is True:
            continue
        action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
        return {
            "tool": action.get("tool", ""),
            "args": action.get("args", {}),
            "error": result.get("error") or result.get("stderr") or result,
        }
    return None


def _print_run_diagnostic_summary(summary: dict[str, object]) -> None:
    print(f"run_id: {summary['run_id']}")
    print(f"status: {summary['status']}")
    print(f"steps: {summary['steps']}")
    print(f"events: {summary['events']}")
    print(f"tool_calls: {summary['tool_calls']}")
    print(f"failed_tools: {summary['failed_tools']}")
    usage = summary.get("usage")
    if isinstance(usage, dict):
        print(f"tokens: {usage.get('total_tokens', '-')}")
    changes = summary.get("changes")
    if isinstance(changes, dict):
        print(f"changes: {changes.get('changed_count', 0)}")
    if summary.get("final"):
        final = str(summary["final"]).replace("\n", " ")[:200]
        print(f"final: {final}")
    if summary.get("error"):
        print(f"error: {summary['error']}")
    if summary.get("last_failed_tool"):
        print("last_failed_tool:")
        print(json.dumps(summary["last_failed_tool"], ensure_ascii=False, indent=2))
    if summary.get("last_error"):
        print("last_error:")
        print(json.dumps(summary["last_error"], ensure_ascii=False, indent=2))


def _show_changes(args: argparse.Namespace) -> int:
    store = _store_for_workspace(args.workspace, args.config)
    changes = store.load_changes(args.run_id)
    if args.json:
        print(json.dumps(changes, ensure_ascii=False, indent=2))
        return 0
    changed_count = changes.get("changed_count", 0)
    print(f"changed_count: {changed_count}")
    for key in ("added", "modified", "deleted"):
        values = changes.get(key, [])
        if not isinstance(values, list):
            values = []
        print(f"{key}: {len(values)}")
        for value in values:
            print(f"  {value}")
    return 0


def _show_checkpoint(args: argparse.Namespace) -> int:
    config = HarnessConfig.offline(args.workspace, args.config)
    workspace = Workspace(config.workspace)
    checkpoint = RunCheckpointStore(workspace.checkpoints_dir).load(args.run_id)
    if args.json:
        print(json.dumps(checkpoint, ensure_ascii=False, indent=2))
        return 0
    messages = checkpoint.get("messages", [])
    message_count = len(messages) if isinstance(messages, list) else 0
    print(f"run_id: {checkpoint.get('run_id', args.run_id)}")
    print(f"status: {checkpoint.get('status', '')}")
    print(f"step: {checkpoint.get('step', 0)}")
    print(f"messages: {message_count}")
    print(f"saved_at: {checkpoint.get('saved_at', '')}")
    plan = checkpoint.get("plan", {})
    if isinstance(plan, dict):
        counts = plan.get("counts", {})
        print(f"plan_counts: {json.dumps(counts, ensure_ascii=False)}")
    return 0


def _list_providers(args: argparse.Namespace) -> int:
    providers = [PROVIDER_PRESETS[name].to_dict() for name in provider_names()]
    if args.json:
        print(json.dumps(providers, ensure_ascii=False, indent=2))
        return 0
    for provider in providers:
        print(
            f"{provider['name']}\t{provider['default_model']}\t{provider['api_key_env']}\t{provider['base_url']}"
        )
    return 0


def _list_profiles(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(PROFILE_DESCRIPTIONS, ensure_ascii=False, indent=2))
        return 0
    for group, profiles in PROFILE_DESCRIPTIONS.items():
        print(f"{group}:")
        for profile in profiles:
            print(f"  {profile['name']}\t{profile['description']}")
    return 0


def _server(args: argparse.Namespace) -> int:
    serve(args.workspace, args.host, args.port, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
