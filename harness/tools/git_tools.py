"""Git 类工具实现。

通过 CommandExecutor 调用 git 命令,不直接 subprocess,
这样所有命令都经过统一的命令策略(policy)审批与日志记录。
"""

from __future__ import annotations

from harness.runtime.executor import CommandExecutor


class GitStatusTool:
    """git_status:展示 git 状态(分支、未跟踪文件、改动)。"""
    name = "git_status"
    description = "Show concise git status, including branch and untracked files."
    args_schema = {
        "type": "object",
        "properties": {
            "short": {"type": "boolean", "default": True},
            "branch": {"type": "boolean", "default": True},
        },
    }

    def __init__(self, executor: CommandExecutor) -> None:
        self.executor = executor

    def run(self, args: dict[str, object]) -> dict[str, object]:
        short = bool(args.get("short", True))
        branch = bool(args.get("branch", True))
        # 根据参数组合 git status 命令
        if short:
            command = "git status --short --branch" if branch else "git status --short"
        else:
            command = "git status --branch" if branch else "git status"
        return self.executor.run(command)


class GitDiffTool:
    """git_diff:展示工作区改动(staged 或 unstaged)。"""
    name = "git_diff"
    description = "Show git diff for the workspace."
    args_schema = {
        "type": "object",
        "properties": {"staged": {"type": "boolean", "default": False}},
    }

    def __init__(self, executor: CommandExecutor) -> None:
        self.executor = executor

    def run(self, args: dict[str, object]) -> dict[str, object]:
        staged = bool(args.get("staged", False))
        command = "git diff --staged" if staged else "git diff"
        return self.executor.run(command)
