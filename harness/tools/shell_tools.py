"""Shell 工具实现。

RunCommandTool:让模型在工作区内执行 shell 命令。
所有命令都通过 CommandExecutor,经过命令策略(policy)审批与日志记录。
"""

from __future__ import annotations

from harness.runtime.executor import CommandExecutor


class RunCommandTool:
    """run_command:执行任意 shell 命令(受命令策略约束)。"""
    name = "run_command"
    description = "Run a shell command in the workspace under the command policy."
    args_schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self, executor: CommandExecutor) -> None:
        self.executor = executor

    def run(self, args: dict[str, object]) -> dict[str, object]:
        command = args.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("Missing required string arg: command")
        # executor.run 内部会做策略校验、超时控制、输出截断
        return self.executor.run(command)
