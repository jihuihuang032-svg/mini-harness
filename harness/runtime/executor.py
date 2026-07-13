"""命令执行器:把 policy(策略)+ approval(审批)+ subprocess 组合在一起。

RunCommandTool 和 git_status / git_diff 都通过它执行命令,
保证所有命令都走统一的策略校验、审批流程、超时控制和输出截断。
"""

from __future__ import annotations

import subprocess

from harness.runtime.approval import ApprovalController
from harness.runtime.policy import CommandPolicy
from harness.runtime.workspace import Workspace


class CommandExecutor:
    """命令执行器:策略评估 -> 审批 -> subprocess -> 截断输出。"""

    def __init__(
        self,
        workspace: Workspace,
        policy: CommandPolicy,
        timeout_seconds: int,
        max_output_chars: int,
        approval: ApprovalController | None = None,
    ) -> None:
        self.workspace = workspace
        self.policy = policy
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        # 默认 never 模式,approval_required 命令一律拒绝
        self.approval = approval or ApprovalController("never")

    def run(self, command: str) -> dict[str, object]:
        """执行命令,返回统一的 dict 格式。

        返回字段:
            - ok:命令是否成功(returncode == 0 且被批准执行)
            - command:执行的命令
            - returncode:退出码(被拒绝时为 None)
            - risk:风险评估结果
            - approval:审批决定
            - stdout / stderr:输出(已截断)
        """
        # 1. 策略评估
        assessment = self.policy.assess(command)
        # 2. 审批决定
        decision = self.approval.decide(command, assessment)
        if not decision.approved:
            # 拒绝执行:返回 ok=False + 风险/审批信息
            return {
                "ok": False,
                "command": command,
                "returncode": None,
                "risk": assessment.to_dict(),
                "approval": decision.to_dict(),
                "stdout": "",
                "stderr": decision.reason,
            }
        # 3. 实际执行:subprocess.run 是阻塞调用
        completed = subprocess.run(
            command,
            cwd=self.workspace.root,
            shell=True,  # shell=True 才能用 | 等管道语法
            text=True,
            capture_output=True,  # 捕获 stdout/stderr
            timeout=self.timeout_seconds,
        )
        return {
            "ok": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "risk": assessment.to_dict(),
            "approval": decision.to_dict(),
            "stdout": _truncate(completed.stdout, self.max_output_chars),
            "stderr": _truncate(completed.stderr, self.max_output_chars),
        }


def _truncate(text: str, max_chars: int) -> str:
    """截断超长输出,末尾追加截断提示。

    模块级函数,被 file_tools 等其它模块复用(因此不带下划线也可,
    但保留 _ 前缀表示"内部工具")。
    """
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n... <truncated {omitted} chars>"
