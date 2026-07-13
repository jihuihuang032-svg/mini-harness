"""审批控制器:决定 approval_required 命令是否实际执行。

三种模式:
    - never:从不审批,approval_required 一律拒绝
    - on-request:弹窗让用户决定(通过 prompt 回调)
    - auto:approval_required 自动放行(只用于受信任环境)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from harness.runtime.policy import CommandAssessment


ApprovalMode = Literal["never", "on-request", "auto"]


@dataclass(frozen=True)
class ApprovalDecision:
    """审批决定:是否批准 + 模式 + 原因。"""
    approved: bool
    mode: ApprovalMode
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"approved": self.approved, "mode": self.mode, "reason": self.reason}


class ApprovalController:
    """审批控制器:把 CommandAssessment 转成 ApprovalDecision。"""

    def __init__(
        self,
        mode: ApprovalMode = "never",
        prompt: Callable[[str], str] | None = None,
    ) -> None:
        """
        @param mode: 审批模式
        @param prompt: 用户输入回调,默认用内置 input()(可注入 mock 便于测试)
        """
        self.mode = mode
        self.prompt = prompt or input

    def decide(self, command: str, assessment: CommandAssessment) -> ApprovalDecision:
        """根据评估结果和当前模式决定是否批准。"""
        if assessment.level == "low":
            # 低风险直接放行
            return ApprovalDecision(True, self.mode, "low-risk command")
        if assessment.level == "denied":
            # 策略禁止的命令一律拒绝
            return ApprovalDecision(False, self.mode, f"denied by policy: {assessment.reason}")
        if self.mode == "auto":
            # auto 模式:approval_required 自动放行
            return ApprovalDecision(True, self.mode, f"auto-approved: {assessment.reason}")
        if self.mode == "never":
            # never 模式:approval_required 一律拒绝
            return ApprovalDecision(False, self.mode, f"approval required but approval mode is never: {assessment.reason}")
        # on-request 模式:让用户决定
        answer = self.prompt(
            f"Command requires approval ({assessment.reason}): {command}\nAllow? [y/N] "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return ApprovalDecision(True, self.mode, f"approved by user: {assessment.reason}")
        return ApprovalDecision(False, self.mode, f"rejected by user: {assessment.reason}")
