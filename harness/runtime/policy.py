"""命令风险策略:评估命令风险等级。

把每条命令分到三个等级:
    - low:无明显风险,直接放行
    - approval_required:有风险(如 git push / npm install),需要审批
    - denied:绝对禁止(如 rm -rf /、format、shutdown)
策略分两种 profile:
    - default:黑名单模式,只要不在禁止/审批列表里就放行
    - strict:白名单模式,只有命中 allowed_patterns 才放行
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


RiskLevel = Literal["low", "approval_required", "denied"]
CommandProfile = Literal["default", "strict"]


@dataclass(frozen=True)
class CommandAssessment:
    """命令评估结果:风险等级 + 原因。"""
    level: RiskLevel
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "reason": self.reason}


@dataclass(frozen=True)
class CommandPolicy:
    """命令策略:三组正则模式 + 一个 profile。"""
    profile: CommandProfile
    denied_patterns: tuple[tuple[re.Pattern[str], str], ...]      # 绝对禁止
    approval_patterns: tuple[tuple[re.Pattern[str], str], ...]     # 需审批
    allowed_patterns: tuple[tuple[re.Pattern[str], str], ...] = ()  # 白名单(strict 模式才用)

    @classmethod
    def default(cls, profile: CommandProfile = "default") -> "CommandPolicy":
        """构造默认策略:三组内置正则规则。

        @classmethod 类似 Java static factory method。
        """
        if profile not in {"default", "strict"}:
            raise ValueError("Command profile must be one of: default, strict.")
        # 绝对禁止:可能造成不可逆损失的命令
        denied_patterns = [
            (r"\brm\s+-rf\b", "recursive force delete"),
            (r"\bdel\s+/[sq]\b", "quiet recursive delete"),
            (r"\bformat\b", "disk format command"),
            (r"\bshutdown\b", "machine shutdown"),
            (r"\breboot\b", "machine reboot"),
            (r"\bgit\s+reset\s+--hard\b", "destructive git reset"),
            (r"\bgit\s+clean\s+-fd\b", "destructive git clean"),
            (r">\s*/dev/sd[a-z]", "raw disk write"),
        ]
        # 需审批:有副作用但不致命
        approval_patterns = [
            (r"\bgit\s+push\b", "publishes repository changes"),
            (r"\bgit\s+commit\b", "creates repository history"),
            (r"\bnpm\s+install\b", "installs dependencies"),
            (r"\bpip\s+install\b", "installs dependencies"),
            (r"\bpoetry\s+add\b", "changes dependencies"),
            (r"\bcurl\b.*\|", "downloads and executes piped content"),
            (r"\birm\b.*\|", "downloads and executes piped content"),
        ]
        # 白名单:strict 模式下只有命中这些才放行
        allowed_patterns = [
            (r"^\s*git\s+(status|diff|log|show|branch)\b", "read-only git inspection"),
            (r"^\s*python\s+-m\s+(unittest|compileall)\b", "python test or compile command"),
            (r"^\s*python\s+--version\s*$", "python version check"),
            (r"^\s*python\s+-V\s*$", "python version check"),
            (r"^\s*dir\b", "directory listing"),
            (r"^\s*ls\b", "directory listing"),
            (r"^\s*Get-ChildItem\b", "directory listing"),
            (r"^\s*rg\b", "ripgrep search"),
        ]
        return cls(
            profile=profile,
            denied_patterns=tuple((re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in denied_patterns),
            approval_patterns=tuple((re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in approval_patterns),
            allowed_patterns=tuple((re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in allowed_patterns),
        )

    def assess(self, command: str) -> CommandAssessment:
        """评估命令风险,返回 CommandAssessment。

        判断优先级(从高到低):
            1. denied_patterns 命中 -> denied
            2. approval_patterns 命中 -> approval_required
            3. strict 模式且不在 allowed_patterns -> denied
            4. 其它 -> low(放行)
        """
        for pattern, reason in self.denied_patterns:
            if pattern.search(command):
                return CommandAssessment("denied", reason)
        for pattern, reason in self.approval_patterns:
            if pattern.search(command):
                return CommandAssessment("approval_required", reason)
        if self.profile == "strict" and not self._is_strict_allowed(command):
            return CommandAssessment("denied", "command is not allowed by strict profile")
        return CommandAssessment("low", "no risky pattern matched")

    def validate(self, command: str) -> CommandAssessment:
        """校验命令,denied 直接抛 PermissionError(供直接调用方使用)。"""
        assessment = self.assess(command)
        if assessment.level == "denied":
            raise PermissionError(f"Command denied by policy: {assessment.reason}: {command}")
        return assessment

    def _is_strict_allowed(self, command: str) -> bool:
        """strict 模式下判断命令是否命中白名单。"""
        return any(pattern.search(command) for pattern, _ in self.allowed_patterns)
