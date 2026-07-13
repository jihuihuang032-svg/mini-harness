"""上下文管理模块。

负责三件事:
    1. 构造初始消息序列(system + 仓库摘要 + 用户任务)
    2. 每轮调用模型前裁剪上下文,保证消息总体不超过字符预算
    3. 压缩工具结果,防止单次输出耗尽上下文

类比 Java:这里的 ContextManager 类似一个 PromptStrategy,
基于"字符预算"而非 tokenizer(因为本框架不绑死某个 provider)。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from harness.messages import Message
from harness.runtime.workspace import Workspace


# 这些目录在生成仓库摘要时被忽略,避免污染上下文
DEFAULT_IGNORED_DIRS = {
    ".git",
    ".harness",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}


@dataclass(frozen=True)
class ContextBudget:
    """上下文预算:三个字符上限,类似 Java 的配置常量。"""
    max_message_chars: int = 60_000       # 上下文消息总字符数上限
    max_tool_result_chars: int = 12_000  # 单次工具结果字符数上限
    max_summary_files: int = 120         # 仓库摘要最多采样的文件数


class ContextManager:
    """上下文管理器:负责消息初始化、裁剪、工具结果压缩。"""

    def __init__(self, workspace: Workspace, budget: ContextBudget) -> None:
        self.workspace = workspace
        self.budget = budget
        # 仓库摘要在构造时就生成,后续复用
        self.repo_summary = self.build_repo_summary()

    def initial_messages(self, system_prompt: str, task: str) -> list[Message]:
        """构造初始消息序列:system + 仓库摘要 + 用户任务。"""
        context = "Repository context:\n" + self.repo_summary
        return [
            Message("system", system_prompt),
            Message("user", context),
            Message("user", task),
        ]

    def prepare_for_model(self, messages: list[Message]) -> list[Message]:
        """模型调用前裁剪上下文。

        策略:
            - 总大小没超预算 -> 原样返回
            - 超预算但消息少 -> 整体截断
            - 超预算且消息多 -> 保留前 3 条(system/摘要/任务)+ 尾部最新消息,
              中间被丢弃的消息会替换为一条提示
        """
        if _messages_size(messages) <= self.budget.max_message_chars:
            return messages
        if len(messages) <= 3:
            return [_truncate_message(message, self.budget.max_message_chars // max(len(messages), 1)) for message in messages]

        # 保留 system、仓库摘要和原始任务,再尽量保留最近对话尾部。
        head = messages[:3]
        tail = messages[3:]
        kept_tail: list[Message] = []
        remaining = self.budget.max_message_chars - _messages_size(head)
        # 从尾部倒序保留,因为最近的消息对模型决策最重要
        for message in reversed(tail):
            size = len(message.content)
            if size <= remaining:
                kept_tail.append(message)
                remaining -= size
            elif remaining > 500:
                kept_tail.append(_truncate_message(message, remaining))
                remaining = 0
                break
            else:
                break
        kept_tail.reverse()
        omitted = len(tail) - len(kept_tail)
        if omitted > 0:
            # 给模型留个提示:中间被丢弃了 N 条消息
            summary = Message("user", f"Context manager omitted {omitted} older messages to stay within budget.")
            return head + [summary] + kept_tail
        return head + kept_tail

    def compress_tool_result(self, result: dict[str, object]) -> dict[str, object]:
        """压缩工具结果,超过预算时只保留截断后的内容。"""
        return _truncate_jsonable(result, self.budget.max_tool_result_chars)

    def build_repo_summary(self) -> str:
        """扫描工作区生成仓库摘要。

        返回 JSON 字符串,包含:根路径、采样文件数、扩展名统计、文件相对路径列表。
        受 max_summary_files 限制。
        """
        files: list[str] = []
        extension_counts: Counter[str] = Counter()  # Counter 类似 Guava Multiset
        for path in sorted(self.workspace.root.rglob("*")):  # rglob 递归遍历
            if len(files) >= self.budget.max_summary_files:
                break
            if _should_ignore(path):
                continue
            if path.is_file():
                rel = self.workspace.relative(path)
                files.append(rel)
                extension_counts[path.suffix or "<no_ext>"] += 1
        summary = {
            "root": str(self.workspace.root),
            "file_count_sampled": len(files),
            "extension_counts": dict(sorted(extension_counts.items())),
            "files": files,
            "truncated": len(files) >= self.budget.max_summary_files,
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)


def _should_ignore(path: Path) -> bool:
    """路径任一部分出现在忽略目录集合里就忽略。"""
    return any(part in DEFAULT_IGNORED_DIRS for part in path.parts)


def _messages_size(messages: list[Message]) -> int:
    """消息列表总字符数(含角色名)。"""
    return sum(len(message.role) + len(message.content) for message in messages)


def _truncate_message(message: Message, max_chars: int) -> Message:
    """截断单条消息,保留原角色。"""
    return Message(message.role, _truncate_text(message.content, max_chars))


def _truncate_jsonable(value: object, max_chars: int) -> dict[str, object]:
    """把任意可 JSON 序列化的对象压缩到字符上限内。

    不超限 -> 原样返回
    超限   -> 返回 ok=False, truncated=True, content=截断后的字符串
    """
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        if isinstance(value, dict):
            return value
        return {"ok": True, "value": value}
    # 工具输出进入模型前必须截断,防止单次输出耗尽上下文预算。
    truncated_text = _truncate_text(text, max_chars)
    return {
        "ok": False,
        "truncated": True,
        "original_chars": len(text),
        "content": truncated_text,
    }


def _truncate_text(text: str, max_chars: int) -> str:
    """截断字符串并追加"已截断"提示。"""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    suffix = f"\n... <context truncated {omitted} chars>"
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix
