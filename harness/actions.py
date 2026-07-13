"""动作协议模块。

模型不可信:它的输出是文本,框架必须按本模块定义的动作协议解析。
本模块负责:
    1. 解析模型文本为 AgentAction(plan/todo_update/tool_call/final 四种之一)。
    2. 把 OpenAI 原生 tool_calls 规范化为统一的 tool_call JSON 字符串,
       这样 Agent 主循环不需要区分模型走的是 JSON 协议还是原生函数调用协议。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


# Literal 类似 Java 枚举,把字符串限制在固定集合内
ActionType = Literal["plan", "todo_update", "tool_call", "final"]


@dataclass(frozen=True)
class AgentAction:
    """解析后的动作对象。frozen=True 表示不可变。"""
    type: ActionType
    tool: str | None = None
    args: dict[str, object] | None = None
    content: str | None = None
    items: list[object] | None = None

    def to_log_dict(self) -> dict[str, object]:
        """转换为可写入 trace 的字典,仅包含非空字段。"""
        data: dict[str, object] = {"type": self.type}
        if self.tool is not None:
            data["tool"] = self.tool
        if self.args is not None:
            data["args"] = self.args
        if self.content is not None:
            data["content"] = self.content
        if self.items is not None:
            data["items"] = self.items
        return data


def parse_action(text: str) -> AgentAction:
    """把模型返回的文本解析为 AgentAction。

    解析策略:
        1. 容忍 ```代码块``` 包裹
        2. 容忍 JSON 前后有杂字(用正则提取第一个 {...})
        3. 校验 type 字段并按类型校验其它必填字段
    解析失败抛 ValueError,Agent 会把错误反馈回循环让模型修复。
    """
    parsed = _parse_json_object(text)
    action_type = parsed.get("type")
    if action_type == "final":
        content = parsed.get("content")
        if not isinstance(content, str):
            raise ValueError("Final action requires string field 'content'.")
        return AgentAction(type="final", content=content)
    if action_type == "tool_call":
        tool = parsed.get("tool")
        args = parsed.get("args", {})
        if not isinstance(tool, str) or not tool:
            raise ValueError("Tool action requires non-empty string field 'tool'.")
        if not isinstance(args, dict):
            raise ValueError("Tool action field 'args' must be an object.")
        return AgentAction(type="tool_call", tool=tool, args=args)
    if action_type == "plan":
        items = parsed.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("Plan action requires non-empty array field 'items'.")
        return AgentAction(type="plan", items=items)
    if action_type == "todo_update":
        items = parsed.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("Todo update action requires non-empty array field 'items'.")
        return AgentAction(type="todo_update", items=items)
    raise ValueError("Action type must be 'plan', 'todo_update', 'tool_call', or 'final'.")


def action_from_openai_message(message: dict[str, Any]) -> str | None:
    """把 OpenAI 原生 tool_calls 规范化为统一的 tool_call JSON 字符串。

    返回值是字符串而不是 AgentAction,因为 Agent 主循环会把模型输出原样
    追加到 messages(assistant 角色),所以这里返回字符串便于复用。
    返回 None 表示这条消息没有原生 tool_calls。
    """
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    # 只取第一个 tool_call,因为本框架约定模型每轮只产生一个动作
    first = tool_calls[0]
    if not isinstance(first, dict):
        raise ValueError("OpenAI tool call must be an object.")
    function = first.get("function")
    if not isinstance(function, dict):
        raise ValueError("OpenAI tool call missing function object.")
    name = function.get("name")
    raw_args = function.get("arguments", "{}")
    if not isinstance(name, str) or not name:
        raise ValueError("OpenAI tool call function.name must be a non-empty string.")
    if raw_args is None:
        raw_args = "{}"
    if not isinstance(raw_args, str):
        raise ValueError("OpenAI tool call function.arguments must be a JSON string.")
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI tool call arguments are not valid JSON: {exc}") from exc
    if not isinstance(args, dict):
        raise ValueError("OpenAI tool call arguments must decode to an object.")
    return json.dumps({"type": "tool_call", "tool": name, "args": args}, ensure_ascii=False)


def _parse_json_object(text: str) -> dict[str, Any]:
    """容错地解析 JSON 对象。

    步骤:
        1. strip 空白
        2. 如果被 ``` 包裹,去掉代码块标记
        3. 直接 json.loads;失败则用正则提取第一个 {...} 再尝试
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return JSON: {text[:500]}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model action must be a JSON object.")
    return parsed


def _strip_code_fence(text: str) -> str:
    """去掉 ```...``` 代码块包裹,只保留中间内容。"""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
