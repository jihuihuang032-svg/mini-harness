"""Action protocol parsing for Mini Harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


ActionType = Literal["plan", "todo_update", "tool_call", "final"]


@dataclass(frozen=True)
class AgentAction:
    type: ActionType
    tool: str | None = None
    args: dict[str, object] | None = None
    content: str | None = None
    items: list[object] | None = None

    def to_log_dict(self) -> dict[str, object]:
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
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
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
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = _first_json_object(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Model action must be a JSON object.")
    return parsed


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Model did not return JSON: {text[:500]}")


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()