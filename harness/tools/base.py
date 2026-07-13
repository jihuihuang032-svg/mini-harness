"""工具基础结构:Tool 协议、ToolRouter 路由器、ToolProfile 权限配置。

设计要点:
    - Tool 是 Protocol(鸭子类型接口),任何有 name/description/args_schema/run 的对象都是 Tool
    - ToolRouter 注册所有工具,按 profile 过滤可见工具
    - 即使模型绕过 prompt 调用了隐藏工具,路由层也会再次拦截(安全防御)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from harness.schema import SchemaValidationError, validate_args


ToolProfile = str  # 字符串字面量类型别名,实际取值为 "full"/"review"/"read-only"


# profile 控制模型可见和可执行的工具面,便于在不同任务里收紧权限
# None 表示不限制(所有工具都可见);set 列出允许的工具名
TOOL_PROFILES: dict[ToolProfile, set[str] | None] = {
    "full": None,  # 不限制
    "review": {"list_files", "read_file", "search_text", "git_status", "git_diff", "run_command"},
    "read-only": {"list_files", "read_file", "search_text", "git_status", "git_diff"},
}


@dataclass(frozen=True)
class ToolSpec:
    """工具规格:暴露给模型的元数据(name/description/args_schema)。"""
    name: str
    description: str
    args_schema: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
        }


class Tool(Protocol):
    """工具协议:每个工具都要满足此接口。

    类似 Java interface,但 Python Protocol 不需要显式 implements,
    只要类有同名字段和方法就自动满足(structural typing)。
    """
    name: str
    description: str
    args_schema: dict[str, object]

    def run(self, args: dict[str, object]) -> dict[str, object]:
        """执行工具调用,返回 JSON 可序列化的结果 dict。"""


class ToolRouter:
    """工具路由器:注册工具、过滤可见工具、分发调用、统一异常处理。"""

    def __init__(self, profile: ToolProfile = "full") -> None:
        if profile not in TOOL_PROFILES:
            raise ValueError(f"Unknown tool profile: {profile}")
        self.profile = profile
        self._tools: dict[str, Tool] = {}  # 工具名 -> 工具实例

    def register(self, tool: Tool) -> None:
        """注册一个工具。"""
        self._tools[tool.name] = tool

    def specs(self) -> list[dict[str, object]]:
        """返回当前 profile 下可见的工具规格列表(按名排序)。"""
        return [
            ToolSpec(tool.name, tool.description, tool.args_schema).to_dict()
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
            if self._is_allowed(tool.name)
        ]

    def call(self, name: str, args: dict[str, object]) -> dict[str, object]:
        """按名调用工具,统一返回 {ok, ...} 格式。

        - 工具不存在 -> {ok: False, error: ...}
        - 工具被 profile 禁用 -> {ok: False, error: ...}(二次防御)
        - 参数校验失败 -> {ok: False, error: ...}
        - 工具内部抛异常 -> {ok: False, error: ...}
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}", "available_tools": sorted(self._tools)}
        # 即使模型绕过 prompt 调用了隐藏工具,也必须在路由层再次拦截
        if not self._is_allowed(name):
            return {
                "ok": False,
                "error": f"Tool {name} is not allowed by profile {self.profile}.",
                "profile": self.profile,
                "available_tools": sorted(tool.name for tool in self._tools.values() if self._is_allowed(tool.name)),
            }
        try:
            # 参数校验:失败抛 SchemaValidationError
            validated_args = validate_args(tool.args_schema, args)
        except SchemaValidationError as exc:
            return {"ok": False, "error": f"Invalid args for tool {name}: {exc}"}
        try:
            return tool.run(validated_args)
        except Exception as exc:
            # 任何工具异常都被包装成 {ok: False, error: ...},防止 Agent 主循环崩溃
            return {"ok": False, "error": str(exc)}

    def _is_allowed(self, name: str) -> bool:
        """判断工具是否在当前 profile 允许范围内。"""
        allowed = TOOL_PROFILES[self.profile]
        return allowed is None or name in allowed
