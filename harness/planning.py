"""计划与待办事项模块。

实现 Agent 的待办清单(Todo List)功能:
    - 计划状态用 PlanState 持有,作为单一可变对象在 Agent 主循环中被更新
    - 每个 todo 是一个 TodoItem,有 pending/in_progress/completed 三种状态
    - 模型通过 plan 动作替换整个计划,通过 todo_update 动作增量更新状态
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# 状态枚举(注意是 "completed" 不是 "done")
TodoStatus = Literal["pending", "in_progress", "completed"]

_VALID_STATUSES: tuple[TodoStatus, ...] = ("pending", "in_progress", "completed")


@dataclass
class TodoItem:
    """单个待办项:id + 内容 + 状态。

    id 由 PlanState.replace 自动分配(若模型未提供)。
    """
    id: str
    content: str
    status: TodoStatus = "pending"

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "content": self.content, "status": self.status}


@dataclass
class PlanState:
    """整个计划的状态:任务列表。

    mutable,因为 Agent 主循环每一步都可能更新它。
    内部用 list[TodoItem] 存储,对外暴露 snapshot()/replace()/update()。
    """
    items: list[TodoItem] = field(default_factory=list)

    def replace(self, raw_items: list[object]) -> dict[str, object]:
        """用模型 plan 动作提供的 items 替换整个计划,返回新 snapshot。

        接受两种格式的 item:
            1. "字符串":自动分配 id("1", "2", ...),status="pending"
            2. {"id": "...", "content": "...", "status": "..."}:完整对象,id 可选(不传则自动分配)
        status 必须是 pending/in_progress/completed 之一,否则抛 ValueError。
        """
        parsed = _parse_items(raw_items)
        # 给缺 id 的项分配序号(从 1 开始,跳过已存在的 id)
        existing_ids = {item.id for item in parsed}
        next_id = 1
        for item in parsed:
            if not item.id:
                while str(next_id) in existing_ids:
                    next_id += 1
                item.id = str(next_id)
                existing_ids.add(item.id)
                next_id += 1
        self.items = parsed
        return self.snapshot()

    def update(self, raw_items: list[object]) -> dict[str, object]:
        """增量更新 todo 状态(按 id 匹配),返回新 snapshot。

        每个 raw_item 必须包含 id 字段,匹配现有 plan 中的项。
        若 id 不存在则抛 ValueError。
        """
        if not isinstance(raw_items, list):
            raise ValueError(f"todo_update items must be a list, got {type(raw_items).__name__}")
        index_by_id = {item.id: index for index, item in enumerate(self.items)}
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ValueError(f"todo_update item must be an object, got {type(raw).__name__}")
            item_id = raw.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("todo_update item must have non-empty string 'id'.")
            if item_id not in index_by_id:
                raise ValueError(f"Unknown todo id: {item_id}")
            index = index_by_id[item_id]
            if "status" in raw:
                status = raw["status"]
                if status not in _VALID_STATUSES:
                    raise ValueError(f"Todo status must be one of {_VALID_STATUSES}, got {status!r}")
                self.items[index].status = status
            if "content" in raw and isinstance(raw["content"], str):
                self.items[index].content = raw["content"]
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        """返回当前计划的快照 dict(用于 trace/checkpoint)。

        结构:
            {
                "items": [{"id": ..., "content": ..., "status": ...}, ...],
                "counts": {"pending": int, "in_progress": int, "completed": int},
            }
        """
        counts = {"pending": 0, "in_progress": 0, "completed": 0}
        for item in self.items:
            counts[item.status] += 1
        return {
            "items": [item.to_dict() for item in self.items],
            "counts": counts,
        }

    def render_for_model(self) -> str:
        """渲染成给模型看的 todo 文本,作为 system reminder 注入。"""
        if not self.items:
            return ""
        lines = ["Current plan:"]
        for item in self.items:
            mark = _status_mark(item.status)
            lines.append(f"  [{item.id}] {mark} {item.content}")
        return "\n".join(lines)

    @staticmethod
    def from_snapshot(snapshot: dict[str, object]) -> "PlanState":
        """从 checkpoint 的 snapshot dict 重建 PlanState。

        @param snapshot: 形如 {"items": [{"id": ..., "content": ..., "status": ...}], "counts": {...}}
        """
        if not isinstance(snapshot, dict):
            return PlanState()
        raw_items = snapshot.get("items", [])
        if not isinstance(raw_items, list):
            return PlanState()
        items: list[TodoItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("id", ""))
            content = str(raw.get("content", ""))
            status = raw.get("status", "pending")
            if status not in _VALID_STATUSES:
                status = "pending"
            items.append(TodoItem(id=item_id, content=content, status=status))  # type: ignore[arg-type]
        return PlanState(items=items)


def _parse_items(raw_items: list[object]) -> list[TodoItem]:
    """从模型 plan 动作的 items 字段解析出 TodoItem 列表。

    接受两种格式:
        1. "字符串":content=字符串,id 自动分配
        2. {"id": "...", "content": "...", "status": "..."}:完整对象,id 可选
    """
    if not isinstance(raw_items, list):
        raise ValueError(f"plan items must be a list, got {type(raw_items).__name__}")
    parsed: list[TodoItem] = []
    for raw in raw_items:
        if isinstance(raw, str):
            if not raw:
                raise ValueError("Plan item string must be non-empty.")
            parsed.append(TodoItem(id="", content=raw, status="pending"))
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"Plan item must be string or object, got {type(raw).__name__}")
        content = raw.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("Plan item must have non-empty string 'content'.")
        status = raw.get("status", "pending")
        if status not in _VALID_STATUSES:
            raise ValueError(f"Todo status must be one of {_VALID_STATUSES}, got {status!r}")
        item_id = raw.get("id", "")
        if not isinstance(item_id, str) and item_id is not None:
            raise ValueError("Plan item 'id' must be a string if provided.")
        parsed.append(TodoItem(id=str(item_id) if item_id else "", content=content, status=status))  # type: ignore[arg-type]
    return parsed


def _status_mark(status: TodoStatus) -> str:
    """状态对应的符号:类似 checkbox 显示。"""
    if status == "completed":
        return "[x]"
    if status == "in_progress":
        return "[>]"
    return "[ ]"
