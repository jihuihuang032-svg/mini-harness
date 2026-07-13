"""离线 mock 模型客户端。

不调用任何真实 LLM,按预设脚本返回响应,用于:
    - 离线 demo:不消耗 API 配额就能跑通整个 Agent 流程
    - 单元测试:断言可重现

每次调用 _response_for_call 按调用次数返回不同脚本响应:
    第 1 次 -> plan(制定计划)
    第 2 次 -> tool_call(调用 list_files)
    第 3 次 -> todo_update(标记完成)
    第 4 次及以后 -> final(结束)
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from harness.messages import Message


class MockModelClient:
    """确定性的 mock 模型,响应完全可预测。"""

    def __init__(self, calls: int = 0) -> None:
        # calls 用于外部观察调用了多少次,也用来选择脚本响应
        self.calls = calls

    def complete(self, messages: list[Message]) -> str:
        self.calls += 1
        return self._response_for_call(self.calls)

    def stream_complete(self, messages: list[Message]) -> Iterator[str]:
        # 流式模式下,把响应切成两段 yield,模拟流式生成
        self.calls += 1
        response = self._response_for_call(self.calls)
        midpoint = max(1, len(response) // 2)
        yield response[:midpoint]
        yield response[midpoint:]

    def _response_for_call(self, call_number: int) -> str:
        """根据调用次数返回不同脚本响应。"""
        if call_number == 1:
            # 第 1 步:制定计划
            return json.dumps(
                {
                    "type": "plan",
                    "items": [
                        {"id": "1", "content": "Inspect workspace files", "status": "in_progress"},
                        {"id": "2", "content": "Summarize the offline run", "status": "pending"},
                    ],
                }
            )
        if call_number == 2:
            # 第 2 步:调用 list_files 工具
            return json.dumps(
                {
                    "type": "tool_call",
                    "tool": "list_files",
                    "args": {"path": ".", "limit": 50},
                }
            )
        if call_number == 3:
            # 第 3 步:更新 todo 状态为完成
            return json.dumps(
                {
                    "type": "todo_update",
                    "items": [
                        {"id": "1", "status": "completed"},
                        {"id": "2", "status": "completed"},
                    ],
                }
            )
        # 第 4 步起:返回 final,结束 Agent 循环
        return json.dumps(
            {
                "type": "final",
                "content": "Offline mock run completed. The harness created a plan, called list_files, updated todos, and returned through the agent loop.",
            }
        )
