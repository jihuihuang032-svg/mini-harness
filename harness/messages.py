"""消息数据结构。

一条 Message = 一个角色 + 一段内容。
对应 OpenAI Chat Completions 协议里的 {role, content}。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen=True:消息不可变,便于在多线程间共享
class Message:
    """对话中的一条消息。

    类比 Java record:不可变、有自动生成的 equals/hashCode/构造器。
    """
    role: str    # 取值如 "system" / "user" / "assistant" / "tool"
    content: str

    def to_api(self) -> dict[str, str]:
        """序列化为 OpenAI 协议格式 {"role": ..., "content": ...}。"""
        return {"role": self.role, "content": self.content}

    # to_dict 作为 to_api 的别名,便于 Java 习惯阅读
    to_dict = to_api
