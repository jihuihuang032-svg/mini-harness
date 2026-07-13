"""模型客户端接口定义。

通过 typing.Protocol 定义"鸭子类型接口":不需要显式继承,
只要实现了 complete / stream_complete 两个方法就被视为 ModelClient。
类似 Java 中的 interface,但更松( structural typing 而非 nominal)。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from harness.messages import Message


class ModelClient(Protocol):
    """模型客户端协议。

    所有模型客户端(真实 LLM、mock)都要满足这两个方法,
    Agent 只依赖此协议,不绑死具体实现 —— 类似 Spring 依赖接口编程。
    """

    def complete(self, messages: list[Message]) -> str:
        """同步生成下一轮回复(整段返回)。"""

    def stream_complete(self, messages: list[Message]) -> Iterator[str]:
        """流式生成下一轮回复,逐块 yield。
        类似 Java 8 的 Stream<String>,惰性产出。
        """
