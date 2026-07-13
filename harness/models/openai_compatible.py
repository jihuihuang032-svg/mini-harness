"""OpenAI 兼容的模型客户端实现。

通过 urllib 调用任何 OpenAI 兼容的 /chat/completions 接口(DeepSeek/Qwen/GLM 等),
不引入 openai SDK,保持依赖最小。

支持两种响应模式:
    - 非流式:整段返回 assistant 内容(或原生 tool_calls)
    - 流式:SSE 逐块 yield,流末尾如果是 tool_call 会合并 delta 后再 yield

错误处理:
    - 临时性 HTTP 错误(408/409/425/429/5xx)按指数退避重试
    - 认证/权限/请求格式错误立即抛出
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from harness.actions import action_from_openai_message
from harness.config import HarnessConfig
from harness.messages import Message


class OpenAICompatibleClient:
    """调用 OpenAI 兼容接口的模型客户端。"""

    def __init__(self, config: HarnessConfig, tool_specs: list[dict[str, object]] | None = None) -> None:
        """
        @param config: 全局配置(取 base_url/api_key/model/temperature 等)
        @param tool_specs: 工具规格列表,开启 native_tools 时会作为 functions 传给模型
        """
        self.config = config
        self.tool_specs = tool_specs or []
        # 保存最后一次响应的元数据(usage/finish_reason 等),供 trace 使用
        self.last_response_metadata: dict[str, object] = {}

    def complete(self, messages: list[Message]) -> str:
        """非流式调用,返回 assistant 文本或规范化的 tool_call JSON 字符串。"""
        payload = self._payload(messages, stream=False)
        request = self._request(payload)
        with self._open_with_retries(request) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.last_response_metadata = extract_response_metadata(data, stream=False)

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected model response: {data}") from exc
        if not isinstance(message, dict):
            raise RuntimeError(f"Unexpected model message: {message!r}")
        # 如果模型走原生 function calling,这里把 tool_calls 规范化为统一 JSON 字符串
        try:
            native_action = action_from_openai_message(message)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if native_action is not None:
            return native_action
        # 否则取 content 文本(模型走 JSON 协议时也是这种)
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Model content is not text: {content!r}")
        return content

    def stream_complete(self, messages: list[Message]) -> Iterator[str]:
        """流式调用,逐块 yield 文本或末尾合并后的 tool_call JSON。"""
        payload = self._payload(messages, stream=True)
        request = self._request(payload)
        metadata: dict[str, object] = {"stream": True}
        with self._open_with_retries(request) as response:
            yield from parse_openai_sse_lines(response, metadata=metadata)
        self.last_response_metadata = metadata

    def _payload(self, messages: list[Message], stream: bool) -> dict[str, object]:
        """构造请求体。原生工具协议和 JSON mode 都由配置显式开启。"""
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [message.to_api() for message in messages],
            "temperature": self.config.temperature,
            "stream": stream,
        }
        if self.config.native_tools and self.tool_specs:
            # 开启原生 function calling:把工具规格转成 OpenAI tools 字段
            payload["tools"] = tool_specs_to_openai_tools(self.tool_specs)
            payload["tool_choice"] = "auto"
        if self.config.json_mode:
            # 强制模型返回 JSON 对象(部分 provider 支持)
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _request(self, payload: dict[str, object]) -> urllib.request.Request:
        """构造 HTTP 请求对象(不发送)。"""
        url = f"{self.config.base_url}/chat/completions"
        return urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                # 流式时按 text/event-stream 接收
                "Accept": "text/event-stream" if payload.get("stream") else "application/json",
            },
            method="POST",
        )

    @contextmanager
    def _open_with_retries(self, request: urllib.request.Request) -> Iterator[object]:
        """发送请求并返回响应对象,带重试。

        @contextmanager 让此函数可以用 with 语法,类似 Java try-with-resources。
        """
        response = _call_with_retries(
            lambda: urllib.request.urlopen(request, timeout=self.config.timeout_seconds),
            max_retries=self.config.model_max_retries,
            backoff_seconds=self.config.model_retry_backoff_seconds,
        )
        with response:
            yield response


def _call_with_retries(
    operation: Callable[[], object],
    max_retries: int,
    backoff_seconds: float,
) -> object:
    """对 operation 做带退避的重试。

    只重试临时性 HTTP 错误(408/409/425/429/5xx)和网络错误,
    认证、限权和请求格式问题应立即抛出。
    """
    attempts = max(0, max_retries) + 1
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except urllib.error.HTTPError as exc:
            if attempt >= attempts or not _is_retryable_http_status(exc.code):
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Model API HTTP {exc.code}: {body}") from exc
            _sleep_before_retry(backoff_seconds, attempt)
        except urllib.error.URLError as exc:
            if attempt >= attempts:
                raise RuntimeError(f"Model API request failed after {attempt} attempts: {exc}") from exc
            _sleep_before_retry(backoff_seconds, attempt)
    raise RuntimeError("Model API request failed.")


def _is_retryable_http_status(status: int) -> bool:
    """判断 HTTP 状态码是否值得重试。"""
    return status in {408, 409, 425, 429} or 500 <= status <= 599


def _sleep_before_retry(backoff_seconds: float, attempt: int) -> None:
    """指数退避:delay = backoff_seconds * attempt。"""
    delay = max(0.0, backoff_seconds) * attempt
    if delay:
        time.sleep(delay)


def tool_specs_to_openai_tools(specs: list[dict[str, object]]) -> list[dict[str, object]]:
    """把框架内部的工具规格转换为 OpenAI tools 字段格式。

    输入:{name, description, args_schema}
    输出:[{"type": "function", "function": {name, description, parameters}}]
    """
    tools: list[dict[str, object]] = []
    for spec in specs:
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid tool spec name: {name!r}")
        description = spec.get("description")
        args_schema = spec.get("args_schema")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description if isinstance(description, str) else "",
                    "parameters": args_schema if isinstance(args_schema, dict) else {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def extract_response_metadata(data: object, stream: bool) -> dict[str, object]:
    """从非流式响应里提取元数据(id/model/usage/finish_reason 等),供 trace 用。"""
    if not isinstance(data, dict):
        return {"stream": stream}
    metadata: dict[str, object] = {"stream": stream}
    for key in ("id", "model", "created", "system_fingerprint"):
        value = data.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    usage = data.get("usage")
    if isinstance(usage, dict):
        metadata["usage"] = _json_object(usage)
    finish_reason = _finish_reason(data)
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    return metadata


def parse_openai_sse_lines(lines: Iterator[bytes], metadata: dict[str, object] | None = None) -> Iterator[str]:
    """解析 SSE 流,逐块 yield 文本。

    流式响应里:
        - 文本 delta 直接 yield
        - tool_call delta 会分片到达,需要边接收边合并,流末尾再 yield 完整的 tool_call JSON
    """
    tool_call: dict[str, object] | None = None
    yielded_content = False
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        # SSE 协议:空行是事件分隔,:开头是注释
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            # 流结束:如果有未输出的 tool_call,现在合并并输出
            if not yielded_content and tool_call is not None:
                try:
                    action = action_from_openai_message({"tool_calls": [tool_call]})
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
                if action is not None:
                    yield action
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid streaming JSON event: {data}") from exc
        if metadata is not None:
            _update_stream_metadata(metadata, event)
        chunk = _extract_delta_content(event)
        if chunk:
            # 文本 delta
            yielded_content = True
            yield chunk
            continue
        # tool_call delta,累加合并
        tool_call = _merge_tool_call_delta(tool_call, event)


def _extract_delta_content(event: object) -> str:
    """从单个 SSE event 中提取 delta.content 文本。"""
    if not isinstance(event, dict):
        return ""
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _update_stream_metadata(metadata: dict[str, object], event: object) -> None:
    """把单个 SSE event 中的元数据更新到累加器。"""
    if not isinstance(event, dict):
        return
    for key in ("id", "model", "created", "system_fingerprint"):
        value = event.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    usage = event.get("usage")
    if isinstance(usage, dict):
        metadata["usage"] = _json_object(usage)
    finish_reason = _finish_reason(event)
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason


def _finish_reason(data: dict[str, object]) -> str | None:
    """从响应里取 choices[0].finish_reason。"""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    reason = first.get("finish_reason")
    return reason if isinstance(reason, str) else None


def _json_object(raw: dict[object, object]) -> dict[str, object]:
    """把任意 dict 过滤为纯 JSON 可序列化的 dict。"""
    return {str(key): value for key, value in raw.items() if _is_json_value(value)}


def _is_json_value(value: object) -> bool:
    """递归判断 value 是否可被 JSON 序列化。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _merge_tool_call_delta(current: dict[str, object] | None, event: object) -> dict[str, object] | None:
    """把流式 tool_call delta 合并到 current。

    OpenAI 流式协议里,tool_call 的 function.name 和 function.arguments
    都可能被切成多片,需要逐字拼接。
    """
    if not isinstance(event, dict):
        return current
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return current
    first = choices[0]
    if not isinstance(first, dict):
        return current
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return current
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return current
    incoming = tool_calls[0]
    if not isinstance(incoming, dict):
        return current
    merged = dict(current or {})
    if isinstance(incoming.get("id"), str):
        merged["id"] = incoming["id"]
    if isinstance(incoming.get("type"), str):
        merged["type"] = incoming["type"]
    incoming_function = incoming.get("function")
    if isinstance(incoming_function, dict):
        function = dict(merged.get("function") if isinstance(merged.get("function"), dict) else {})
        # name 是追加式拼接
        name = incoming_function.get("name")
        if isinstance(name, str):
            function["name"] = function.get("name", "") + name
        # arguments 也是追加式拼接
        arguments = incoming_function.get("arguments")
        if isinstance(arguments, str):
            function["arguments"] = function.get("arguments", "") + arguments
        merged["function"] = function
    return merged
