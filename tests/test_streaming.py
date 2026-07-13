from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from harness.agent import Agent
from harness.config import HarnessConfig
from harness.models.mock import MockModelClient
from harness.models.openai_compatible import OpenAICompatibleClient, extract_response_metadata, parse_openai_sse_lines, tool_specs_to_openai_tools
from harness.runtime.executor import CommandExecutor
from harness.runtime.logger import RunLogger
from harness.runtime.policy import CommandPolicy
from harness.runtime.workspace import Workspace
from harness.tools import build_default_router


class StreamingTests(unittest.TestCase):
    def test_parse_openai_sse_lines_extracts_delta_content(self) -> None:
        lines = iter(
            [
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n",
                b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n",
                b"data: [DONE]\n",
            ]
        )

        self.assertEqual(list(parse_openai_sse_lines(lines)), ["hel", "lo"])

    def test_parse_openai_sse_lines_assembles_tool_call_delta(self) -> None:
        lines = iter(
            [
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_\"}}]}}]}\n",
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n",
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"README.md\\\"}\"}}]}}]}\n",
                b"data: [DONE]\n",
            ]
        )

        chunks = list(parse_openai_sse_lines(lines))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(json.loads(chunks[0]), {"type": "tool_call", "tool": "read_file", "args": {"path": "README.md"}})

    def test_streaming_agent_logs_chunks_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("demo", encoding="utf-8")
            config = HarnessConfig.offline(str(root))
            workspace = Workspace(config.workspace)
            executor = CommandExecutor(workspace, CommandPolicy.default(), timeout_seconds=10, max_output_chars=1000)
            router = build_default_router(workspace, executor, max_output_chars=1000)
            logger = RunLogger(workspace.logs_dir)
            chunks: list[str] = []

            result = Agent(
                config,
                MockModelClient(),
                router,
                logger,
                workspace=workspace,
                stream=True,
                stream_callback=chunks.append,
            ).run("offline streaming smoke test")

            self.assertEqual(result.steps, 4)
            self.assertGreater(len(chunks), 0)
            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
            self.assertIn("model_stream_chunk", {record["kind"] for record in records})

    def test_openai_client_normalizes_native_tool_call_response(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
        )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\":\"README.md\"}",
                                },
                            }
                        ],
                    }
                }
            ]
        }

        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            text = OpenAICompatibleClient(config).complete([])

        self.assertEqual(json.loads(text), {"type": "tool_call", "tool": "read_file", "args": {"path": "README.md"}})

    def test_openai_client_records_response_metadata(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
        )
        response = {
            "id": "chatcmpl-1",
            "model": "test-model",
            "created": 123,
            "system_fingerprint": "fp_test",
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "{\"type\":\"final\",\"content\":\"ok\"}"}}],
        }
        client = OpenAICompatibleClient(config)

        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            client.complete([])

        self.assertEqual(client.last_response_metadata["id"], "chatcmpl-1")
        self.assertEqual(client.last_response_metadata["finish_reason"], "stop")
        self.assertEqual(client.last_response_metadata["usage"]["total_tokens"], 13)

    def test_extract_response_metadata_ignores_non_json_usage_values(self) -> None:
        metadata = extract_response_metadata(
            {
                "usage": {"prompt_tokens": 1, "bad": object()},
                "choices": [{"finish_reason": "length"}],
            },
            stream=False,
        )

        self.assertEqual(metadata["usage"], {"prompt_tokens": 1})
        self.assertEqual(metadata["finish_reason"], "length")

    def test_parse_openai_sse_lines_updates_metadata(self) -> None:
        lines = iter(
            [
                b"data: {\"id\":\"chunk-1\",\"model\":\"test-model\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n",
                b"data: {\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3},\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n",
                b"data: [DONE]\n",
            ]
        )
        metadata: dict[str, object] = {}

        chunks = list(parse_openai_sse_lines(lines, metadata=metadata))

        self.assertEqual(chunks, ["ok"])
        self.assertEqual(metadata["id"], "chunk-1")
        self.assertEqual(metadata["model"], "test-model")
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertEqual(metadata["usage"]["total_tokens"], 3)

    def test_tool_specs_to_openai_tools(self) -> None:
        tools = tool_specs_to_openai_tools(
            [
                {
                    "name": "read_file",
                    "description": "Read a file.",
                    "args_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        )

        self.assertEqual(tools[0]["type"], "function")
        function = tools[0]["function"]
        self.assertEqual(function["name"], "read_file")
        self.assertEqual(function["parameters"]["required"], ["path"])

    def test_openai_payload_includes_tools_when_enabled(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
            native_tools=True,
        )
        client = OpenAICompatibleClient(
            config,
            [{"name": "read_file", "description": "Read a file.", "args_schema": {"type": "object", "properties": {}}}],
        )

        payload = client._payload([], stream=False)

        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"][0]["function"]["name"], "read_file")

    def test_openai_payload_omits_tools_by_default(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
        )
        client = OpenAICompatibleClient(
            config,
            [{"name": "read_file", "description": "Read a file.", "args_schema": {"type": "object", "properties": {}}}],
        )

        payload = client._payload([], stream=False)

        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_openai_payload_includes_response_format_when_json_mode_enabled(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
            json_mode=True,
        )
        client = OpenAICompatibleClient(config)

        payload = client._payload([], stream=False)

        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_openai_payload_omits_response_format_by_default(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
        )
        client = OpenAICompatibleClient(config)

        payload = client._payload([], stream=False)

        self.assertNotIn("response_format", payload)

    def test_openai_client_retries_retryable_http_error(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
            model_max_retries=1,
            model_retry_backoff_seconds=0,
        )
        response = {"choices": [{"message": {"role": "assistant", "content": "{\"type\":\"final\",\"content\":\"ok\"}"}}]}
        calls = [
            _http_error(429, b"rate limited"),
            _FakeResponse(response),
        ]

        with patch("urllib.request.urlopen", side_effect=calls) as urlopen:
            text = OpenAICompatibleClient(config).complete([])

        self.assertEqual(text, "{\"type\":\"final\",\"content\":\"ok\"}")
        self.assertEqual(urlopen.call_count, 2)

    def test_openai_client_does_not_retry_non_retryable_http_error(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
            model_max_retries=3,
            model_retry_backoff_seconds=0,
        )

        with patch("urllib.request.urlopen", side_effect=_http_error(401, b"bad key")) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "Model API HTTP 401"):
                OpenAICompatibleClient(config).complete([])

        self.assertEqual(urlopen.call_count, 1)

    def test_openai_client_retries_url_error_until_exhausted(self) -> None:
        config = HarnessConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            workspace=Path("."),
            model_max_retries=2,
            model_retry_backoff_seconds=0,
        )

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("temporary")) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                OpenAICompatibleClient(config).complete([])

        self.assertEqual(urlopen.call_count, 3)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.test/v1/chat/completions",
        code=code,
        msg="error",
        hdrs={},
        fp=BytesIO(body),
    )


if __name__ == "__main__":
    unittest.main()
