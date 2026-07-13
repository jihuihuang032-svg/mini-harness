from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from harness.config import HarnessConfig


class ConfigProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = [
            "HARNESS_PROVIDER",
            "HARNESS_BASE_URL",
            "HARNESS_API_KEY",
            "HARNESS_MODEL",
            "HARNESS_WORKSPACE",
            "HARNESS_MAX_STEPS",
            "HARNESS_TIMEOUT_SECONDS",
            "HARNESS_MODEL_MAX_RETRIES",
            "HARNESS_MODEL_RETRY_BACKOFF_SECONDS",
            "HARNESS_MAX_TOOL_OUTPUT_CHARS",
            "HARNESS_MAX_CONTEXT_CHARS",
            "HARNESS_MAX_SUMMARY_FILES",
            "HARNESS_MAX_RUN_TOKENS",
            "HARNESS_TEMPERATURE",
            "HARNESS_APPROVAL",
            "HARNESS_TOOL_PROFILE",
            "HARNESS_COMMAND_PROFILE",
            "HARNESS_NATIVE_TOOLS",
            "HARNESS_JSON_MODE",
            "HARNESS_TRACE_MESSAGES",
            "HARNESS_TRACE_MODEL_RESPONSES",
            "DEEPSEEK_API_KEY",
            "DASHSCOPE_API_KEY",
        ]
        self.old = {key: os.environ.get(key) for key in self.keys}
        for key in self.keys:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in self.keys:
            os.environ.pop(key, None)
            if self.old[key] is not None:
                os.environ[key] = self.old[key] or ""

    def test_provider_supplies_base_url_model_and_provider_key(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "test-key"

        config = HarnessConfig.from_env(provider_override="deepseek")

        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.api_key, "test-key")

    def test_env_overrides_provider_defaults(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "provider-key"
        os.environ["HARNESS_API_KEY"] = "generic-key"
        os.environ["HARNESS_BASE_URL"] = "https://example.test/v1"
        os.environ["HARNESS_MODEL"] = "custom-model"

        config = HarnessConfig.from_env(provider_override="deepseek")

        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.model, "custom-model")
        self.assertEqual(config.api_key, "generic-key")

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            HarnessConfig.from_env(provider_override="missing")

    def test_config_file_supplies_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text(
                """{
  "provider": "deepseek",
  "api_key": "file-key",
  "max_steps": 7,
  "timeout_seconds": 9,
  "model_max_retries": 4,
  "model_retry_backoff_seconds": 0.5,
  "max_tool_output_chars": 1000,
  "max_context_chars": 2000,
  "max_summary_files": 11,
  "max_run_tokens": 99,
  "temperature": 0.2,
  "approval": "auto",
  "tool_profile": "review",
  "command_profile": "strict",
  "native_tools": true,
  "json_mode": true,
  "trace_messages": true,
  "trace_model_responses": true
}""",
                encoding="utf-8",
            )

            config = HarnessConfig.from_env(workspace_override=tmp, config_override=str(config_path))

            self.assertEqual(config.provider, "deepseek")
            self.assertEqual(config.api_key, "file-key")
            self.assertEqual(config.max_steps, 7)
            self.assertEqual(config.timeout_seconds, 9)
            self.assertEqual(config.model_max_retries, 4)
            self.assertEqual(config.model_retry_backoff_seconds, 0.5)
            self.assertEqual(config.max_tool_output_chars, 1000)
            self.assertEqual(config.max_context_chars, 2000)
            self.assertEqual(config.max_summary_files, 11)
            self.assertEqual(config.max_run_tokens, 99)
            self.assertEqual(config.temperature, 0.2)
            self.assertEqual(config.approval, "auto")
            self.assertEqual(config.tool_profile, "review")
            self.assertEqual(config.command_profile, "strict")
            self.assertTrue(config.native_tools)
            self.assertTrue(config.json_mode)
            self.assertTrue(config.trace_messages)
            self.assertTrue(config.trace_model_responses)

    def test_env_overrides_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text(
                """{
  "provider": "deepseek",
  "api_key": "file-key",
  "model": "file-model",
  "max_steps": 7
}""",
                encoding="utf-8",
            )
            os.environ["HARNESS_MODEL"] = "env-model"
            os.environ["HARNESS_MAX_STEPS"] = "3"
            os.environ["HARNESS_NATIVE_TOOLS"] = "yes"
            os.environ["HARNESS_JSON_MODE"] = "true"
            os.environ["HARNESS_TRACE_MESSAGES"] = "true"
            os.environ["HARNESS_TRACE_MODEL_RESPONSES"] = "true"
            os.environ["HARNESS_MODEL_MAX_RETRIES"] = "5"
            os.environ["HARNESS_MODEL_RETRY_BACKOFF_SECONDS"] = "0.25"
            os.environ["HARNESS_MAX_RUN_TOKENS"] = "123"

            config = HarnessConfig.from_env(workspace_override=tmp, config_override=str(config_path))

            self.assertEqual(config.model, "env-model")
            self.assertEqual(config.max_steps, 3)
            self.assertTrue(config.native_tools)
            self.assertTrue(config.json_mode)
            self.assertTrue(config.trace_messages)
            self.assertTrue(config.trace_model_responses)
            self.assertEqual(config.model_max_retries, 5)
            self.assertEqual(config.model_retry_backoff_seconds, 0.25)
            self.assertEqual(config.max_run_tokens, 123)

    def test_invalid_tool_profile_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"tool_profile": "danger"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                HarnessConfig.offline(workspace_override=tmp, config_override=str(config_path))

    def test_invalid_command_profile_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"command_profile": "unsafe"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                HarnessConfig.offline(workspace_override=tmp, config_override=str(config_path))

    def test_invalid_native_tools_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"native_tools": "yes"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                HarnessConfig.offline(workspace_override=tmp, config_override=str(config_path))

    def test_invalid_json_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"json_mode": "yes"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                HarnessConfig.offline(workspace_override=tmp, config_override=str(config_path))

    def test_invalid_trace_messages_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"trace_messages": "yes"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                HarnessConfig.offline(workspace_override=tmp, config_override=str(config_path))

    def test_invalid_trace_model_responses_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"trace_model_responses": "yes"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                HarnessConfig.offline(workspace_override=tmp, config_override=str(config_path))

    def test_negative_model_retry_settings_raise(self) -> None:
        with self.assertRaises(ValueError):
            HarnessConfig(
                base_url="https://example.test/v1",
                api_key="test-key",
                model="test-model",
                workspace=Path("."),
                model_max_retries=-1,
            )
        with self.assertRaises(ValueError):
            HarnessConfig(
                base_url="https://example.test/v1",
                api_key="test-key",
                model="test-model",
                workspace=Path("."),
                model_retry_backoff_seconds=-0.1,
            )
        with self.assertRaises(ValueError):
            HarnessConfig(
                base_url="https://example.test/v1",
                api_key="test-key",
                model="test-model",
                workspace=Path("."),
                max_run_tokens=-1,
            )

    def test_config_file_allows_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text('{"max_steps": 2}', encoding="utf-8-sig")

            config = HarnessConfig.offline(workspace_override=tmp, config_override=str(config_path))

            self.assertEqual(config.max_steps, 2)


if __name__ == "__main__":
    unittest.main()
