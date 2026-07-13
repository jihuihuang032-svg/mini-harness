from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from harness.doctor import run_doctor


class DoctorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = [
            "HARNESS_PROVIDER",
            "HARNESS_API_KEY",
            "HARNESS_BASE_URL",
            "HARNESS_MODEL",
            "HARNESS_NATIVE_TOOLS",
            "HARNESS_JSON_MODE",
            "HARNESS_TRACE_MESSAGES",
            "HARNESS_TRACE_MODEL_RESPONSES",
            "HARNESS_MODEL_MAX_RETRIES",
            "HARNESS_MODEL_RETRY_BACKOFF_SECONDS",
            "HARNESS_MAX_RUN_TOKENS",
            "DEEPSEEK_API_KEY",
        ]
        self.old = {key: os.environ.get(key) for key in self.keys}
        for key in self.keys:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in self.keys:
            os.environ.pop(key, None)
            if self.old[key] is not None:
                os.environ[key] = self.old[key] or ""

    def test_mock_doctor_does_not_require_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_doctor(tmp, mock=True)

            self.assertTrue(report.ok)
            checks = {check.name: check for check in report.checks}
            self.assertTrue(checks["system_prompt"].ok)
            self.assertTrue(checks["system_prompt"].message.endswith("system.md"))
            self.assertTrue(checks["tools"].ok)
            self.assertIn("read-only tools available", checks["tools"].message)
            self.assertEqual(checks["provider"].message, "mock")
            self.assertEqual(checks["api_key"].message, "not required for mock mode")
            self.assertEqual(checks["command_profile"].message, "default")
            self.assertEqual(checks["native_tools"].message, "false")
            self.assertEqual(checks["json_mode"].message, "false")
            self.assertEqual(checks["trace_messages"].message, "false")
            self.assertEqual(checks["trace_model_responses"].message, "false")
            self.assertEqual(checks["model_max_retries"].message, "2")
            self.assertEqual(checks["model_retry_backoff_seconds"].message, "1.0")
            self.assertEqual(checks["max_run_tokens"].message, "0")

    def test_provider_doctor_reports_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_doctor(tmp, provider_override="deepseek")

            self.assertFalse(report.ok)
            checks = {check.name: check for check in report.checks}
            self.assertTrue(checks["provider"].ok)
            self.assertFalse(checks["api_key"].ok)
            self.assertIn("DEEPSEEK_API_KEY", checks["api_key"].message)

    def test_provider_doctor_accepts_provider_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["DEEPSEEK_API_KEY"] = "test-key"

            report = run_doctor(tmp, provider_override="deepseek")

            self.assertTrue(report.ok)

    def test_doctor_reports_invalid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "harness.json"
            config_path.write_text("{bad", encoding="utf-8")

            report = run_doctor(tmp, config_path=str(config_path), mock=True)

            self.assertFalse(report.ok)
            checks = {check.name: check for check in report.checks}
            self.assertFalse(checks["config"].ok)


if __name__ == "__main__":
    unittest.main()
