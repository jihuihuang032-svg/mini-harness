from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.config import HarnessConfig
from harness.runtime.run_config import run_config_snapshot


class RunConfigSnapshotTests(unittest.TestCase):
    def test_run_config_snapshot_includes_stable_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = HarnessConfig.offline(tmp)
            tool_specs = [
                {
                    "name": "read_file",
                    "description": "Read a file.",
                    "args_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ]

            snapshot = run_config_snapshot(
                config,
                mode="mock",
                stream=False,
                tool_profile="read-only",
                command_profile="strict",
                approval="never",
                tool_specs=tool_specs,
                system_prompt="system prompt",
            )

            self.assertEqual(snapshot["tool_count"], 1)
            self.assertEqual(snapshot["tool_names"], ["read_file"])
            self.assertEqual(snapshot["max_run_tokens"], 0)
            self.assertEqual(snapshot["json_mode"], False)
            self.assertEqual(snapshot["trace_messages"], False)
            self.assertEqual(snapshot["trace_model_responses"], False)
            self.assertEqual(len(snapshot["tool_schema_sha256"]), 64)
            self.assertEqual(len(snapshot["system_prompt_sha256"]), 64)
            self.assertNotIn("api_key", snapshot)
            self.assertNotIn("system_prompt", snapshot)


if __name__ == "__main__":
    unittest.main()
