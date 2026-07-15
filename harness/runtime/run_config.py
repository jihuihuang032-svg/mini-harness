"""Build non-secret run configuration snapshots for trace logs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from harness.config import HarnessConfig
from harness.runtime.logger import TRACE_SCHEMA_VERSION


def run_config_snapshot(
    config: HarnessConfig,
    *,
    mode: str,
    stream: bool,
    tool_profile: str,
    command_profile: str,
    approval: str,
    tool_specs: list[dict[str, object]],
    system_prompt: str,
    resume_from: str | None = None,
) -> dict[str, object]:
    tool_names = sorted(str(spec.get("name", "")) for spec in tool_specs)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "mode": mode,
        "stream": stream,
        "tool_profile": tool_profile,
        "command_profile": command_profile,
        "approval": approval,
        "resume_from": resume_from,
        "model": config.model,
        "base_url": config.base_url,
        "provider": config.provider,
        "api_key_sha256": _sha256(config.api_key),
        "workspace": str(config.workspace),
        "max_steps": config.max_steps,
        "timeout_seconds": config.timeout_seconds,
        "max_tool_output_chars": config.max_tool_output_chars,
        "max_context_chars": config.max_context_chars,
        "max_summary_files": config.max_summary_files,
        "max_run_tokens": config.max_run_tokens,
        "temperature": config.temperature,
        "model_max_retries": config.model_max_retries,
        "model_retry_backoff_seconds": config.model_retry_backoff_seconds,
        "native_tools": config.native_tools,
        "json_mode": config.json_mode,
        "trace_messages": config.trace_messages,
        "trace_model_responses": config.trace_model_responses,
        "tool_count": len(tool_specs),
        "tool_names": tool_names,
        "tool_schema_sha256": _sha256(json.dumps(tool_specs, ensure_ascii=False, sort_keys=True)),
        "system_prompt_chars": len(system_prompt),
        "system_prompt_sha256": _sha256(system_prompt),
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()