"""运行配置快照:每次 run 启动时,把关键配置序列化为可读 dict 落盘到 trace。

目的:事后审计"这次 run 是用什么参数跑的"——模型名、provider、approval 模式、
工具可见性、context 预算等。为了安全,api_key 和 system_prompt 原文不会写入,
只写它们的 sha256 指纹(便于对比是否变化)。
"""

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
    """构造运行配置快照 dict,会被 logger.event("run_config", ...) 写入 trace。

    @param config: 全局配置
    @param mode: "mock" 或 "model"
    @param stream: 是否流式输出
    @param tool_profile: 工具 profile(full/review/read-only)
    @param command_profile: 命令 profile(default/strict)
    @param approval: 审批模式(never/on-request/auto)
    @param tool_specs: 暴露给模型的工具规格列表(已经过 profile 过滤)
    @param system_prompt: 渲染后的 system prompt 原文(只写 sha256,不写原文)
    @param resume_from: 若为 resume,则填源 run_id
    @return: 可序列化 dict,包含所有需要审计的运行参数
    """
    # 工具名按字母排序,确保 diff 时顺序稳定
    tool_names = sorted(str(spec.get("name", "")) for spec in tool_specs)
    return {
        # 元数据
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        # 运行模式参数
        "mode": mode,
        "stream": stream,
        "tool_profile": tool_profile,
        "command_profile": command_profile,
        "approval": approval,
        "resume_from": resume_from,
        # 模型配置(不写 api_key,只写它的 sha256 指纹)
        "model": config.model,
        "base_url": config.base_url,
        "provider": config.provider,
        "api_key_sha256": _sha256(config.api_key),
        # 工作区与预算
        "workspace": str(config.workspace.root),
        "max_steps": config.max_steps,
        "timeout_seconds": config.timeout_seconds,
        "max_tool_output_chars": config.max_tool_output_chars,
        "max_context_chars": config.max_context_chars,
        "max_summary_files": config.max_summary_files,
        "max_run_tokens": config.max_run_tokens,
        "temperature": config.temperature,
        "model_max_retries": config.model_max_retries,
        "model_retry_backoff_seconds": config.model_retry_backoff_seconds,
        # 模型行为开关
        "native_tools": config.native_tools,
        "json_mode": config.json_mode,
        "trace_messages": config.trace_messages,
        "trace_model_responses": config.trace_model_responses,
        # 工具集合的指纹(便于对比两次 run 是否暴露了相同工具)
        "tool_count": len(tool_specs),
        "tool_names": tool_names,
        "tool_schema_sha256": _sha256(json.dumps(tool_specs, ensure_ascii=False, sort_keys=True)),
        # system prompt 的指纹(不写原文,避免泄露 prompt 内容)
        "system_prompt_chars": len(system_prompt),
        "system_prompt_sha256": _sha256(system_prompt),
    }


def _sha256(text: str) -> str:
    """计算字符串 sha256 的十六进制摘要(64 字符)。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
