"""项目初始化:harness init 命令实现。

在指定目录生成 harness.json(配置模板)和可选的 .env(环境变量模板)。
类似 Spring Initializr:把样板文件落盘,让用户后续填具体值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# harness.json 模板:provider=deepseek 的默认配置,所有字段都列出便于参考
HARNESS_JSON_TEMPLATE = """{
  "provider": "deepseek",
  "model": "deepseek-chat",
  "workspace": ".",
  "max_steps": 20,
  "timeout_seconds": 60,
  "model_max_retries": 2,
  "model_retry_backoff_seconds": 1,
  "max_tool_output_chars": 12000,
  "max_context_chars": 60000,
  "max_summary_files": 120,
  "max_run_tokens": 0,
  "temperature": 0,
  "approval": "never",
  "tool_profile": "full",
  "command_profile": "default",
  "native_tools": false,
  "json_mode": false,
  "trace_messages": false,
  "trace_model_responses": false
}
"""


# .env 模板:把所有支持的环境变量列出来,大部分用注释默认关闭
ENV_TEMPLATE = """# Provider presets: deepseek, qwen, kimi, glm, doubao
# Set HARNESS_PROVIDER plus the provider-specific key, or use HARNESS_BASE_URL/HARNESS_MODEL/HARNESS_API_KEY directly.
HARNESS_PROVIDER=deepseek
DEEPSEEK_API_KEY=replace-me
# DASHSCOPE_API_KEY=replace-me
# MOONSHOT_API_KEY=replace-me
# ZHIPU_API_KEY=replace-me
# ARK_API_KEY=replace-me

# Custom OpenAI-compatible endpoint override.
# HARNESS_BASE_URL=https://api.deepseek.com
# HARNESS_API_KEY=replace-me
# HARNESS_MODEL=deepseek-chat

HARNESS_WORKSPACE=.
HARNESS_MAX_STEPS=20
HARNESS_TIMEOUT_SECONDS=60
HARNESS_MODEL_MAX_RETRIES=2
HARNESS_MODEL_RETRY_BACKOFF_SECONDS=1
HARNESS_MAX_TOOL_OUTPUT_CHARS=12000
HARNESS_MAX_CONTEXT_CHARS=60000
HARNESS_MAX_SUMMARY_FILES=120
HARNESS_MAX_RUN_TOKENS=0
HARNESS_APPROVAL=never
HARNESS_TOOL_PROFILE=full
HARNESS_COMMAND_PROFILE=default
HARNESS_NATIVE_TOOLS=false
HARNESS_JSON_MODE=false
HARNESS_TRACE_MESSAGES=false
HARNESS_TRACE_MODEL_RESPONSES=false
# Streaming is controlled by CLI flag: python -m harness.cli run --stream ...
"""


@dataclass(frozen=True)
class InitResult:
    """单个文件的写入结果:路径 + 状态(written/skipped)。"""
    path: Path
    status: str

    def to_dict(self) -> dict[str, object]:
        return {"path": str(self.path), "status": self.status}


def init_workspace(workspace: str | None = None, include_env: bool = False, force: bool = False) -> list[InitResult]:
    """在 workspace 目录生成 harness.json(以及可选的 .env)。

    @param workspace: 工作区目录(不传则当前目录)
    @param include_env: 是否同时生成 .env 模板
    @param force: 是否覆盖已存在的文件(默认 skip)
    @return: 每个文件的写入结果
    """
    root = Path(workspace or ".").resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = [_write_template(root / "harness.json", HARNESS_JSON_TEMPLATE, force)]
    if include_env:
        results.append(_write_template(root / ".env", ENV_TEMPLATE, force))
    return results


def _write_template(path: Path, content: str, force: bool) -> InitResult:
    """写入单个模板文件,已存在且 force=False 时返回 skipped。"""
    if path.exists() and not force:
        return InitResult(path=path, status="skipped")
    path.write_text(content, encoding="utf-8")
    return InitResult(path=path, status="written")
