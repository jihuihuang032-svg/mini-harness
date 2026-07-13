"""诊断命令:harness doctor 实现。

逐项检查工作区是否就绪:workspace / 配置文件 / provider / api_key /
工具配置 / 命令策略 / 审批模式等。返回 DoctorReport 供 CLI 打印或 JSON 输出。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from harness.config import CONFIG_FILE_NAME
from harness.models.providers import get_provider
from harness.runtime.executor import CommandExecutor
from harness.runtime.policy import CommandPolicy
from harness.runtime.workspace import Workspace
from harness.tools import build_default_router


@dataclass(frozen=True)
class DoctorCheck:
    """单条检查:名字 + 是否通过 + 描述信息。"""
    name: str
    ok: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "message": self.message}


@dataclass(frozen=True)
class DoctorReport:
    """诊断报告:所有检查的汇总。ok = 所有 check 都通过。"""
    ok: bool
    checks: list[DoctorCheck]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}


def run_doctor(
    workspace: str | None = None,
    config_path: str | None = None,
    provider_override: str | None = None,
    mock: bool = False,
) -> DoctorReport:
    """运行所有检查,返回 DoctorReport。

    @param workspace: 工作区目录(不传则用 HARNESS_WORKSPACE 环境变量或当前目录)
    @param config_path: harness.json 路径(不传则用 <workspace>/harness.json)
    @param provider_override: 强制使用的 provider 名称(覆盖配置)
    @param mock: mock 模式,跳过真实 provider/api_key 检查
    """
    checks: list[DoctorCheck] = []
    root = Path(workspace or os.getenv("HARNESS_WORKSPACE") or ".").resolve()
    checks.append(DoctorCheck("workspace", root.exists() and root.is_dir(), str(root)))
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    checks.append(DoctorCheck("system_prompt", prompt_path.exists() and prompt_path.stat().st_size > 0, str(prompt_path)))
    try:
        doctor_workspace = Workspace(root)
        doctor_executor = CommandExecutor(
            doctor_workspace,
            CommandPolicy.default("strict"),
            timeout_seconds=1,
            max_output_chars=1000,
        )
        tool_count = len(build_default_router(doctor_workspace, doctor_executor, 1000, tool_profile="read-only").specs())
        checks.append(DoctorCheck("tools", tool_count > 0, f"{tool_count} read-only tools available"))
    except Exception as exc:
        checks.append(DoctorCheck("tools", False, str(exc)))

    config_file = _config_path(root, config_path)
    project_config = _load_config(config_file, checks)

    provider_name = provider_override or os.getenv("HARNESS_PROVIDER") or _string_value(project_config, "provider")
    if mock:
        checks.append(DoctorCheck("provider", True, "mock"))
        checks.append(DoctorCheck("api_key", True, "not required for mock mode"))
    else:
        provider = _provider(provider_name, checks)
        base_url = os.getenv("HARNESS_BASE_URL") or _string_value(project_config, "base_url") or (provider.base_url if provider else "")
        model = os.getenv("HARNESS_MODEL") or _string_value(project_config, "model") or (provider.default_model if provider else "")
        api_key = os.getenv("HARNESS_API_KEY") or _string_value(project_config, "api_key") or ""
        if not api_key and provider is not None:
            api_key = os.getenv(provider.api_key_env, "")
        checks.append(DoctorCheck("base_url", bool(base_url), base_url or "missing"))
        checks.append(DoctorCheck("model", bool(model), model or "missing"))
        key_hint = provider.api_key_env if provider is not None else "HARNESS_API_KEY"
        checks.append(DoctorCheck("api_key", bool(api_key), f"{key_hint} {'set' if api_key else 'missing'}"))

    tool_profile = os.getenv("HARNESS_TOOL_PROFILE") or _string_value(project_config, "tool_profile") or "full"
    checks.append(DoctorCheck("tool_profile", tool_profile in {"full", "review", "read-only"}, tool_profile))
    command_profile = os.getenv("HARNESS_COMMAND_PROFILE") or _string_value(project_config, "command_profile") or "default"
    checks.append(DoctorCheck("command_profile", command_profile in {"default", "strict"}, command_profile))
    approval = os.getenv("HARNESS_APPROVAL") or _string_value(project_config, "approval") or "never"
    checks.append(DoctorCheck("approval", approval in {"never", "on-request", "auto"}, approval))
    native_tools = _bool_setting(project_config, "native_tools", "HARNESS_NATIVE_TOOLS", False)
    checks.append(DoctorCheck("native_tools", native_tools in {"true", "false"}, native_tools))
    json_mode = _bool_setting(project_config, "json_mode", "HARNESS_JSON_MODE", False)
    checks.append(DoctorCheck("json_mode", json_mode in {"true", "false"}, json_mode))
    trace_messages = _bool_setting(project_config, "trace_messages", "HARNESS_TRACE_MESSAGES", False)
    checks.append(DoctorCheck("trace_messages", trace_messages in {"true", "false"}, trace_messages))
    trace_model_responses = _bool_setting(
        project_config,
        "trace_model_responses",
        "HARNESS_TRACE_MODEL_RESPONSES",
        False,
    )
    checks.append(
        DoctorCheck(
            "trace_model_responses",
            trace_model_responses in {"true", "false"},
            trace_model_responses,
        )
    )
    retries = _number_setting(project_config, "model_max_retries", "HARNESS_MODEL_MAX_RETRIES", 2, integer=True)
    checks.append(DoctorCheck("model_max_retries", retries.isdigit(), retries))
    backoff = _number_setting(project_config, "model_retry_backoff_seconds", "HARNESS_MODEL_RETRY_BACKOFF_SECONDS", 1.0)
    checks.append(DoctorCheck("model_retry_backoff_seconds", _is_number(backoff), backoff))
    max_run_tokens = _number_setting(project_config, "max_run_tokens", "HARNESS_MAX_RUN_TOKENS", 0, integer=True)
    checks.append(DoctorCheck("max_run_tokens", max_run_tokens.isdigit(), max_run_tokens))
    return DoctorReport(ok=all(check.ok for check in checks), checks=checks)


def _config_path(root: Path, config_path: str | None) -> Path:
    """构造配置文件路径:优先 config_path,否则用 <root>/harness.json。"""
    if config_path:
        return Path(config_path).expanduser().resolve()
    return root / CONFIG_FILE_NAME


def _load_config(path: Path, checks: list[DoctorCheck]) -> dict[str, object]:
    """加载 harness.json,失败时把错误作为一条 DoctorCheck 记录。"""
    if not path.exists():
        checks.append(DoctorCheck("config", True, f"not found: {path}"))
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        checks.append(DoctorCheck("config", False, f"invalid JSON: {exc}"))
        return {}
    if not isinstance(raw, dict):
        checks.append(DoctorCheck("config", False, "config must be a JSON object"))
        return {}
    checks.append(DoctorCheck("config", True, str(path)))
    return raw


def _string_value(config: dict[str, object], key: str) -> str | None:
    value = config.get(key)
    return value if isinstance(value, str) else None


def _bool_setting(config: dict[str, object], key: str, env_name: str, default: bool) -> str:
    """从 环境变量 > 配置文件 > 默认值 取布尔配置,返回规范化字符串 'true' / 'false'。"""
    raw = os.getenv(env_name)
    if raw is not None:
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return "true"
        if value in {"0", "false", "no", "off"}:
            return "false"
        return raw
    configured = config.get(key)
    if configured is None:
        return "true" if default else "false"
    if isinstance(configured, bool):
        return "true" if configured else "false"
    return str(configured)


def _number_setting(
    config: dict[str, object],
    key: str,
    env_name: str,
    default: int | float,
    integer: bool = False,
) -> str:
    raw = os.getenv(env_name)
    if raw is not None:
        return raw
    configured = config.get(key)
    if configured is None:
        return str(default)
    if integer and isinstance(configured, int) and not isinstance(configured, bool):
        return str(configured)
    if not integer and isinstance(configured, (int, float)) and not isinstance(configured, bool):
        return str(configured)
    return str(configured)


def _is_number(value: str) -> bool:
    """判断字符串是否可解析为数字(int 或 float)。"""
    try:
        float(value)
    except ValueError:
        return False
    return True


def _provider(name: str | None, checks: list[DoctorCheck]):
    try:
        provider = get_provider(name)
    except ValueError as exc:
        checks.append(DoctorCheck("provider", False, str(exc)))
        return None
    if provider is None:
        checks.append(DoctorCheck("provider", True, "custom"))
        return None
    checks.append(DoctorCheck("provider", True, provider.name))
    return provider
