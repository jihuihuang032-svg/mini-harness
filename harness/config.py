"""配置加载模块。

负责从多个来源(优先级从高到低):
    1. CLI 命令行标志(由 cli.py 处理后以参数覆盖形式传入)
    2. 环境变量(适合放敏感信息如 API Key)
    3. harness.json 项目配置文件
    4. Provider 预设(如 deepseek 自动填 base_url)
    5. 内置默认值
装配成不可变的 HarnessConfig 对象,类似 Spring 的 @ConfigurationProperties。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from harness.models.providers import get_provider


CONFIG_FILE_NAME = "harness.json"


def _load_dotenv(path: Path) -> None:
    """简易版 .env 加载器。

    只做 KEY=VALUE 注入到 os.environ,且使用 setdefault(已存在的不覆盖),
    避免引入 python-dotenv 依赖。
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # 跳过空行和注释
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_project_config(workspace_override: str | None, config_override: str | None) -> dict[str, object]:
    """定位并加载 harness.json 文件,返回原始 dict。

    查找顺序:显式 --config > <workspace>/harness.json > <cwd>/harness.json
    """
    if config_override:
        path = Path(config_override).expanduser()
    elif workspace_override:
        path = Path(workspace_override).expanduser() / CONFIG_FILE_NAME
    else:
        path = Path.cwd() / CONFIG_FILE_NAME
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))  # utf-8-sig 兼容 BOM
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid config JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return raw


# ---- 类型化字段读取辅助:从 dict 中安全读取并校验类型 ----
# 类似 Java 中 BeanWrapper / MapStruct 的类型转换

def _string_value(config: dict[str, object], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config field {key!r} must be a string.")
    return value


def _int_value(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        # Python 中 True/False 是 int 的子类,这里显式拒绝
        raise ValueError(f"Config field {key!r} must be an integer.")
    return value


def _float_value(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config field {key!r} must be a number.")
    return float(value)


def _bool_value(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Config field {key!r} must be a boolean.")
    return value


def _env_int(name: str, config: dict[str, object], key: str, default: int) -> int:
    """优先读环境变量,其次读 config 文件,最后用 default。"""
    raw = os.getenv(name)
    return int(raw) if raw is not None else _int_value(config, key, default)


def _env_float(name: str, config: dict[str, object], key: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else _float_value(config, key, default)


def _env_bool(name: str, config: dict[str, object], key: str, default: bool) -> bool:
    """环境变量布尔解析:支持 1/true/yes/on 与 0/false/no/off。"""
    raw = os.getenv(name)
    if raw is None:
        return _bool_value(config, key, default)
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean.")


def _approval_mode(config: dict[str, object]) -> str:
    """读取审批模式:never/on-request/auto。"""
    value = os.getenv("HARNESS_APPROVAL") or _string_value(config, "approval") or "never"
    if value not in {"never", "on-request", "auto"}:
        raise ValueError("Approval mode must be one of: never, on-request, auto.")
    return value


def _tool_profile(config: dict[str, object]) -> str:
    """读取工具权限配置:full/review/read-only。"""
    value = os.getenv("HARNESS_TOOL_PROFILE") or _string_value(config, "tool_profile") or "full"
    if value not in {"full", "review", "read-only"}:
        raise ValueError("Tool profile must be one of: full, review, read-only.")
    return value


def _command_profile(config: dict[str, object]) -> str:
    """读取命令策略配置:default/strict。"""
    value = os.getenv("HARNESS_COMMAND_PROFILE") or _string_value(config, "command_profile") or "default"
    if value not in {"default", "strict"}:
        raise ValueError("Command profile must be one of: default, strict.")
    return value


@dataclass(frozen=True)  # 不可变配置对象,线程安全
class HarnessConfig:
    """全局配置。

    字段含义见 README。frozen=True 让对象不可变,
    可以放心地在多线程(HTTP server)之间共享。
    """
    base_url: str
    api_key: str
    model: str
    workspace: Path
    provider: str | None = None
    max_steps: int = 20
    timeout_seconds: int = 60
    model_max_retries: int = 2
    model_retry_backoff_seconds: float = 1.0
    max_tool_output_chars: int = 12_000  # 12_000 是 Python 数字字面量分隔符,等同 12000
    max_context_chars: int = 60_000
    max_summary_files: int = 120
    max_run_tokens: int = 0
    temperature: float = 0.0
    approval: str = "never"
    tool_profile: str = "full"
    command_profile: str = "default"
    native_tools: bool = False
    json_mode: bool = False
    trace_messages: bool = False
    trace_model_responses: bool = False

    def __post_init__(self) -> None:
        """dataclass 提供的钩子,在 __init__ 之后调用,用于校验。

        frozen dataclass 不能在普通方法里赋值,但 __post_init__ 里可以通过
        object.__setattr__ 绕过;这里只做校验。
        """
        if self.model_max_retries < 0:
            raise ValueError("model_max_retries must be non-negative.")
        if self.model_retry_backoff_seconds < 0:
            raise ValueError("model_retry_backoff_seconds must be non-negative.")
        if self.max_run_tokens < 0:
            raise ValueError("max_run_tokens must be non-negative.")

    @classmethod
    def from_env(
        cls,
        workspace_override: str | None = None,
        provider_override: str | None = None,
        config_override: str | None = None,
    ) -> "HarnessConfig":
        """从真实环境装载配置(调用真实 LLM)。

        @classmethod 类似 Java 的 static factory method。
        装配链:.env -> harness.json -> provider 预设 -> 环境变量 -> 默认值
        """
        _load_dotenv(Path.cwd() / ".env")
        project_config = _load_project_config(workspace_override, config_override)
        workspace_value = workspace_override or os.getenv("HARNESS_WORKSPACE") or _string_value(project_config, "workspace") or "."
        provider_name = provider_override or os.getenv("HARNESS_PROVIDER") or _string_value(project_config, "provider")
        provider = get_provider(provider_name)

        # 显式环境变量 > 配置文件 > provider 预设
        base_url = (os.getenv("HARNESS_BASE_URL") or _string_value(project_config, "base_url") or (provider.base_url if provider else "")).rstrip("/")
        model = os.getenv("HARNESS_MODEL") or _string_value(project_config, "model") or (provider.default_model if provider else "")
        api_key = os.getenv("HARNESS_API_KEY") or _string_value(project_config, "api_key") or ""
        if not api_key and provider is not None:
            # 使用 provider 专属的环境变量名(如 DEEPSEEK_API_KEY)
            api_key = os.getenv(provider.api_key_env, "")

        if not base_url:
            raise ValueError("HARNESS_BASE_URL is required when no provider preset is selected.")
        if not api_key:
            key_hint = f" or {provider.api_key_env}" if provider is not None else ""
            raise ValueError(f"HARNESS_API_KEY{key_hint} is required.")
        if not model:
            raise ValueError("HARNESS_MODEL is required when no provider preset is selected.")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            workspace=Path(workspace_value).resolve(),
            provider=provider.name if provider is not None else None,
            max_steps=_env_int("HARNESS_MAX_STEPS", project_config, "max_steps", 20),
            timeout_seconds=_env_int("HARNESS_TIMEOUT_SECONDS", project_config, "timeout_seconds", 60),
            model_max_retries=_env_int("HARNESS_MODEL_MAX_RETRIES", project_config, "model_max_retries", 2),
            model_retry_backoff_seconds=_env_float(
                "HARNESS_MODEL_RETRY_BACKOFF_SECONDS",
                project_config,
                "model_retry_backoff_seconds",
                1.0,
            ),
            max_tool_output_chars=_env_int("HARNESS_MAX_TOOL_OUTPUT_CHARS", project_config, "max_tool_output_chars", 12_000),
            max_context_chars=_env_int("HARNESS_MAX_CONTEXT_CHARS", project_config, "max_context_chars", 60_000),
            max_summary_files=_env_int("HARNESS_MAX_SUMMARY_FILES", project_config, "max_summary_files", 120),
            max_run_tokens=_env_int("HARNESS_MAX_RUN_TOKENS", project_config, "max_run_tokens", 0),
            temperature=_env_float("HARNESS_TEMPERATURE", project_config, "temperature", 0.0),
            approval=_approval_mode(project_config),
            tool_profile=_tool_profile(project_config),
            command_profile=_command_profile(project_config),
            native_tools=_env_bool("HARNESS_NATIVE_TOOLS", project_config, "native_tools", False),
            json_mode=_env_bool("HARNESS_JSON_MODE", project_config, "json_mode", False),
            trace_messages=_env_bool("HARNESS_TRACE_MESSAGES", project_config, "trace_messages", False),
            trace_model_responses=_env_bool(
                "HARNESS_TRACE_MODEL_RESPONSES",
                project_config,
                "trace_model_responses",
                False,
            ),
        )

    @classmethod
    def offline(cls, workspace_override: str | None = None, config_override: str | None = None) -> "HarnessConfig":
        """离线 mock 模式:不需要 API Key,base_url 填占位字符串。

        仍保留 harness.json 中除 provider/api_key 之外的配置(如 max_steps),
        因为 mock 模式也要受步数/上下文预算等约束。
        """
        _load_dotenv(Path.cwd() / ".env")
        project_config = _load_project_config(workspace_override, config_override)
        workspace_value = workspace_override or os.getenv("HARNESS_WORKSPACE") or _string_value(project_config, "workspace") or "."
        return cls(
            base_url="mock://offline",
            api_key="mock",
            model="mock-model",
            workspace=Path(workspace_value).resolve(),
            provider="mock",
            max_steps=_env_int("HARNESS_MAX_STEPS", project_config, "max_steps", 20),
            timeout_seconds=_env_int("HARNESS_TIMEOUT_SECONDS", project_config, "timeout_seconds", 60),
            model_max_retries=_env_int("HARNESS_MODEL_MAX_RETRIES", project_config, "model_max_retries", 2),
            model_retry_backoff_seconds=_env_float(
                "HARNESS_MODEL_RETRY_BACKOFF_SECONDS",
                project_config,
                "model_retry_backoff_seconds",
                1.0,
            ),
            max_tool_output_chars=_env_int("HARNESS_MAX_TOOL_OUTPUT_CHARS", project_config, "max_tool_output_chars", 12_000),
            max_context_chars=_env_int("HARNESS_MAX_CONTEXT_CHARS", project_config, "max_context_chars", 60_000),
            max_summary_files=_env_int("HARNESS_MAX_SUMMARY_FILES", project_config, "max_summary_files", 120),
            max_run_tokens=_env_int("HARNESS_MAX_RUN_TOKENS", project_config, "max_run_tokens", 0),
            temperature=0.0,
            approval=_approval_mode(project_config),
            tool_profile=_tool_profile(project_config),
            command_profile=_command_profile(project_config),
            native_tools=_env_bool("HARNESS_NATIVE_TOOLS", project_config, "native_tools", False),
            json_mode=_env_bool("HARNESS_JSON_MODE", project_config, "json_mode", False),
            trace_messages=_env_bool("HARNESS_TRACE_MESSAGES", project_config, "trace_messages", False),
            trace_model_responses=_env_bool(
                "HARNESS_TRACE_MODEL_RESPONSES",
                project_config,
                "trace_model_responses",
                False,
            ),
        )
