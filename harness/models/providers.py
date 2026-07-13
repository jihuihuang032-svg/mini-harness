"""Provider 预设。

每个 provider(如 DeepSeek/Qwen/GLM)对应一组默认配置:
    - base_url:API 入口
    - default_model:默认模型名
    - api_key_env:对应的 API Key 环境变量名
config.py 会查找匹配的预设,自动填默认值,简化用户配置。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    """单个 provider 预设,不可变。"""
    name: str              # 唯一标识(小写)
    display_name: str      # 展示名
    base_url: str          # API 入口
    default_model: str     # 默认模型
    api_key_env: str       # API Key 环境变量名
    notes: str = ""        # 备注

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "api_key_env": self.api_key_env,
            "notes": self.notes,
        }


# 内置 provider 预设注册表:provider 名 -> ProviderPreset
# 类似 Spring 中用 @Bean 装配的一组固定 Bean
PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        name="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        notes="OpenAI-compatible DeepSeek API.",
    ),
    "qwen": ProviderPreset(
        name="qwen",
        display_name="Qwen DashScope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        notes="Alibaba Cloud DashScope OpenAI-compatible mode.",
    ),
    "kimi": ProviderPreset(
        name="kimi",
        display_name="Kimi Moonshot",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        api_key_env="MOONSHOT_API_KEY",
        notes="Moonshot AI OpenAI-compatible API.",
    ),
    "glm": ProviderPreset(
        name="glm",
        display_name="GLM Zhipu",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        api_key_env="ZHIPU_API_KEY",
        notes="Zhipu/BigModel OpenAI-compatible API.",
    ),
    "doubao": ProviderPreset(
        name="doubao",
        display_name="Doubao Ark",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="doubao-seed-1-6",
        api_key_env="ARK_API_KEY",
        notes="Volcengine Ark OpenAI-compatible API. Model names often use endpoint ids in production.",
    ),
}


def provider_names() -> list[str]:
    """返回所有 provider 名(按字母序)。"""
    return sorted(PROVIDER_PRESETS)


def get_provider(name: str | None) -> ProviderPreset | None:
    """按名查找 provider 预设。

    - name 为空 / "custom" / "none" -> 返回 None(用户走自定义配置)
    - 未注册的 name -> 抛 ValueError 并列出可用 provider
    """
    if not name:
        return None
    key = name.strip().lower()
    if key in {"", "custom", "none"}:
        return None
    try:
        return PROVIDER_PRESETS[key]
    except KeyError as exc:
        available = ", ".join(provider_names())
        raise ValueError(f"Unknown provider {name!r}. Available providers: {available}") from exc
