"""模型客户端子包。

通过本包对外暴露:
    - ModelClient:模型客户端协议(类似 Java 接口)
    - MockModelClient:离线 mock 实现
    - OpenAICompatibleClient:OpenAI 兼容 API 实现
    - ProviderPreset:provider 预设(deepseek/qwen/glm 等)
    - get_provider/provider_names/PROVIDER_PRESETS:provider 注册表
"""
from harness.models.base import ModelClient
from harness.models.mock import MockModelClient
from harness.models.providers import PROVIDER_PRESETS, ProviderPreset, get_provider, provider_names

__all__ = [
    "ModelClient",
    "MockModelClient",
    "PROVIDER_PRESETS",
    "ProviderPreset",
    "get_provider",
    "provider_names",
]
