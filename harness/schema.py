"""工具参数 JSON Schema 校验。

模型给出的工具参数不能直接信任。这里实现一个轻量 schema 子集，负责：
- 检查必填字段
- 注入 default 默认值
- 校验基础类型、enum、数组和对象
失败时抛出 SchemaValidationError，让 agent 把错误反馈给模型继续修正。
"""

from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    """工具参数校验失败。"""


def validate_args(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """校验并返回参数副本，避免修改调用方传入的原始 dict。"""
    errors: list[str] = []
    validated = dict(arguments)
    _apply_defaults(schema, validated)
    _validate_value(schema, validated, "$", errors)
    if errors:
        raise SchemaValidationError("; ".join(errors))
    return validated


def _apply_defaults(schema: dict[str, Any], value: Any) -> None:
    # default 只在 object.properties 层注入，足够覆盖当前工具参数协议。
    if schema.get("type") != "object" or not isinstance(value, dict):
        return
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return
    for name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue
        if name not in value and "default" in prop_schema:
            value[name] = prop_schema["default"]
        elif name in value:
            _apply_defaults(prop_schema, value[name])


def _validate_value(schema: dict[str, Any], value: Any, path: str, errors: list[str]) -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_type(expected_type, value):
        errors.append(_type_error(path, expected_type, value))
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in enum {schema['enum']}")
        return
    if expected_type == "object":
        _validate_object(schema, value, path, errors)
    elif expected_type == "array":
        _validate_array(schema, value, path, errors)


def _validate_object(schema: dict[str, Any], value: dict[str, Any], path: str, errors: list[str]) -> None:
    required = schema.get("required", [])
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in value:
                errors.append(f"Missing required arg: {name}")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return
    for name, prop_schema in properties.items():
        if isinstance(name, str) and isinstance(prop_schema, dict) and name in value:
            _validate_value(prop_schema, value[name], f"{path}.{name}", errors)


def _validate_array(schema: dict[str, Any], value: list[Any], path: str, errors: list[str]) -> None:
    items_schema = schema.get("items")
    if not isinstance(items_schema, dict):
        return
    for index, item in enumerate(value):
        _validate_value(items_schema, item, f"{path}[{index}]", errors)


def _type_error(path: str, expected: str, value: Any) -> str:
    arg_name = path.rsplit(".", 1)[-1] if "." in path else path
    readable = {
        "string": "string",
        "number": "number",
        "integer": "integer",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }.get(expected, expected)
    if arg_name and arg_name != "$":
        return f"Arg '{arg_name}' must be {readable}"
    return f"{path}: expected type {expected}, got {type(value).__name__}"


def _matches_type(expected: str, value: Any) -> bool:
    # bool 是 int 的子类；工具参数里 integer/number 必须显式排除 bool。
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False
