"""Shared tool-schema helpers."""

from __future__ import annotations

from pydantic import BaseModel, TypeAdapter

from wabizabi.types import JsonObject, JsonValue

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def _strip_titles(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_strip_titles(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_titles(item) for key, item in value.items() if key != "title"}
    return value


def tool_input_schema(arguments_type: type[BaseModel]) -> JsonObject:
    """Return a provider-neutral input schema for a tool arguments model."""
    schema = arguments_type.model_json_schema()
    return _JSON_OBJECT_ADAPTER.validate_python(_strip_titles(schema))


__all__ = ["tool_input_schema"]
