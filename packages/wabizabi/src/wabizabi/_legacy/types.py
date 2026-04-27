"""Shared public typing primitives."""

from __future__ import annotations

from pydantic import TypeAdapter

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def json_value_from_unknown(value: object) -> JsonValue:
    """Validate arbitrary Python data as a JSON-compatible value."""
    return _JSON_VALUE_ADAPTER.validate_python(value)


def json_object_from_unknown(value: object) -> JsonObject:
    """Validate arbitrary Python data as a JSON object."""
    return _JSON_OBJECT_ADAPTER.validate_python(value)


__all__ = [
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "json_object_from_unknown",
    "json_value_from_unknown",
]
