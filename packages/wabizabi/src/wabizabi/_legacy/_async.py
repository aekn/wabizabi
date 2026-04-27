"""Shared async utilities."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable


def is_async_callable(value: object) -> bool:
    """Return True if *value* is an async callable."""
    if inspect.iscoroutinefunction(value):
        return True
    if not callable(value):
        return False
    return inspect.iscoroutinefunction(type(value).__call__)


async def resolve[T](result: T | Awaitable[T]) -> T:
    """Await a value if it is awaitable, otherwise return it directly.

    Used to normalize sync/async hook, processor, and validator callables.
    """
    if inspect.isawaitable(result):
        return await result
    return result


__all__ = ["is_async_callable", "resolve"]
