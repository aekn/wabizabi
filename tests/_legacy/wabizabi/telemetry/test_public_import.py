"""Tests for optional public OpenTelemetry imports."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_public_telemetry_import_survives_missing_optional_otel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("opentelemetry"):
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    saved_modules: dict[str, ModuleType | None] = {
        name: sys.modules.pop(name, None)
        for name in ("wabizabi.telemetry", "wabizabi.telemetry.otel")
    }
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    try:
        telemetry = importlib.import_module("wabizabi.telemetry")
        recorder_type = telemetry.OpenTelemetryRecorder
        with pytest.raises(RuntimeError, match="wabizabi\\[otel\\]"):
            recorder_type()
    finally:
        sys.modules.pop("wabizabi.telemetry", None)
        sys.modules.pop("wabizabi.telemetry.otel", None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
