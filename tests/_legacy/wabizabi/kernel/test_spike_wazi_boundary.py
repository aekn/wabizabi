"""Spike 6: Wazi lazy app loading and package boundary.

Validates that:
- wazi can load agents without importing kernel internals
- wabizabi does not import from wazi (verified by import analysis)
- The kernel package boundary is clean
"""

from __future__ import annotations

import importlib
import sys


class TestPackageBoundary:
    """wabizabi kernel must not depend on wazi."""

    def test_wabizabi_kernel_does_not_import_wazi(self) -> None:
        """Importing kernel modules must not pull in wazi."""
        # Clear any cached wazi imports
        wazi_modules = [k for k in sys.modules if k.startswith("wazi")]
        for m in wazi_modules:
            del sys.modules[m]

        # Import kernel modules
        importlib.import_module("wabizabi.kernel.records")
        importlib.import_module("wabizabi.kernel.log")
        importlib.import_module("wabizabi.kernel.branch")

        # wazi should not have been imported as a side effect
        wazi_imported = [k for k in sys.modules if k.startswith("wazi.") or k == "wazi"]
        assert wazi_imported == [], f"Kernel imported wazi modules: {wazi_imported}"

    def test_wabizabi_does_not_import_wazi(self) -> None:
        """The main wabizabi package must not import from wazi."""
        wazi_modules_before = {k for k in sys.modules if k.startswith("wazi")}

        importlib.import_module("wabizabi")

        wazi_modules_after = {k for k in sys.modules if k.startswith("wazi")}
        new_wazi = wazi_modules_after - wazi_modules_before
        assert new_wazi == set(), f"wabizabi imported wazi modules: {new_wazi}"


class TestWaziLoaderDoesNotImportKernelInternals:
    """wazi.loader should work with public wabizabi API only."""

    def test_loader_imports_only_public_surface(self) -> None:
        """wazi.loader should import from wabizabi, not wabizabi.kernel."""
        loader = importlib.import_module("wazi.loader")
        source_file = loader.__file__
        assert source_file is not None

        with open(source_file) as f:
            source = f.read()

        # Loader should not import from kernel internals
        assert "wabizabi.kernel" not in source, (
            "wazi.loader imports from wabizabi.kernel — shell must not depend on kernel internals"
        )

    def test_wazi_app_imports_only_public_surface(self) -> None:
        """wazi.app should import from wabizabi public surface, not kernel."""
        app = importlib.import_module("wazi.app")
        source_file = app.__file__
        assert source_file is not None

        with open(source_file) as f:
            source = f.read()

        # App should not import from kernel internals
        assert "wabizabi.kernel" not in source, (
            "wazi.app imports from wabizabi.kernel — shell must not depend on kernel internals"
        )
