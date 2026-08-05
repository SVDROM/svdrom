from __future__ import annotations

from svdrom._pydmd_compat import precompile_pydmd


def pytest_configure() -> None:
    """Precompile PyDMD before pytest imports it with warnings promoted to errors."""
    # this is necessary to avoid SyntaxWarnings from PyDMD raising errors during
    # pytest runs
    precompile_pydmd(force=True)
