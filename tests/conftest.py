from svdrom._pydmd_compat import precompile_pydmd


def pytest_configure() -> None:
    """Precompile PyDMD before pytest imports it with warnings promoted to errors."""
    precompile_pydmd(force=True)
