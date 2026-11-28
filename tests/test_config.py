from __future__ import annotations

import dask
import pytest

import svdrom.config as config


@pytest.fixture(autouse=True)
def _restore_config():
    """Restore the editable config to its default after each test."""
    original = dict(config._editable_config)
    with dask.config.set({"array.backend": "numpy"}):
        yield
    config._editable_config.clear()
    config._editable_config.update(original)


def test_get_whole_config():
    """get() without a key returns the full merged config."""
    config.set(stack_coord_name="samples")
    cfg = config.get()
    assert cfg["hankel_coord_name"] == "hankel_lag"
    assert cfg["hankel_time_mapping_attr"] == "hankel_time_mapping"
    assert cfg["stack_coord_name"] == "samples"


def test_get_single_key():
    """get(key) returns the value for the given key."""
    config.set(stack_coord_name="samples")
    assert config.get("stack_coord_name") == "samples"


def test_get_unknown_key_returns_none():
    """get() with an unknown key returns None."""
    assert config.get("does_not_exist") is None


def test_set_editable_key():
    """set() updates an editable config key."""
    config.set(stack_coord_name="space")
    assert config.get("stack_coord_name") == "space"


@pytest.mark.parametrize("value", ["", 123, None])
def test_set_stack_coord_name_invalid(value):
    """set() rejects a non-string or empty stack_coord_name."""
    with pytest.raises(ValueError, match="non-empty string"):
        config.set(stack_coord_name=value)


def test_set_unknown_key_raises():
    """set() raises KeyError for an unknown editable config key."""
    with pytest.raises(KeyError, match="Unknown editable config key"):
        config.set(unknown_key="value")


def test_backend_tracks_dask() -> None:
    """The svdrom backend key reflects Dask, including temporary contexts."""
    config.set(array_backend="numpy")
    with dask.config.set({"array.backend": "cupy"}):
        assert config.get("array_backend") == "cupy", "Expected Dask's active backend."
    assert config.get("array_backend") == "numpy", "Expected the restored backend."


@pytest.mark.parametrize("value", ["cuda", "", None, 42, []])
def test_invalid_backend_is_atomic(value: object) -> None:
    """Reject invalid backends without partially updating other keys."""
    original = config.get()
    with pytest.raises(ValueError, match="array_backend"):
        config.set(stack_coord_name="changed", array_backend=value)
    assert config.get() == original, "Expected all configuration to remain unchanged."
    assert (
        dask.config.get("array.backend") == "numpy"
    ), "Expected unchanged Dask config."


def test_backend_sets_dask_and_switches_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend selection forwards to Dask and can be reset without importing CuPy."""
    monkeypatch.setattr(config, "get_xp", lambda: None)
    config.set(array_backend="cupy")
    assert (
        dask.config.get("array.backend") == "cupy"
    ), "Expected CuPy configured in Dask."
    config.set(array_backend="numpy")
    assert (
        config.get("array_backend") == "numpy"
    ), "Expected NumPy configured in svdrom."
    assert (
        dask.config.get("array.backend") == "numpy"
    ), "Expected NumPy restored in Dask."


def test_failed_backend_import_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    """An optional dependency failure leaves both configurations unchanged."""

    def unavailable() -> None:
        msg = "CuPy missing"
        raise ImportError(msg)

    monkeypatch.setattr(config, "get_xp", unavailable)
    original = config.get()
    with pytest.raises(ImportError, match="CuPy missing"):
        config.set(stack_coord_name="changed", array_backend="cupy")
    assert config.get() == original, "Expected unchanged svdrom configuration."
    assert (
        dask.config.get("array.backend") == "numpy"
    ), "Expected unchanged Dask backend."
