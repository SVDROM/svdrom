from __future__ import annotations

import pytest

import svdrom.config as config


@pytest.fixture(autouse=True)
def _restore_config():
    """Restore the editable config to its default after each test."""
    original = dict(config._editable_config)
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
