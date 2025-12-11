_fixed_config = {
    "hankel_coord_name": "hankel_lag",
    "hankel_original_time_attr": "original_time",
}

_editable_config = {
    "stack_coord_name": "samples",
}


def _global_config():
    return {**_fixed_config, **_editable_config}


def get(key=None):
    """Get the current configuration.
    If key is None, returns the whole global config.
    Otherwise, returns the value for the given key.
    """
    if key is None:
        return _global_config()
    return _global_config().get(key)


def set(**kwargs):
    """Update configuration values for the current session.

    Examples
    --------
    >>> import svdrom.config

    When stacking dimensions together, call the
    resulting dimension "space":
    >>> svdrom.config.set(stack_coord_name="space")
    """
    for k, v in kwargs.items():
        if k in _editable_config:
            _editable_config[k] = v
        else:
            msg = (
                f"Unknown editable config key: {k}. "
                f"Editable config keys are: {list(_editable_config.keys())}."
            )
            raise KeyError(msg)
