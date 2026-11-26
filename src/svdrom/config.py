from __future__ import annotations

from typing import Any

import dask

from svdrom._backend import get_xp

_fixed_config = {
    "hankel_coord_name": "hankel_lag",
    "hankel_time_mapping_attr": "hankel_time_mapping",
}

_editable_config = {
    "stack_coord_name": "samples",
}


def _global_config() -> dict[str, str]:
    return {
        **_fixed_config,
        **_editable_config,
        "array_backend": dask.config.get("array.backend", "numpy"),
    }


def get(key: str | None = None) -> Any:
    """Get the current configuration.
    If key is None, returns the whole global config.
    Otherwise, returns the value for the given key.
    """
    if key is None:
        return _global_config()
    return _global_config().get(key)


def set(**kwargs: str) -> None:
    """Update configuration values for the current session.

    ``array_backend`` updates Dask's process-wide array backend. Use a
    ``dask.config.set`` context to scope backend selection to a block.

    Examples
    --------
    >>> import svdrom.config

    When stacking dimensions together, call the
    resulting dimension "space":
    >>> svdrom.config.set(stack_coord_name="space")
    """
    editable_keys = (*_editable_config, "array_backend")
    for k, v in kwargs.items():
        if k not in editable_keys:
            msg = (
                f"Unknown editable config key: {k}. "
                f"Editable config keys are: {list(editable_keys)}."
            )
            raise KeyError(msg)
        if k == "stack_coord_name" and (not isinstance(v, str) or not v):
            msg = "stack_coord_name must be a non-empty string"
            raise ValueError(msg)
        if k == "array_backend":
            if v not in ("numpy", "cupy"):
                msg = "array_backend must be 'numpy' or 'cupy'."
                raise ValueError(msg)
            with dask.config.set({"array.backend": v}):
                get_xp()
    if "array_backend" in kwargs:
        dask.config.set({"array.backend": kwargs["array_backend"]})
    if "stack_coord_name" in kwargs:
        _editable_config["stack_coord_name"] = kwargs["stack_coord_name"]
