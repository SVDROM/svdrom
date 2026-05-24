from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import warnings


def _syntax_warnings_are_errors() -> bool:
    return any(
        action == "error" and issubclass(SyntaxWarning, category)
        for action, _, category, _, _ in warnings.filters
    )


def precompile_pydmd(*, force: bool = False) -> None:
    """Import PyDMD once with normal warning handling to populate bytecode caches."""
    if not force and not _syntax_warnings_are_errors():
        return

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "default"

    with contextlib.suppress(OSError):
        subprocess.run(
            [sys.executable, "-c", "import pydmd"],
            check=False,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
