"""
svdrom: A Python package for SVD-based reduced order modeling of large datasets,
directly on your laptop.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from svdrom.dmd import OptDMD
    from svdrom.pod import POD
    from svdrom.svd import TruncatedSVD

__version__ = version(__name__)

__all__ = ("__version__", "OptDMD", "POD", "TruncatedSVD")


def __getattr__(name: str) -> object:
    """Lazy imports for the public decomposition classes."""
    if name == "OptDMD":
        from svdrom.dmd import OptDMD

        return OptDMD
    if name == "POD":
        from svdrom.pod import POD

        return POD
    if name == "TruncatedSVD":
        from svdrom.svd import TruncatedSVD

        return TruncatedSVD
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return the public names exported by this package."""
    return sorted(set(globals()) | set(__all__))
