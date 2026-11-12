from __future__ import annotations

import pytest

import svdrom


@pytest.mark.parametrize(
    "name",
    ["POD", "TruncatedSVD", "OptDMD"],
)
def test_top_level_exports(name: str) -> None:
    """Main decomposition classes are available from the top-level namespace."""
    assert name in dir(svdrom)
    assert name in svdrom.__all__


def test_spod_not_top_level_export() -> None:
    """SPOD is not exposed at the top level while it remains a placeholder."""
    assert "SPOD" not in svdrom.__all__
    assert not hasattr(svdrom, "SPOD")
