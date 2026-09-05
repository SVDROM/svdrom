from __future__ import annotations

import pytest

from svdrom.spod import SPOD


def test_spod_instantiates_without_error() -> None:
    """SPOD can be imported and instantiated as a placeholder."""
    model = SPOD(n_components=2)
    assert isinstance(model, SPOD)
    assert model._n_components == 2


def test_spod_fit_raises_not_implemented() -> None:
    """Calling fit on the placeholder raises NotImplementedError."""
    model = SPOD(n_components=2)
    with pytest.raises(NotImplementedError):
        model.fit(None)


def test_spod_reconstruct_raises_not_implemented() -> None:
    """Calling reconstruct on the placeholder raises NotImplementedError."""
    model = SPOD(n_components=2)
    with pytest.raises(NotImplementedError):
        model.reconstruct(None)
