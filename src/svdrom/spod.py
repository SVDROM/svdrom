from __future__ import annotations

from svdrom.svdrom_base import DecompositionModel


class SPOD(DecompositionModel):
    """Spectral Proper Orthogonal Decomposition (SPOD).

    This class is a placeholder for a future SPOD implementation. It is
    not yet functional and will raise :class:`NotImplementedError` when
    its methods are called.
    """

    def __init__(self, n_components: int) -> None:
        super().__init__(n_components=n_components)

    def fit(self, *args, **kwargs) -> DecompositionModel:  # type: ignore[override]
        """Fit the SPOD model to the data.

        Raises
        ------
        NotImplementedError
            SPOD is not implemented yet.
        """
        msg = (
            "SPOD (Spectral Proper Orthogonal Decomposition) is not "
            "implemented yet and will be part of a future release."
        )
        raise NotImplementedError(msg)

    def reconstruct(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Reconstruct the data using the fitted model.

        Raises
        ------
        NotImplementedError
            SPOD is not implemented yet.
        """
        msg = (
            "SPOD (Spectral Proper Orthogonal Decomposition) is not "
            "implemented yet and will be part of a future release."
        )
        raise NotImplementedError(msg)
