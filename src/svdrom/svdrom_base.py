from abc import ABC, abstractmethod

import xarray as xr

from svdrom.logger import setup_logger

logger = setup_logger("Base", "base.log")


class DecompositionModel(ABC):
    """Abstract Base Class for all SVD-based Reduced Order Models.

    Enforces a common interface for SVD, POD, DMD, SPOD, etc.
    """

    def __init__(self, n_components: int) -> None:
        """
        Parameters
        ----------
        n_components : int
            The number of components/modes to keep.
        """
        self._n_components = n_components

    @abstractmethod
    def fit(self, *args, **kwargs) -> "DecompositionModel":
        """Fit the model to the data.

        Parameters
        ----------
        *args : list
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """

    @abstractmethod
    def reconstruct(
        self, *args, **kwargs
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """Reconstruct the data using the fitted model.

        Returns
        -------
        xr.DataArray
            The reconstructed data.
        """

    def _check_is_fitted(self, attributes: list[str]) -> None:
        """Checks if the model is fitted by verifying the existence
        of specific attributes.

        Parameters
        ----------
        attributes : List[str]
            List of attribute names to check (e.g. ['_u', '_s']).
        """
        for attr in attributes:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                msg = (
                    f"This {self.__class__.__name__} instance is not fitted yet. "
                    "Call 'fit' with appropriate arguments before using this estimator."
                )
                logger.error(msg)
                raise RuntimeError(msg)
