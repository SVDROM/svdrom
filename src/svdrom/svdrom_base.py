from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
import xarray as xr
import dask.array as da 

from svdrom.logger import setup_logger

logger = setup_logger("Base", "base.log")


class DecompositionModel(ABC):
    """Abstract Base Class for all SVD-based Reduced Order Models.

    Enforces a common interface for SVD, POD, DMD, SPOD, etc.
    """

    def __init__(self, n_components: int):
        """
        Parameters
        ----------
        n_components : int
            The number of components/modes to keep.
        """
        self._n_components = n_components
        self._u: xr.DataArray | None = None
        self._s: np.ndarray | None = None
        self._v: xr.DataArray | None = None

    @property
    def n_components(self) -> int:
        """Number of components/modes (read-only)."""
        return self._n_components

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """Fit the model to the data.

        Parameters
        ----------
        *args : list
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        pass

    @abstractmethod
    def reconstruct(self, *args, **kwargs) -> xr.DataArray:
        """Reconstruct the data using the fitted model.

        Returns
        -------
        xr.DataArray
            The reconstructed data.
        """
        pass

    def _check_is_fitted(self, attributes: List[str]) -> None:
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
                logger.exception(msg)
                raise RuntimeError(msg)

    def reconstruct(
        self,
        time: slice | int | str | None = None,
        time_dimension: str = "time",
        memory_limit_bytes: float = 1e9,
    ) -> xr.DataArray:
        """Reconstruct the data using the fitted model components (U, S, V).

        This method computes the approximation $X \\approx U \\Sigma V^T$.
        It supports reconstructing specific snapshots, slices of time, or the
        entire dataset. It automatically handles memory management by switching
        to Dask if the estimated reconstruction size exceeds the limit.

        Parameters
        ----------
        time: slice | int | str | None, optional
            The time span over which to perform the reconstruction.
            - If None: Reconstructs the entire training dataset.
            - If int: Reconstructs a single snapshot by integer index.
            - If str: Reconstructs a single snapshot by coordinate label.
            - If slice: Reconstructs a range. Slices containing integers are
              treated as indices; slices containing strings are treated as labels.
        time_dimension: str, default 'time'
            The name of the time dimension in the right singular vectors (_v).
        memory_limit_bytes: float, default 1e9 (1 GB)
            The memory threshold. If the estimated reconstruction size exceeds
            this, Dask is used for lazy computation. Otherwise, NumPy is used.

        Returns
        -------
        xr.DataArray
            The reconstructed data.
        """
        self._check_is_fitted(["_u", "_s", "_v"])

        if time_dimension not in self._v.dims:
            msg = f"Dimension '{time_dimension}' not found in right singular vectors."
            logger.exception(msg)
            raise ValueError(msg)

        try:
            if time is None:
                v_subset = self._v
            elif isinstance(time, int):
                v_subset = self._v.isel({time_dimension: time})
            elif isinstance(time, str):
                v_subset = self._v.sel({time_dimension: time})
            elif isinstance(time, slice):
                is_index_slice = True
                if (isinstance(time.start, str)) or (isinstance(time.stop, str)):
                    is_index_slice = False
                
                if is_index_slice:
                    v_subset = self._v.isel({time_dimension: time})
                else:
                    v_subset = self._v.sel({time_dimension: time})
            else:
                msg = "Parameter 'time' must be a slice, int, str, or None."
                logger.exception(msg)
                raise ValueError(msg)
        except KeyError as e:
            msg = f"Could not slice the time dimension '{time_dimension}' with key {time}."
            logger.exception(msg)
            raise KeyError(msg) from e

        # Estimate Reconstruction Size
        # Shape approximation: U (spatial) * V_subset (temporal) * 8 bytes (float64)
        # U shape is (n_features, n_components)
        n_features = self._u.shape[0]
        # Handle case where v_subset is 1D (single snapshot) vs 2D (multiple)
        if time_dimension in v_subset.dims:
            n_time = v_subset.sizes[time_dimension]
        else:
            n_time = 1
            
        estimated_bytes = n_features * n_time * 8
        
        msg = f"Estimated reconstruction size is {estimated_bytes/1e6:.2f} MB."
        logger.info(msg)

        use_dask = estimated_bytes > memory_limit_bytes
        
        # We pre-multiply U * S for efficiency: X = (U*S) @ V.T
        lhs = self._u * self._s
        rhs = v_subset

        if use_dask:
            logger.info("Memory limit exceeded. Using Dask for reconstruction.")
            if not isinstance(lhs.data, da.Array):
                lhs = lhs.chunk("auto")
            if not isinstance(rhs.data, da.Array):
                rhs = rhs.chunk("auto")
        else:
            logger.info("Within memory limit. Using NumPy for reconstruction.")
            # if arrays were lazy, bring them into memory to avoid dask overhead for small ops
            if isinstance(lhs.data, da.Array):
                lhs = lhs.compute()
            if isinstance(rhs.data, da.Array):
                rhs = rhs.compute()

        logger.info("Computing reconstruction...")
        reconstruction = lhs.dot(rhs, dim="components")
      
        logger.info("Done.")
        
        return reconstruction