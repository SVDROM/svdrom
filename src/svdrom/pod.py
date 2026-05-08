import dask.array as da
import numpy as np
import xarray as xr

from svdrom.logger import setup_logger
from svdrom.preprocessing import StandardScaler
from svdrom.svd import TruncatedSVD

logger = setup_logger("POD", "pod.log")


class POD(TruncatedSVD):
    def __init__(
        self,
        n_modes: int,
        svd_algorithm: str = "tsqr",
        compute_modes: bool = True,
        compute_time_coeffs: bool = True,
        compute_energy_ratio: bool = False,
        rechunk: bool = False,
        remove_mean: bool = True,
        time_dimension: str = "time",
    ):
        super().__init__(
            n_components=n_modes,
            algorithm=svd_algorithm,
            compute_u=compute_modes,
            compute_v=compute_time_coeffs,
            compute_var_ratio=compute_energy_ratio,
            rechunk=rechunk,
        )

        self._time_coeffs: xr.DataArray | None = None
        self._remove_mean: bool = remove_mean
        self._time_dim: str = time_dimension

    @property
    def modes(self) -> xr.DataArray | None:
        """POD (spatial) modes (read-only)."""
        return self._u

    @property
    def time_coeffs(self) -> xr.DataArray | None:
        """Time coefficients (read-only)."""
        return self._time_coeffs

    @property
    def energy(self) -> np.ndarray | None:
        """Energy (variance) explained by each POD mode (read-only)."""
        if self._s is not None:
            return self._s**2
        return None

    @property
    def explained_energy_ratio(self) -> np.ndarray | da.Array | None:
        """Ratio of total energy (read-only)."""
        return self._explained_var_ratio

    def _preprocess_array(self, X: xr.DataArray) -> xr.DataArray:
        """Transpose the array if the user-specified time dimension
        is not along the columns. Remove the temporal average if
        requested by the user.
        """
        if X.dims.index(self._time_dim) != 1:
            X = X.T
        if self._remove_mean:
            scaler = StandardScaler()
            X = scaler(X, dim=self._time_dim)
            assert isinstance(X, xr.DataArray), "Expected DataArray after scaling."
        return X

    def fit(
        self,
        X: xr.DataArray,
        **kwargs,
    ) -> "POD":
        """Fit the POD model to the input array.

        Parameters
        ----------
        X: xr.DataArray, shape (n_spatial_points, n_snapshots)
            The input array to fit the POD model on. The array must
            be Dask-backed.
        **kwargs:
            Additional keyword arguments to pass to the randomized
            SVD algorithm used as a backend for the POD computation.
            See dask.array.linalg.svd_compressed for more details.
        """
        if self._time_dim not in X.dims:
            msg = (
                f"Specified time dimension '{self._time_dim}' "
                "is not a dimension of the input array."
            )
            raise ValueError(msg)
        X = self._preprocess_array(X)
        super().fit(X, **kwargs)
        assert self._s is not None
        assert self._v is not None
        self._time_coeffs = xr.apply_ufunc(
            np.matmul, np.diag(self._s), self._v, dask="allowed"
        )
        return self

    def compute_modes(self) -> None:
        """Compute the POD spatial modes if they are
        still a lazy Dask collection.
        """
        msg = "Computing POD spatial modes..."
        logger.info(msg)
        super().compute_u()
        msg = "Done."
        logger.info(msg)

    def compute_energy_ratio(self) -> None:
        """Compute the ratio of captured total energy if it is
        still a lazy Dask collection.
        """
        msg = "Computing the energy ratio..."
        logger.info(msg)
        super().compute_var_ratio()
        msg = "Done."
        logger.info(msg)

    def compute_time_coeffs(self) -> None:
        """Compute the POD time coefficients if they are still a
        lazy Dask collection.
        """
        self._check_is_fitted(["_time_coeffs"])
        assert self._time_coeffs is not None
        msg = "Computing the POD time coefficients..."
        logger.info(msg)
        self._time_coeffs = self._time_coeffs.compute()
        msg = "Done."
        logger.info(msg)
