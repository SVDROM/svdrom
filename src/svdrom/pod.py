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
        remove_mean: bool = False,
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

        self._energy: np.ndarray | None = None
        self._remove_mean: bool = remove_mean
        self._time_dim: str = time_dimension

    @property
    def modes(self) -> xr.DataArray | None:
        """POD (spatial) modes (read-only)."""
        return self._u

    @property
    def time_coeffs(self) -> xr.DataArray | None:
        """Time coefficients (read-only)."""
        return self._v

    @property
    def energy(self) -> np.ndarray | None:
        """Energy (variance) explained by each POD mode (read-only)."""
        if self._s is not None and self._u is not None:
            return self._s**2 / self._u.shape[0]
        return None

    @property
    def explained_energy_ratio(self) -> np.ndarray | da.Array | None:
        return self._explained_var_ratio

    def _preprocess_array(self, X: xr.DataArray) -> xr.DataArray:
        """Transpose the array if the user-specified time dimension
        is not along the columns. Remove the temporal average if
        requested by the user.
        """
        dims = X.dims
        if dims.index(self._time_dim) != 1:
            X = X.T
        if self._remove_mean:
            scaler = StandardScaler()
            X = scaler(
                X,
                dim=self._time_dim,
                with_std=False,
            )
            assert isinstance(X, xr.DataArray), "Expected DataArray after scaling."
        return X

    def fit(
        self,
        X: xr.DataArray,
        **kwargs,
    ) -> None:
        if self._time_dim not in X.dims:
            msg = (
                f"Specified time dimension '{self._time_dim}' "
                "is not a dimension of the input array."
            )
            raise ValueError(msg)
        X = self._preprocess_array(X)
        super().fit(X, **kwargs)

    def compute_modes(self) -> None:
        super().compute_u()

    def compute_time_coeffs(self) -> None:
        super().compute_v()

    def compute_energy_ratio(self) -> None:
        super().compute_var_ratio()
