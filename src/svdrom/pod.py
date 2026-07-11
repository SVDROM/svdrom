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
        self._mean: xr.DataArray | None = None
        self._scale_factor: float | None = None

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
        requested by the user. Also scale by the square root of the number
        of snapshots, so that modal energy is independent of the number
        of snapshots.
        """
        if X.dims.index(self._time_dim) != 1:
            X = X.T
        if self._remove_mean:
            scaler = StandardScaler()
            X = scaler(X, dim=self._time_dim)
            assert isinstance(X, xr.DataArray), "Expected DataArray after scaling."
            self._mean = scaler._mean
        n_snapshots = X.sizes[self._time_dim]
        self._scale_factor = n_snapshots**0.5
        return X / self._scale_factor

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

        n_snapshots = X.sizes[self._time_dim]
        space_dim = next(d for d in X.dims if d != self._time_dim)
        n_spatial_points = X.sizes[space_dim]
        max_components = min(n_spatial_points, n_snapshots)
        if self._n_components >= max_components:
            msg = (
                "n_components must be less than min(n_spatial_points, "
                f"n_snapshots). Got n_components: {self.n_components}, "
                f"n_spatial_points: {n_spatial_points}, n_snapshots: {n_snapshots}."
            )
            logger.error(msg)
            raise ValueError(msg)

        X = self._preprocess_array(X)
        super().fit(X, **kwargs)
        assert isinstance(self._s, np.ndarray)
        assert isinstance(self._v, xr.DataArray)
        self._time_coeffs = xr.DataArray(
            self._s.reshape(-1, 1) * self._v.data,
            dims=self._v.dims,
            coords=self._v.coords,
            attrs=self._v.attrs,
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

    def extended_pod(
        self,
        C: xr.DataArray,
        remove_mean: bool = True,
        compute: bool = True,
    ) -> xr.DataArray:
        """Compute the extended POD modes as proposed by Boree (2003).

        Extended POD finds the correlation between the pre-computed POD
        modes (e.g. from velocity fields) and another quantity measured
        simultaneously in time, such as temperature or pressure.

        The extended POD modes are given by:

            chi_j = (1 / (lambda_j * N)) * sum_i(a_ij * c_i')

        where N is the number of snapshots, lambda_j is the energy of
        the j-th POD mode, a_ij = phi_j . X'(t_i) are the (unscaled)
        time coefficients of the j-th POD mode (i.e. the projection of
        the fluctuating snapshot of the primary field X onto mode phi_j),
        and c_i' is the fluctuating part of the simultaneously measured
        quantity C.

        Unlike the POD spatial modes, the extended POD modes are not
        unit-norm. Their norm quantifies the spatial energy in C' that
        is linearly correlated with mode j, and can be used to define a
        scalar correlation coefficient (Boree, 2003).

        Note: the stored `time_coeffs` attribute equals a_ij / sqrt(N),
        so the formula as implemented becomes:

            chi_j = (1 / (lambda_j * sqrt(N))) * sum_i(a_ij_stored * c_i')

        Parameters
        ----------
        C : xr.DataArray
            The simultaneously measured spatio-temporal field. Must have
            the same time dimension (with matching size) as the array on
            which POD was fitted, but may have a different spatial
            dimension (different number of spatial points). Must be
            Dask-backed.
        remove_mean : bool, default True
            Whether to remove the temporal mean from C before computing
            the extended POD modes. Set to False if C is already a
            fluctuating quantity.
        compute : bool, default True
            If True, eagerly compute and return a NumPy-backed DataArray.
            If False, return a lazy Dask-backed DataArray.

        Returns
        -------
        xr.DataArray
            The extended POD modes with shape (n_space_C, n_modes),
            returned in the same format as the POD spatial modes.
        """
        self._check_is_fitted(["_time_coeffs", "_s", "_scale_factor"])
        assert self._time_coeffs is not None
        assert self._s is not None
        assert self._scale_factor is not None

        if self._time_dim not in C.dims:
            msg = (
                f"Specified time dimension '{self._time_dim}' "
                "is not a dimension of the input array."
            )
            raise ValueError(msg)

        if C.dims.index(self._time_dim) != 1:
            C = C.T

        n_snapshots = C.sizes[self._time_dim]
        expected_n_snapshots = self._time_coeffs.sizes[self._time_dim]
        if n_snapshots != expected_n_snapshots:
            msg = (
                f"Number of snapshots in input array ({n_snapshots}) does not "
                f"match the number used during fit ({expected_n_snapshots})."
            )
            raise ValueError(msg)

        # Check that the time coordinates match (simultaneous measurement)
        if self._time_dim in C.coords and self._time_dim in self._time_coeffs.coords:
            c_time_coords = C.coords[self._time_dim].values
            fit_time_coords = self._time_coeffs.coords[self._time_dim].values
            if not np.array_equal(c_time_coords, fit_time_coords):
                msg = (
                    "Time coordinates of the input array do not match "
                    "those of the training data. Extended POD requires "
                    "the fields to be measured simultaneously."
                )
                raise ValueError(msg)

        C_prime = C - C.mean(dim=self._time_dim) if remove_mean else C

        energy = self._s**2
        # Extended POD: chi = (1 / scale_factor) * C' @ A^T @ Lambda^{-1}
        chi = C_prime @ self._time_coeffs.T / (self._scale_factor * energy)

        space_dim = next(d for d in C.dims if d != self._time_dim)
        chi = xr.DataArray(
            chi.data,
            dims=[space_dim, "components"],
            coords={
                space_dim: C.coords[space_dim],
                "components": np.arange(self._n_components),
            },
        )

        if compute:
            chi = chi.compute()

        return chi

    def transform(self, X: xr.DataArray, compute: bool = True) -> xr.DataArray:
        """Transform the input array by projecting onto the computed POD modes.

        The same preprocessing applied during fit() (mean removal and
        scaling) is automatically applied to the input array before
        projection. The input array is automatically transposed if
        needed to ensure the time dimension is along the columns.

        Parameters
        ----------
        X: xr.DataArray, shape (n_space, n_snapshots) or (n_snapshots, n_space)
            The array to be transformed. Must have the same spatial
            dimension (number of spatial points) as the original array
            on which POD was fitted. Do not remove the mean from the
            array before calling this method, as the mean of the training
            data will be subtracted automatically.
        compute: bool, default True
            If True, eagerly compute and return a NumPy-backed DataArray.
            If False, return a lazy Dask-backed DataArray without triggering
            computation.

        Returns
        -------
        xr.DataArray: The temporal coefficients resulting from projecting
        the input array onto the pre-computed POD modes. Shape is
        (n_modes, n_snapshots).
        """
        self._check_is_fitted(["_u"])
        assert self._u is not None  # needed for mypy checks

        # Apply same preprocessing as during fit()
        if X.dims.index(self._time_dim) != 1:
            X = X.T
        if self._remove_mean:
            self._check_is_fitted(["_mean"])
            assert self._mean is not None  # needed for mypy checks
            X = X - self._mean
        assert self._scale_factor is not None
        X = X / self._scale_factor

        # Project onto POD modes (proper POD projection)
        if compute:
            result = (self._u.T @ X).compute()
        else:
            # Ensure lazy computation by using Dask-backed u
            if isinstance(self._u.data, np.ndarray):
                u_lazy = xr.DataArray(
                    da.from_array(self._u.data),
                    dims=self._u.dims,
                    coords=self._u.coords,
                )
                result = u_lazy.T @ X
            else:
                result = self._u.T @ X

        return result

    def reconstruct(
        self, snapshot: int | str, snapshot_dim: str | None = None
    ) -> xr.DataArray:
        if snapshot_dim is None:
            snapshot_dim = self._time_dim
        result = super().reconstruct(snapshot, snapshot_dim)
        # undo pre-processing
        assert self._scale_factor is not None
        result = result * self._scale_factor
        if self._remove_mean:
            assert self._mean is not None
            result = result + self._mean
        return result
