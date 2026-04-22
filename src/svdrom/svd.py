import dask
import dask.array as da
import numpy as np
import xarray as xr

import svdrom.config as config
from svdrom.logger import setup_logger
from svdrom.svdrom_base import DecompositionModel

logger = setup_logger("SVD", "svd.log")


class TruncatedSVD(DecompositionModel):
    def __init__(
        self,
        n_components: int,
        algorithm: str = "tsqr",
        compute_u: bool = True,
        compute_v: bool = True,
        compute_var_ratio: bool = False,
        rechunk: bool = False,
    ):
        """Linear dimensionality reduction using truncated
        Singular Value Decomposition (SVD).

        This class follows heavily the design of
        `dask_ml.decomposition.TruncatedSVD`, but it has been
        modified to work with Dask-backed Xarray DataArrays.

        Parameters
        ----------
        n_components: int
            Number of SVD components to keep. Must be less than
            the number of features of the input array.
        algorithm: str, {'tsqr', 'randomized'}, (default 'tsqr')
            SVD algorithm to use. 'tsqr' only supports chunking
            along one dimension (i.e. the array should be tall-and-skinny
            or short-and-fat), while 'randomized' supports chunking
            in both dimensions. 'tsqr' is more precise, but it is slower
            and struggles to operate on very large arrays. 'randomized' is
            less precise but is much faster and can more easily operate on
            very large arrays. See the Notes section for more.
        compute_u: bool, (default True)
            Whether to eagerly compute the left singular vectors, `u`. If
            set to False, the computational graph for `u` is built but not
            computed and `u` is returned as a lazy Dask collection.
        compute_v: bool, (default True)
            Whether to eagerly compute the right singular vectors, `v`. If
            set to False, the computational graph for `v` is built but not
            computed and `v` is returned as a lazy Dask collection.
        compute_var_ratio: bool, (default False)
            Whether to eagerly compute the ratio of explained variance by
            each SVD component. If False, the computational graph is built
            but not computed and is returned as a lazy Dask collection.
        rechunk: bool, (default False)
            If using the 'randomized' algorithm, whether to rechunk the input
            array to ensure a single chunk along the smallest dimension.
            Rechunking will always be performed when using the 'tsqr' algorithm
            if the input array requires it, regardless of this parameter.

        Attributes
        ----------
        u: xr.DataArray, shape (n_samples, n_components)
            The left singular vectors. If `compute_u=True`,
            the xarray object is numpy-backed, otherwise it's
            dask-backed.
        s: np.ndarray, shape (n_components,)
            The singular vectors.
        v: xr.DataArray, shape (n_components, n_features)
            The right singular vectors. If `compute_v=True`,
            the xarray object is numpy-backed, otherwise it's
            dask-backed.
        explained_var_ratio: np.ndarray | da.Array, shape (n_components,)
            The ratio of explained variance by each SVD component.
            If `compute_var_ratio=True` returned as a numpy array,
            otherwise returned as a dask array.
        n_components: int
            The number of SVD components requested.
        algorithm: str
            The SVD algorithm requested.

        Notes
        -----
        The 'tsqr' algorithm is implemented via Dask's `dask.array.linalg.svd`
        function. The 'randomized' algorithm is implemented via Dask's
        `dask.array.linalg.svd_compressed` function.
        """
        super().__init__(n_components=n_components)  # Inherit from baseclass

        self._algorithm = algorithm
        self._compute_u = compute_u
        self._compute_v = compute_v
        self._compute_var_ratio = compute_var_ratio
        self._rechunk = rechunk
        self._u: xr.DataArray | None = None
        self._s: np.ndarray | None = None
        self._v: xr.DataArray | None = None
        self._explained_var_ratio: np.ndarray | da.Array | None = None

    @property
    def n_components(self) -> int:
        """Number of SVD components (read-only)."""
        return self._n_components

    @property
    def s(self):
        """Singular values (read-only)."""
        return self._s

    @property
    def u(self):
        """Left singular vectors (read-only)."""
        return self._u

    @property
    def v(self):
        """Right singular vectors (read-only)."""
        return self._v

    @property
    def explained_var_ratio(self):
        """Ratio of explained variance (read-only)."""
        return self._explained_var_ratio

    @property
    def algorithm(self):
        """SVD algorithm to use (read-only)."""
        return self._algorithm

    def _rechunk_array(self, X: da.Array) -> da.Array:
        """Rechunks the input array to ensure optimal chunk sizes
        for SVD computation based on array shape. Dask's `svd()` algorithm
        can only handle chunking in one direction, whereas the `svd_compressed()`
        algorithm can handle chunking in both directions.
        """
        nb = X.numblocks
        msg = (
            "Will rechunk the array before fitting the SVD. "
            "This will add some overhead."
        )
        nr, nc = X.shape
        if nr > nc and nb[1] > 1:
            msg = "The input array is considered tall-and-skinny. " + msg
            logger.info(msg)
            return X.rechunk({0: "auto", 1: -1})  # -1 means "all in one chunk"
        if nr < nc and nb[0] > 1:
            msg = "The input array is considered short-and-fat. " + msg
            logger.info(msg)
            return X.rechunk({0: -1, 1: "auto"})
        if nr == nc and not (nb[0] == 1 or nb[1] == 1):
            msg = "The input array is square. " + msg
            logger.info(msg)
            return X.rechunk({0: "auto", 1: -1})
        return X

    def _check_array(self, X: xr.DataArray):
        """Checks if the input array is valid for SVD computation."""
        if X.ndim != 2:
            msg = (
                "The input array must be 2-dimensional. "
                f"Got a {X.ndim}-dimensional array."
            )
            logger.error(msg)
            raise ValueError(msg)
        if self._n_components >= X.shape[1]:
            msg = (
                "n_components must be less than n_features. "
                f"Got n_components: {self.n_components}, n_features: {X.shape[1]}."
            )
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(X.data, da.Array):
            msg = (
                f"The {self.__class__.__name__} class only supports Dask-backed "
                f"Xarray DataArrays. Got {type(X.data)} instead."
            )
            logger.error(msg)
            raise TypeError(msg)

    def _singular_vectors_to_dataarray(
        self, singular_vectors: np.ndarray, X: xr.DataArray
    ) -> xr.DataArray:
        """Transform the singular vectors into a Xarray DataArray following
        the dimensions and coordinates of the DataArray on which SVD was
        performed. The function automatically identifies whether the input
        singular vectors are left or right singular vectors.
        """
        if singular_vectors.shape[0] == X.shape[0]:
            # this corresponds to `u`: replace second dimension (e.g. 'time')
            old_dims = list(X.dims)
            new_dims = [old_dims[0], "components"]
            coords = {k: v for k, v in X.coords.items() if k != old_dims[1]}
            coords["components"] = np.arange(singular_vectors.shape[1])
            name = "svd_u"
            attrs = None
        elif singular_vectors.shape[1] == X.shape[1]:
            # this corresponds to `v`: replace first dimension (e.g. 'samples')
            old_dims = list(X.dims)
            new_dims = ["components", old_dims[1]]
            coords = {old_dims[1]: X.coords[old_dims[1]]}
            coords["components"] = np.arange(singular_vectors.shape[0])
            name = "svd_v"
            attrs = (
                {
                    config.get("hankel_time_mapping_attr"): X.attrs[
                        config.get("hankel_time_mapping_attr")
                    ]
                }
                if config.get("hankel_time_mapping_attr") in X.attrs
                else None
            )
        else:
            msg = (
                "Cannot transform singular vectors into Xarray DataArray. "
                "Shape of singular_vectors does not match X."
            )
            logger.error(msg)
            raise ValueError(msg)
        return xr.DataArray(
            singular_vectors,
            dims=new_dims,
            coords=coords,
            name=name,
            attrs=attrs,
        )

    def fit(
        self,
        X: xr.DataArray,
        **kwargs,
    ) -> "TruncatedSVD":
        """Fit the SVD model to the input array.

        Parameters
        ----------
        X: (xarray.DataArray), shape (n_samples, n_features)
            The input array to fit the SVD model on.
        **kwargs:
            Additional keyword arguments to pass to the
            randomized SVD algorithm. See
            `dask.array.linalg.svd_compressed` for more details.
        """
        if self._algorithm not in ["tsqr", "randomized"]:
            msg = (
                f"Unsupported algorithm: {self._algorithm}. "
                "Supported algorithms are 'tsqr' and 'randomized'."
            )
            logger.error(msg)
            raise ValueError(msg)

        self._check_array(X)
        X_da = X.data

        if self._algorithm == "tsqr":
            msg = "Will use TSQR algorithm."
            logger.info(msg)
            X_da = self._rechunk_array(X_da)
            u, s, v = da.linalg.svd(X_da)  # employs tsqr internally
            u = u[:, : self._n_components]
            s = s[: self._n_components]
            v = v[: self._n_components]
        else:
            msg = "Will use randomized algorithm."
            logger.info(msg)
            X_da = self._rechunk_array(X_da) if self._rechunk else X_da
            u, s, v = da.linalg.svd_compressed(X_da, self._n_components, **kwargs)

        X_da_transformed = u * s
        explained_var = X_da_transformed.var(axis=0)
        full_var = X_da.var(axis=0).sum()
        explained_var_ratio = explained_var / full_var

        results = []
        results.append(s)

        if self._compute_u:
            results.append(u)
        if self._compute_v:
            results.append(v)
        if self._compute_var_ratio:
            results.append(explained_var_ratio)

        # compute all Dask collections at once
        msg = "Computing SVD results..."
        logger.info(msg)
        computed = dask.compute(*results)
        msg = "Done."
        logger.info(msg)

        s = computed[0]
        idx = 1
        if self._compute_u:
            u = computed[idx]
            idx += 1
        if self._compute_v:
            v = computed[idx]
            idx += 1
        if self._compute_var_ratio:
            explained_var_ratio = computed[idx]
        self._u = self._singular_vectors_to_dataarray(u, X)
        self._s = s
        self._v = self._singular_vectors_to_dataarray(v, X)
        self._explained_var_ratio = explained_var_ratio

        return self

    def compute_u(self) -> None:
        """Compute left singular vectors if they are
        still a lazy Dask collection.
        """
        self._check_is_fitted(["_u"])
        assert self._u is not None  # needed for mypy check
        msg = "Computing left singular vectors..."
        logger.info(msg)
        self._u = self._u.compute()
        msg = "Done."
        logger.info(msg)

    def compute_v(self) -> None:
        """Compute right singular vectors if they are
        still a lazy Dask collection.
        """
        self._check_is_fitted(["_v"])
        assert self._v is not None  # needed for mypy check
        msg = "Computing right singular vectors..."
        logger.info(msg)
        self._v = self._v.compute()
        msg = "Done."
        logger.info(msg)

    def compute_var_ratio(self) -> None:
        """Compute the ratio of explained variance if it is
        still a lazy Dask collection.
        """
        self._check_is_fitted(["_explained_var_ratio"])
        msg = "Computing explained variance ratio..."
        logger.info(msg)
        if isinstance(self._explained_var_ratio, da.Array):
            self._explained_var_ratio = self._explained_var_ratio.compute()
        msg = "Done."
        logger.info(msg)

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        """Transform the input array using the computed right singular vectors.

        Parameters
        ----------
        X: (xarray.DataArray), shape (n_samples, n_features)
            the array to be transformed, which must have the same
            number of features as the original array on which SVD was fitted.

        Returns
        -------
        xarray.DataArray: the transformed array.
        """

        if not isinstance(self._v, xr.DataArray):
            msg = (
                "Computed right singular vectors are "
                "required in order to call transform()."
            )
            logger.error(msg)
            raise ValueError(msg)
        X_da = X.data
        try:
            msg = "Computing transformation..."
            logger.info(msg)
            X_da_transformed = X_da.dot(self._v.data.T)
            X_da_transformed = X_da_transformed.compute()
            msg = "Done."
            logger.info(msg)
        except ValueError as e:
            msg = (
                "Cannot transform the input array with the computed right "
                "singular vectors. Ensure that the input array has the same "
                "number of features as the original array on which SVD was fitted."
            )
            logger.exception(msg)
            raise ValueError(msg) from e
        return self._singular_vectors_to_dataarray(X_da_transformed, X)

    @staticmethod
    def _is_index_slice(s: slice) -> bool:
        """Classify a slice as index-based (integer bounds) or label-based
        (string bounds). ``None`` bounds are compatible with either.
        """
        start, stop = s.start, s.stop
        start_int = start is None or isinstance(start, int)
        stop_int = stop is None or isinstance(stop, int)
        if start_int and stop_int:
            return True
        start_str = start is None or isinstance(start, str)
        stop_str = stop is None or isinstance(stop, str)
        if start_str and stop_str:
            return False
        msg = "Slice bounds must be all int (or None) or all str (or None)."
        logger.error(msg)
        raise TypeError(msg)

    def _select_snapshot(
        self,
        array: xr.DataArray,
        snapshot: slice | int | str | None,
        snapshot_dim: str,
    ) -> xr.DataArray:
        """Subset ``array`` along ``snapshot_dim`` based on the type of
        ``snapshot``. ``None`` returns ``array`` unchanged; integers and
        integer-bounded slices use positional indexing (``isel``); strings
        and string-bounded slices use label indexing (``sel``).
        """
        if snapshot is None:
            return array
        try:
            if isinstance(snapshot, slice):
                if self._is_index_slice(snapshot):
                    return array.isel({snapshot_dim: snapshot})
                return array.sel({snapshot_dim: snapshot})
            if isinstance(snapshot, int):
                return array.isel({snapshot_dim: snapshot})
            if isinstance(snapshot, str):
                return array.sel({snapshot_dim: snapshot})
        except IndexError as e:
            msg = (
                f"Snapshot index {snapshot} is out of bounds along "
                f"dimension '{snapshot_dim}'."
            )
            logger.exception(msg)
            raise IndexError(msg) from e
        except KeyError as e:
            msg = (
                f"Snapshot label {snapshot!r} not found along "
                f"dimension '{snapshot_dim}'."
            )
            logger.exception(msg)
            raise KeyError(msg) from e
        msg = "'snapshot' must be a slice, int, str, or None."
        logger.error(msg)
        raise TypeError(msg)

    def reconstruct(
        self,
        snapshot: slice | int | str | None = None,
        snapshot_dim: str = "time",
    ) -> xr.DataArray:
        """Reconstruct one, many, or all snapshots from the truncated
        SVD decomposition.

        Computes the rank-`k` approximation `U @ diag(S) @ V` restricted
        to the selection along `snapshot_dim`.

        Parameters
        ----------
        snapshot: slice | int | str | None, (default None)
            - ``None``: reconstructs the entire dataset on which the SVD
              was fitted.
            - ``int``: reconstructs a single snapshot by positional index.
            - ``str``: reconstructs all snapshots whose coordinate label
              matches.
            - ``slice``: reconstructs a range along ``snapshot_dim``. Slices
              with integer (or ``None``) bounds are treated as positional
              index slices; slices with string bounds are treated as
              coordinate-label slices.
        snapshot_dim: str, (default 'time')
            The dimension along which snapshots are indexed. Must be a
            dimension of either ``u`` or ``v``.

        Returns
        -------
        xr.DataArray: The reconstructed data.

        Examples
        --------
        # Reconstruct the first snapshot by positional index
        >>> tsvd.reconstruct(0)

        # Reconstruct all snapshots with label '2017-01-01'
        >>> tsvd.reconstruct("2017-01-01")

        # Reconstruct a range of snapshots by label
        >>> tsvd.reconstruct(slice("2017-01-01", "2017-01-31"))

        # Reconstruct the entire training dataset
        >>> tsvd.reconstruct()
        """
        self._check_is_fitted(["_u", "_v", "_s"])
        assert self._u is not None  # needed for mypy check
        assert self._v is not None
        assert self._s is not None

        in_v = snapshot_dim in self._v.dims
        in_u = snapshot_dim in self._u.dims
        if not in_v and not in_u:
            msg = f"Snapshot dimension '{snapshot_dim}' does not exist."
            logger.error(msg)
            raise ValueError(msg)

        # Wrap singular values as a DataArray so broadcasting aligns by
        # the named 'components' dim (raw numpy would align by position).
        s_da = xr.DataArray(self._s, dims=("components",))

        selector = self._v if in_v else self._u
        subset = self._select_snapshot(selector, snapshot, snapshot_dim)

        if in_v:
            return self._u @ (s_da * subset)
        return (subset * s_da) @ self._v
