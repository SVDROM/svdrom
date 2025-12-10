from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.array.lib.stride_tricks import sliding_window_view as sliding_window_view_dask
from numpy.lib.stride_tricks import sliding_window_view as sliding_window_view_np

import svdrom.config as config
from svdrom.logger import setup_logger

logger = setup_logger("Preprocessing", "preprocessing.log")


def variable_spatial_stack(
    X: xr.Dataset | xr.DataArray, dims: Sequence[str]
) -> xr.DataArray:
    """Stack multiple dimensions of an input Xarray Dataset or
    DataArray into a single new 'samples' dimension. You can
    change the default name of the 'samples' dimension using
    svdrom.config.set(stack_coord_name="desired name").

    Parameters
    ----------
    X (xr.Dataset | xr.DataArray): the input data to stack.
    dims (Iterable[str]): the dimensions to stack together into a
    single spatial dimension (e.g. ('x', 'y') or ('lat', 'lon')).

    Returns
    -------
    xr.DataArray: The array with the specified dimensions stacked
    into a single spatial dimension.

    Notes
    -----
    If the input Xarray object is a Dataset containing multiple variables,
    all variables will be stacked together along the new 'samples' dimension.
    The resulting DataArray will have a 'samples' dimension that combines
    the specified spatial dimensions (and the variable dimension if the input
    was a Dataset).
    The returned xarray object is Dask-backed and lazy if the input
    is Dask-backed.
    """

    dims = list(dims)
    if isinstance(X, xr.Dataset):
        msg = "Performing variable stacking."
        logger.info(msg)
        X = X.to_dataarray(dim="variable")
        dims.insert(0, "variable")

    all_dims: list[str] = list(map(str, X.sizes.keys()))
    if not all(dim in all_dims for dim in dims):
        msg = (
            f"Some dimensions {dims} are not present in the input array "
            f"with dimensions {all_dims}."
        )
        logger.exception(msg)
        raise ValueError(msg)

    msg = "Performing spatial stacking."
    logger.info(msg)
    return X.stack({config.get("stack_coord_name"): dims})


class StandardScaler:
    """Preprocessing class for scaling Xarray datasets or data arrays
    by removing the mean and optionally scaling by the standard deviation
    along a specified dimension.

    Attributes
    ----------
    mean (xr.DataArray | xr.Dataset): the mean values computed along the
    specified dimension.
    std (xr.DataArray | xr.Dataset): the standard deviation values computed
    along the specified dimension.
    with_std (bool): whether the data has been scaled to unit variance.
    """

    def __init__(self):
        self._mean = None
        self._std = None
        self._with_std = False

    @property
    def mean(self) -> xr.DataArray | xr.Dataset | None:
        """Mean (read-only)."""
        return self._mean

    @property
    def std(self) -> xr.DataArray | xr.Dataset | None:
        """Standard deviation (read-only)."""
        return self._std

    @property
    def with_std(self) -> bool:
        """Whether data scaled to unit variance or not (read_only)."""
        return self._with_std

    def __call__(
        self,
        X: xr.Dataset | xr.DataArray,
        dim: str = "time",
        with_std: bool = False,
    ):
        """Scales the input xarray Dataset or DataArray by removing
        the mean and optionally dividing by the standard deviation.

        Parameters:
        -----------
        X (xr.Dataset | xr.DataArray): the input xarray object to be scaled.
        dim (str): the dimension along which to compute the mean and
        standard deviation. Default is "time".
        with_std (bool): if True, scales the data by dividing by the standard
        deviation after subtracting the mean. Default is False.

        Returns:
        --------
        xr.Dataset | xr.DataArray: the scaled xarray object with the mean removed,
        and optionally divided by the standard deviation.

        Note
        ----
        The mean and standard deviation are computed eagerly and stored as NumPy-backed
        xarray objects. The returned xarray object is Dask-backed and lazy if the input
        is Dask-backed.
        """
        self._with_std = with_std
        self._mean = X.mean(dim=dim)
        msg = f"Computing mean along dimension {dim}..."
        logger.info(msg)
        self._mean = self._mean.compute()
        msg = "Finished computing mean."
        logger.info(msg)
        if self._with_std:
            self._std = X.std(dim=dim)
            msg = f"Computing std along dimension {dim}..."
            logger.info(msg)
            self._std = self._std.compute()
            msg = "Finished computing std."
            logger.info(msg)
            return (X - self._mean) / self._std
        return X - self._mean


def hankel_preprocessing(X: xr.DataArray, d: int = 2) -> xr.DataArray:
    """Hankel pre-processing.

    Given a matrix with dimensions (m x n), where 'm' is the number of
    samples (e.g. spatial observations) and 'n' is the number of snapshots
    or temporal observations, perform time-delay embedding by appending
    the snapshots with time-shifted versions of themselves. This can help
    unveil hidden or latent variables from the data matrix.

    Parameters
    ----------
    X: xr.DataArray
        The input array, with dimensions (m x n), where 'm' is the number
        of samples and 'n' is the number of snapshots. The DataArray can
        be NumPy or Dask-backed.
    d: int
        Hankel matrix rank. Must be an integer equal to or greater than 2.

    Returns
    -------
    xr.DataArray
        The augmented data matrix, with dimensions ((m*d) x (n-d+1)). For
        example, if d=2, the input array is augmented with one time-delay,
        resulting in a matrix with dimensions ((2*m) x (n-1)). If the input
        DataArray is NumPy-backed, the returned DataArray is also NumPy-backed.
        If it is Dask-backed, then the returned DataArray is also Dask-backed.
        The returned DataArray has a new coordinate, but default names 'lag',
        which indicates the time delay relative to the current time stamp. You
        can change the default name of this coordinate using
        svdrom.config.set(hankel_coord_name="desired name").
    """
    if X.ndim != 2:
        msg = "The input array must be two dimensional."
        logger.exception(msg)
        raise ValueError(msg)
    if d < 2:
        msg = "'d' must be an integer equal to or greater than 2."
        logger.exception(msg)
        raise ValueError(msg)
    n_rows = X.shape[0]
    if isinstance(X.data, da.Array):
        X_delayed = (
            sliding_window_view_dask(X.data, d, axis=1)
            .transpose(2, 0, 1)
            .reshape(n_rows * d, -1)
        )
    elif isinstance(X.data, np.ndarray):
        X_delayed = (
            sliding_window_view_np(X.data, d, axis=1)
            .transpose(2, 0, 1)
            .reshape(n_rows * d, -1)
        )
    else:
        msg = "The DataArray must be backed by a NumPy or Dask Array."
        logger.exception(msg)
        raise ValueError(msg)

    dims = X.dims
    samples: Any = np.tile(X[dims[0]].to_numpy(), d)
    if isinstance(X.indexes[dims[0]], pd.MultiIndex):
        samples = pd.MultiIndex.from_tuples(
            samples,
            names=X.indexes[dims[0]].names,
        )
    return xr.DataArray(
        X_delayed,
        dims=dims,
        coords={
            dims[0]: samples,
            dims[1]: X[dims[1]][: -d + 1],
            config.get("hankel_coord_name"): (
                dims[0],
                np.repeat(np.arange(d), X[dims[0]].shape[0]),
            ),
        },
        attrs={
            "original_time": X[dims[1]].values,
        },
    )
