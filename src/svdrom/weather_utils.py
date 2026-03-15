from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr


def compute_rmse(
    ground_truth: xr.DataArray,
    prediction: xr.DataArray,
    lat_weighting: bool = True,
    dims: str | tuple[str, ...] = ("latitude", "longitude"),
) -> xr.DataArray:
    """Compute the Root Mean Squared Error (RMSE) of a prediction,
    averaging along the specified dimension(s).

    Parameters
    ----------
    ground_truth: xr.DataArray
        A dask- or numpy-backed DataArray containing the ground truth
        data.
    prediction: xr.DataArray
        A dask- or numpy-backed DataArray containing the prediction (
        a reconstruction or a forecast). The prediction and ground truth
        must be defined on the same spatio-temporal grid.
    lat_weights: bool, optional
        Whether to apply a weighting function so that spatial locations
        in a lat/lon grid closer to the Equator receive a larger weight
        than those closer to the poles. Default is True.
    dims: str | tuple[str], optional
        Dimensions along which to average the RMSE. The default is
        ("latitude", "longitude"), which would return the RMSE as a function
        of prediction time (assuming a single pressure level).

    Returns
    -------
    xr.DataArray:
        A numpy-backed DataArray containing the calculated RMSE score.

    Notes
    -----
    If the inputs are dask-backed DataArrays, the function will build the
    task graph lazily, which you can then execute. You should set up
    a multi-threading Dask cluster before calling the function.

    Examples
    --------
    Given dask-backed DataArrays for ground truth and prediction, compute the
    RMSE averaged over latitude and longitude:
    >>> from dask.distributed import Client
    >>> client = Client(processes=False)  # set up a multi-threading Dask cluster
    >>> rmse = compute_rmse(ground_truth, prediction)
    >>> # trigger computation and materialize as numpy-backed DataArray
    >>> rmse = rmse.compute()
    """
    if set(ground_truth.dims) != set(prediction.dims):
        msg = "The dimensions of the ground truth data and prediction don't match."
        raise ValueError(msg)
    if "latitude" not in ground_truth.dims:
        msg = "Expected the input data to have a dimension named latitude."
        raise ValueError(msg)
    prediction = prediction.real  # keep only real part of the prediction
    rmse = ground_truth.copy(data=(ground_truth - prediction) ** 2)
    if lat_weighting:
        lat_weights_np = np.cos(np.deg2rad(ground_truth.latitude.values))
        lat_weights_np = lat_weights_np / np.mean(lat_weights_np)
        lat_weights_dict = {}
        lat_weights_dict["data"] = lat_weights_np
        lat_weights_dict["dims"] = "latitude"
        lat_weights_dict["latitude"] = {"data": rmse.latitude, "dims": ("latitude")}
        lat_weights = xr.DataArray.from_dict(lat_weights_dict)
        rmse *= lat_weights
    rmse = rmse.mean(dim=dims)
    rmse = rmse.clip(min=0)
    return xr.ufuncs.sqrt(rmse)


def compute_climatology(
    data: xr.DataArray,
    year: int,
    months: list[int],
) -> xr.DataArray:
    """Given observed data, compute a climatological forecast over
    the specified year and months.

    Parameters
    ----------
    data: xr.DataArray
        A dask- or numpy-backed DataArray containing the observed data.
        The data must have a time dimension with uniform spacing, and
        the time coordinate must be parseable by Pandas.
    year: int
        The year for which to compute the climatology. The year is only
        used to construct the time vector for the climatological forecast,
        and does not affect the actual values of the climatology.
    months: list[int]
        The months for which to compute the climatology. Each month should
        be an integer from 1 to 12. For example, if months=[1, 2], then
        the climatological forecast will be computed for January and February
        of the specified year.

    Returns
    -------
    xr.DataArray:
        A numpy- or dask-backed DataArray containing the calculated climatology,
        with the time coordinate corresponding to the specified year and months.

    Notes
    -----
    If the input data is dask-backed, it is recommended to set up a multi-threading
    Dask cluster before calling the function.

    Examples
    --------
    Given observed data for 2016-2019, compute the climatological forecast for
    Jan and Feb 2020:

    >>> from dask.distributed import Client
    >>> client = Client(processes=False)  # set up a multi-threading Dask cluster
    >>> climatology = compute_climatology(
            ground_truth.sel(time=slice("2016-01-01", "2019-12-31")),
            year=2020,
            months=[1, 2],
        )
    >>> climatology = climatology.compute()  # trigger the computation of the task graph
    >>> client.close()
    """
    if "time" not in data.dims:
        msg = "Expected the input data to have a dimension named time."
        raise ValueError(msg)

    months = sorted(months)

    # construct the time vector for climatology
    dt = (np.unique(np.diff(data.time.values))).astype("timedelta64[h]").astype(int)
    if len(dt) > 1:
        msg = "The time axis of the input data must have uniform spacing."
        raise ValueError(msg)
    dt = dt[0]
    times = pd.date_range(
        f"{year}-01-01", f"{year}-12-31 23:00", freq=timedelta(hours=int(dt))
    )
    times = times[times.month.isin(months)]
    # drop 29 Feb on leap years
    times = times[~((times.month == 2) & (times.day == 29))]

    # handle 29 Feb on leap years on the data, so that
    # every year has exactly 365 days. Here we assign a new coordinate
    # for dayofyear (doy), which is attached to the time dimension. doy
    # is later used to compute the climatology.
    doy = data.time.dt.dayofyear
    # from Mar onwards on leap years, shift 'doy' forward by 1 day
    shift = data.time.dt.is_leap_year & (data.time.dt.month > 2)
    doy_corrected = xr.where(shift, doy - 1, doy)
    data = data.assign_coords(doy=("time", doy_corrected.data))
    # now drop 29 Feb
    data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))

    # keep only the requested months
    data = data.sel(time=data.time.dt.month.isin(months))

    # group by the  day of year and hour of day to compute climatology
    # note: specifying Numpy engine for multi-key groupby to avoid concurrent
    # access error when using Numba (the default engine)
    clima = data.groupby(["doy", "time.hour"]).mean(engine="numpy")
    clima = clima.stack(time=("doy", "hour"))
    clima = clima.drop_vars(["doy", "hour"])

    return clima.assign_coords(time=times)
