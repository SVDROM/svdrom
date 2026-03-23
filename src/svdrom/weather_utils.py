import numpy as np
import pandas as pd
import properscoring as ps  # type: ignore[import-not-found]
import xarray as xr
from weatherbench2.derived_variables import (
    ZonalEnergySpectrum,
    interpolate_spectral_frequencies,
)


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
        A dask- or numpy-backed DataArray containing the calculated RMSE score.

    Notes
    -----
    If the inputs are dask-backed DataArrays, the function will build the
    task graph lazily, which you can then execute manually. You should set up
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
    if rmse.size == 0:
        msg = (
            "The resulting array is empty. Do the ground truth and prediction "
            "arrays share coordinates?"
        )
        raise ValueError(msg)
    if lat_weighting:
        lat_weights = np.cos(np.deg2rad(ground_truth.latitude))
        rmse = rmse.weighted(lat_weights)
    rmse = rmse.mean(dim=dims)
    rmse = rmse.clip(min=0)
    return xr.ufuncs.sqrt(rmse)


def compute_climatology(
    data: xr.DataArray,
    smooth_window: int | None = 61,
) -> xr.DataArray:
    """Given observed data, compute the climatology as a function
    of day of year (doy) and hour of day.

    Parameters
    ----------
    data: xr.DataArray
        A dask- or numpy-backed DataArray containing the observed data.
    smooth_window: int | None
        The size (in days) of a sliding window around each doy-hour combination
        with weights linearly decaying to zero from the center, used to compute
        a rolling weighted mean. This removes sample noise and results in a
        smoother climatology. It should be an odd number. If you don't want to
        perform this weighted average, set to None. The default is 61, same as
        in Weatherbench2.

    Returns
    -------
    xr.DataArray:
        A numpy- or dask-backed DataArray containing the calculated climatology,
        as a function of day of year and hour of day.

    Notes
    -----
    If the input data is dask-backed, it is recommended to set up a multi-threading
    Dask cluster before calling the function. Note that performing the smoothing via
    the rolling weighted average is considerably more expensive and a large amount of
    disk spillage from Dask is expected.

    Examples
    --------
    Given observed data for 2016-2019, compute the climatology:

    >>> from dask.distributed import Client
    >>> client = Client(processes=False)  # set up a multi-threading Dask cluster
    >>> climatology = compute_climatology(era5)
    >>> # say you were only interested in the first 60 days of the year:
    >>> climatology = climatology.sel(dayofyear=slice(1, 60))
    >>> climatology = climatology.compute()  # trigger the computation of the task graph
    >>> client.close()
    """
    if "time" not in data.dims:
        msg = "Expected the input data to have a dimension named time."
        raise ValueError(msg)

    # group by the  day of year and hour of day to compute climatology
    # note: specifying Numpy engine for multi-key groupby to avoid concurrent
    # access error when using Numba (the default engine)
    raw_clim = data.groupby(["time.dayofyear", "time.hour"]).mean(engine="numpy")

    if not smooth_window:
        return raw_clim

    half = smooth_window // 2
    n_days = raw_clim.sizes["dayofyear"]

    # circular padding: last `half` days prepended, first `half` days appended
    padded = xr.concat(
        [
            raw_clim.isel(dayofyear=slice(-half, None)),
            raw_clim,
            raw_clim.isel(dayofyear=slice(None, half)),
        ],
        dim="dayofyear",
    ).assign_coords(dayofyear=np.arange(1, 1 + n_days + 2 * half))

    # triangular weights: center gets weight 1.0, edges (+half and -half days)
    # get 1/(half+1)
    weights = xr.DataArray(
        np.maximum(0.0, 1.0 - np.abs(np.arange(-half, half + 1)) / (half + 1)),
        dims=["window"],
    )

    # now we apply a rolling weighted mean along the window dimension,
    smooth_clim = (
        padded.rolling(dayofyear=smooth_window, center=True)
        .construct("window")
        .isel(dayofyear=slice(half, half + n_days))  # strip padding
        .weighted(weights)
        .mean("window")
        .astype(raw_clim.dtype)  # return to original precision
    )

    # restore original coords
    return smooth_clim.assign_coords(dayofyear=raw_clim.dayofyear.values)


def expand_time_climatology(climatology: xr.DataArray, year: int) -> xr.DataArray:
    """Given climatology as a function of day of year and hour of day (such as the
    one returned by compute_climatology()), convert the dayofyear and hour dimensions
    into a single time dimension.

    Parameters
    ----------
    climatology: xr.DataArray
        Climatology as a function of day of year and hour of day. It must contain
        dimensions "dayofyear" and "hour".

    year: int
        The year that will be used to construct the new time dimension.

    Returns
    -------
    xr.DataArray:
        The climatology with the dayofyear and hour dimensions converted into a
        single time dimension. The output DataArray is numpy-backed.

    Notes
    -----
    If the input DataArray is dask-backed, the function will trigger the computation
    of the underlying task graph and load the data into memory, so be mindful about
    the size of the input DataArray.
    """
    if "dayofyear" not in climatology.dims or "hour" not in climatology.dims:
        msg = (
            "The input climatology is expected to have dimensions "
            "'dayofyear' and 'hour'."
        )
        raise ValueError(msg)
    climatology = climatology.compute()
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="1h")
    times = times[
        times.dayofyear.isin(climatology.dayofyear.values)
        & times.hour.isin(climatology.hour.values)
    ]
    climatology = climatology.stack(time=("dayofyear", "hour"))
    climatology = climatology.drop_vars(["time", "dayofyear", "hour"])
    return climatology.assign_coords(time=times)


def compute_energy_spectrum(
    data: xr.DataArray,
    lat_range: tuple[int, int] | None = (30, 60),
) -> xr.DataArray:
    """Compute the zonal energy spectrum along lines of constant
    latitude as a function of wavenumber (unitless), frequency (1/m)
    and wavelength (m).
    The spectrum is computed using Weatherbench2.

    Parameters
    ----------
    data: xr.DataArray
        The variable on a lat/lon grid for which to compute the energy spectrum.
        The DataArray must be backed by a Numpy array.
    lat_range: tuple[int, int] | None, optional
        Latitude range over which to perform an average of the energy spectrum.
        The default is (30, 60), meaning that the output energy spectrum is the
        average for 30deg < |lat| < 60deg. If lat_range is None, no averaging is
        performed, and the output energy spectrum is given as a function of
        wavenumber instead of frequency.

    Returns
    -------
    xr.DataArray:
        A Numpy-backed DataArray containing the computed zonal energy spectrum.
        If lat_range is None, the spectrum is returned as a function of latitude
        and wavenumber (unitless). If lat_range is a tuple, the spectrum is returned
        as a function of frequency (1/m), where the frequency corresponds to the
        most narrow range of frequencies found in lat_range. To perform the average,
        latitudes closer to the Equator are given a larger weight than latitudes
        closer to the poles.

    Notes
    -----
    The input DataArray must not be backed by a Dask array, so make sure you select
    a relatively small slice and load it onto memory.

    Example
    -------
    >>> era5 = xr.open_dataset("era5_slice.zarr", chunks="auto", engine="zarr")
    >>> temperature = era5["temperature"]
    >>> temperature = temperature.sel(time="2020-01-01")
    >>> temperature.load()
    >>> spectrum = compute_energy_spectrum(temperature)
    """
    if not isinstance(data.data, np.ndarray):
        msg = (
            "The input DataArray must be backed by a Numpy array. "
            "Load the data into memory before calling the function."
        )
        raise ValueError(msg)

    ds = data.to_dataset(name="variable")
    var_name = str(next(iter(ds.data_vars)))
    spectrum = ZonalEnergySpectrum(var_name).compute(ds)

    if lat_range:
        lat = spectrum.latitude
        lat_mask = (np.abs(lat) >= lat_range[0]) & (np.abs(lat) <= lat_range[1])
        spectrum = spectrum.sel(latitude=lat_mask)

        spectrum = interpolate_spectral_frequencies(
            spectrum, wavenumber_dim="zonal_wavenumber"
        )
        weights = np.cos(np.deg2rad(spectrum.latitude))
        spectrum = spectrum.weighted(weights).mean("latitude")

    return spectrum


def compute_crps_gaussian(
    ground_truth: xr.DataArray,
    prediction_mean: xr.DataArray,
    prediction_std: xr.DataArray,
    lat_weighting: bool = True,
    dims: str | tuple[str, ...] | None = ("latitude", "longitude"),
) -> xr.DataArray:
    """Compute the Continuous Ranked Probability Score (CRPS) for
    a probabilistic forecast, assuming a Gaussian distribution.

    Parameters
    ----------
    ground_truth: xr.DataArray
        The ground truth data. Must be a numpy-backed array.
    prediction_mean: xr.DataArray
        The mean of the predictions ensemble. Must be a
        numpy-backed array.
    prediction_std: xr.DataArray
        The standard deviation of the predictions ensemble.
        Must be a numpy-backed array.
    lat_weights: bool, optional
        Whether to apply a weighting function so that spatial locations
        in a lat/lon grid closer to the Equator receive a larger weight
        than those closer to the poles. Default is True.
    dims: str | tuple[str] | None, optional
        Dimensions along which to average the CRPS. The default is
        ("latitude", "longitude"), which would return the CRPS as a function
        of prediction time (assuming a single pressure level). Set to None
        if you don't want to perform averaging.

    Returns
    -------
    xr.DataArray:
        The Continuous Ranked Probability Score as a numpy-backed array.

    Notes
    -----
    The function does not currently support dask-backed arrays. All input
    arrays must be numpy-backed. CRPS is calculated using crps_gaussian() from
    the properscoring library: https://github.com/properscoring/properscoring
    """

    for arr in (ground_truth, prediction_mean, prediction_std):
        if not isinstance(arr.data, np.ndarray):
            msg = (
                "The input DataArray must be numpy-backed. "
                "Call .compute() to materialize it in memory."
            )
            raise ValueError(msg)

    if (prediction_std.values <= 0.0).any():
        msg = (
            "The prediction standard deviation array must "
            "only contain positive values."
        )
        raise ValueError(msg)

    crps = xr.apply_ufunc(
        ps.crps_gaussian,
        ground_truth,
        prediction_mean,
        prediction_std,
        join="exact",
    )

    if lat_weighting:
        lat_weights = np.cos(np.deg2rad(crps.latitude))
        crps = crps.weighted(lat_weights)
    if dims:
        crps = crps.mean(dim=dims)

    return crps.clip(min=0)
