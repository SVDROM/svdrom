import numpy as np
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
        lat_weights = np.cos(np.deg2rad(ground_truth.latitude))
        rmse = rmse.weighted(lat_weights)
    rmse = rmse.mean(dim=dims)
    rmse = rmse.clip(min=0)
    return xr.ufuncs.sqrt(rmse)


def compute_climatology(
    data: xr.DataArray,
    months: list[int] | None = None,
) -> xr.DataArray:
    """Given observed data, compute the climatology as a function
    of day of year and hour of day.

    Parameters
    ----------
    data: xr.DataArray
        A dask- or numpy-backed DataArray containing the observed data.
    months: list[int] | None, optional
        The months for which to compute the climatology. Each month should
        be an integer from 1 to 12. For example, if months=[1, 2], then
        the climatology will be computed for January and February. If months
        is None, climatology is calculated for the whole year. The default is
        None.

    Returns
    -------
    xr.DataArray:
        A numpy- or dask-backed DataArray containing the calculated climatology,
        as a function of day of year and hour of day. In this implementation,
        the returned climatology has at most 365 days. Leap years are handled by
        dropping the 29th of Feb.

    Notes
    -----
    If the input data is dask-backed, it is recommended to set up a multi-threading
    Dask cluster before calling the function.

    Examples
    --------
    Given observed data for 2016-2019, compute the climatology for Jan and Feb:

    >>> from dask.distributed import Client
    >>> client = Client(processes=False)  # set up a multi-threading Dask cluster
    >>> climatology = compute_climatology(
            ground_truth.sel(time=slice("2016-01-01", "2019-12-31")),
            months=[1, 2],
        )
    >>> climatology = climatology.compute()  # trigger the computation of the task graph
    >>> client.close()
    """
    if "time" not in data.dims:
        msg = "Expected the input data to have a dimension named time."
        raise ValueError(msg)

    months = sorted(months) if months else None

    # handle 29 Feb on leap years on the data, so that
    # every year has exactly 365 days. Here we assign a new coordinate
    # for dayofyear, which is attached to the time dimension. dayofyear
    # is then used to compute the climatology.
    doy = data.time.dt.dayofyear
    # from Mar onwards on leap years, shift 'doy' forward by 1 day
    shift = data.time.dt.is_leap_year & (data.time.dt.month > 2)
    doy_corrected = xr.where(shift, doy - 1, doy)
    data = data.assign_coords(dayofyear=("time", doy_corrected.data))
    # now drop 29 Feb
    data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))

    # keep only the requested months
    if months:
        data = data.sel(time=data.time.dt.month.isin(months))

    # group by the  day of year and hour of day to compute climatology
    # note: specifying Numpy engine for multi-key groupby to avoid concurrent
    # access error when using Numba (the default engine)
    return data.groupby(["dayofyear", "time.hour"]).mean(engine="numpy")


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
