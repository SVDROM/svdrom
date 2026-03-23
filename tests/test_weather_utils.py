from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.weather_utils import (
    compute_climatology,
    compute_crps_gaussian,
    compute_energy_spectrum,
    compute_rmse,
    expand_time_climatology,
)


@pytest.fixture()
def data_generator() -> Callable:
    """Generate random prediction and groundtruth DataArrays for testing."""

    def _factory(
        prediction_seed: int = 1234, groundtruth_seed: int = 1235
    ) -> tuple[xr.DataArray, xr.DataArray]:
        time = (pd.date_range("2016-01-01T00", "2019-12-31T00", freq="1D")).to_numpy()
        x = np.arange(-90, 91, 4)
        y = np.arange(0, 361, 4)
        z = np.array([850])

        prediction_generator = DataGenerator(
            x=x,
            y=y,
            z=z,
            t=time,
            vars=["temperature"],
            seed=prediction_seed,
        )
        groundtruth_generator = DataGenerator(
            x=x,
            y=y,
            z=z,
            t=time,
            vars=["temperature"],
            seed=groundtruth_seed,
        )

        prediction_generator.generate_dataarray()
        prediction = prediction_generator.da
        groundtruth_generator.generate_dataarray()
        groundtruth = groundtruth_generator.da

        rename_dict = {"x": "latitude", "y": "longitude", "z": "level"}
        prediction = prediction.rename(rename_dict)
        groundtruth = groundtruth.rename(rename_dict)

        return prediction, groundtruth

    return _factory


@pytest.fixture()
def probabilistic_prediction_generator(
    data_generator,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Generate an ensemble of random predictions, and return their
    mean and standard deviation.
    """
    predictions = []
    n_predictions = 10

    for _ in range(n_predictions):
        prediction, _ = data_generator(prediction_seed=None)
        predictions.append(prediction)

    predictions = xr.concat(predictions, dim="ensemble")
    prediction_mean = predictions.mean("ensemble")
    predictions_std = predictions.std("ensemble")

    return prediction_mean, predictions_std


@pytest.mark.parametrize(
    "dims",
    [
        ("time",),
        ("latitude", "longitude", "level"),
        ("latitude", "longitude", "level", "time"),
    ],
)
def test_compute_rmse(dims, data_generator):
    """Test for the compute_rmse() weather utility function."""
    prediction, groundtruth = data_generator()
    rmse = compute_rmse(groundtruth, prediction, dims=dims, lat_weighting=False)

    match dims:
        case ("time",):
            expected_out_dims = ("latitude", "longitude", "level")
        case ("latitude", "longitude", "level"):
            expected_out_dims = ("time",)
        case ("latitude", "longitude", "level", "time"):
            expected_out_dims = ()
        case _:
            msg = f"Unexpected value for dims: {dims}"
            raise ValueError(msg)

    expected_rmse = (prediction - groundtruth) ** 2  # square
    expected_rmse = expected_rmse.mean(dims)  # mean
    expected_rmse = xr.ufuncs.sqrt(expected_rmse)  # root

    xr.testing.assert_allclose(rmse, expected_rmse)
    assert set(rmse.dims) == set(expected_out_dims)


@pytest.mark.parametrize(
    "smooth_window",
    [
        pytest.param(None, marks=pytest.mark.dependency(name="clima_None")),
        pytest.param(61, marks=pytest.mark.dependency(name="clima_int")),
    ],
)
def test_compute_climatology(smooth_window, data_generator):
    """Test for the compute_climatology() function."""
    _, groundtruth = data_generator()
    climatology = compute_climatology(groundtruth, smooth_window)

    time_series = pd.Series(groundtruth.time.values)
    contains_leap_year = time_series.dt.is_leap_year.any()
    expected_doy = np.arange(1, 367) if contains_leap_year else np.arange(1, 366)

    freq = (
        (np.unique(np.diff(groundtruth.time))[0]).astype("timedelta64[h]").astype("int")
    )
    hours_in_day = 24
    expected_hour = np.arange(0, hours_in_day, step=freq)

    expected_dims = {"latitude", "longitude", "level", "dayofyear", "hour"}
    assert set(climatology.dims) == expected_dims, (
        f"Climatology should have dimensions: {tuple(expected_dims)}, "
        f"but got {tuple(climatology.dims)}."
    )
    assert np.array_equal(climatology.dayofyear.values, expected_doy), (
        f"Expected doyofyear dimension to be {expected_doy}, "
        f"but got {climatology.dayofyear.values}."
    )
    assert np.array_equal(climatology.hour, expected_hour), (
        f"Expected doyofyear dimension to be {expected_hour}, "
        f"but got {climatology.hour.values}."
    )


@pytest.mark.dependency(depends=["clima_None", "clima_int"])
@pytest.mark.parametrize("doy", [slice(1, 60), slice(180, 240)])
@pytest.mark.parametrize("year", [2020, 2021, 2023, 2024])
def test_expand_time_climatology(doy, year, data_generator):
    """Test for the expand_time_climatology() function."""
    _, groundtruth = data_generator()
    climatology = compute_climatology(groundtruth)
    climatology = climatology.sel(dayofyear=doy)
    hours = climatology.hour.values
    climatology = expand_time_climatology(climatology, year)

    expected_time = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="1h")
    expected_time = expected_time[
        expected_time.dayofyear.isin(range(doy.start, doy.stop + 1))
        & expected_time.hour.isin(hours)
    ]
    assert np.array_equal(climatology.time.values, expected_time), (
        f"Expected time vector to be: {expected_time}, "
        f"but instead got: {climatology.time.values}."
    )


def test_compute_energy_spectrum(data_generator):
    """Test for the compute_energy_spectrum() function."""
    prediction, _ = data_generator()
    spectrum = compute_energy_spectrum(prediction)
    expected_dims = ("time", "frequency", "level")

    assert set(spectrum.dims) == set(expected_dims), (
        f"Expected dimensions of spectrum to be {expected_dims}, "
        f"but got {spectrum.dims}."
    )
    assert (
        "wavelength" in spectrum.coords
    ), "Expected wavelength to be a coordinate of spectrum."


@pytest.mark.parametrize("dims", [("latitude", "longitude"), "time", None])
@pytest.mark.parametrize("lat_weighting", [True, False])
def test_compute_crps_gaussian(
    data_generator,
    probabilistic_prediction_generator,
    dims,
    lat_weighting,
):
    """Test for the compute_crps_gaussian() function."""
    _, groundtruth = data_generator()
    prediction_mean, prediction_std = probabilistic_prediction_generator

    crps = compute_crps_gaussian(
        groundtruth,
        prediction_mean,
        prediction_std,
        lat_weighting=lat_weighting,
        dims=dims,
    )

    match dims:
        case ("latitude", "longitude"):
            expected_dims = set(groundtruth.dims) - set(dims)
        case "time":
            expected_dims = set(groundtruth.dims) - {"time"}
        case None:
            expected_dims = set(groundtruth.dims)
        case _:
            msg = f"Unexpected value for dims: {dims}"
            raise ValueError(msg)

    assert set(crps.dims) == expected_dims, (
        f"Expected dimensions of CRPS to be {tuple(expected_dims)}, "
        f"but got {crps.dims} instead."
    )
