import numpy as np
import pandas as pd
import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.weather_utils import (
    compute_climatology,
    compute_energy_spectrum,
    compute_rmse,
)


@pytest.fixture()
def data_generator() -> tuple[xr.DataArray, xr.DataArray]:
    """Generate a prediction and groundtruth DataArrays for testing."""
    time = (pd.date_range("2016-01-01T00", "2019-12-31T00", freq="1D")).to_numpy()
    x = np.arange(-90, 91, 2)
    y = np.arange(0, 361, 2)
    z = np.array([850])
    prediction_generator = DataGenerator(
        x=x,
        y=y,
        z=z,
        t=time,
        vars=["temperature"],
        seed=1234,
    )
    groundtruth_generator = DataGenerator(
        x=x,
        y=y,
        z=z,
        t=time,
        vars=["temperature"],
        seed=1235,
    )

    prediction_generator.generate_dataarray()
    prediction = prediction_generator.da
    groundtruth_generator.generate_dataarray()
    groundtruth = groundtruth_generator.da

    rename_dict = {"x": "latitude", "y": "longitude", "z": "level"}
    prediction = prediction.rename(rename_dict)
    groundtruth = groundtruth.rename(rename_dict)

    return prediction, groundtruth


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
    prediction, groundtruth = data_generator
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


def test_compute_climatology(data_generator):
    """Test for the compute_climatology() function."""
    _, groundtruth = data_generator
    climatology = compute_climatology(groundtruth)

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


def test_compute_energy_spectrum(data_generator):
    """Test for the compute_energy_spectrum() function."""
    prediction, _ = data_generator
    spectrum = compute_energy_spectrum(prediction)
    expected_dims = ("time", "frequency", "level")

    assert set(spectrum.dims) == set(expected_dims), (
        f"Expected dimensions of spectrum to be {expected_dims}, "
        f"but got {spectrum.dims}."
    )
    assert (
        "wavelength" in spectrum.coords
    ), "Expected wavelength to be a coordinate of spectrum."
