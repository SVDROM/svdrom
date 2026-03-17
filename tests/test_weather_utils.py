import numpy as np
import pandas as pd
import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.weather_utils import compute_climatology, compute_rmse


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


@pytest.mark.parametrize("year", [2020, 2021, 2022, 2023])
@pytest.mark.parametrize("months", [[1, 2, 3], [7, 8, 9]])
def test_compute_climatology(year, months, data_generator):
    """Test for the compute_climatology() function."""
    _, groundtruth = data_generator
    freq = np.unique(np.diff(groundtruth.time))[0]
    climatology = compute_climatology(groundtruth, year, months)

    match months:
        case [1, 2, 3]:
            expected_date_range = pd.date_range(
                f"{year}-01-01T00", f"{year}-03-31T00", freq=pd.Timedelta(freq)
            )
        case [7, 8, 9]:
            expected_date_range = pd.date_range(
                f"{year}-07-01T00", f"{year}-09-30T00", freq=pd.Timedelta(freq)
            )
        case _:
            msg = f"Unexpected value for months: {months}"
            raise ValueError(msg)

    # on leap years, climatology should not contain 29 Feb
    date_to_drop = pd.Timestamp("2020-02-29").date()
    mask = expected_date_range.date != date_to_drop
    expected_date_range = expected_date_range[mask]
    expected_date_range = expected_date_range.to_numpy()

    assert set(climatology.dims) == set(groundtruth.dims), (
        "Climatology should have the same dimensions as the dataset "
        "used to compute it."
    )
    assert np.array_equal(
        climatology.time.values, expected_date_range
    ), "Climatology does not have the expected time vector."
