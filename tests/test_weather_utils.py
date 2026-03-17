import numpy as np
import pandas as pd
import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.weather_utils import compute_rmse


@pytest.fixture()
def data_generator() -> tuple[xr.DataArray, xr.DataArray]:
    """Generate a prediction and groundtruth DataArrays for testing."""
    time = (pd.date_range("2017-01-01T00", "2019-12-31T00", freq="1D")).to_numpy()
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
            msg = "Unexpected value for dims: {dims}"
            raise ValueError(msg)

    expected_rmse = (prediction - groundtruth) ** 2  # square
    expected_rmse = expected_rmse.mean(dims)  # mean
    expected_rmse = xr.ufuncs.sqrt(expected_rmse)  # root

    xr.testing.assert_allclose(rmse, expected_rmse)
    assert set(rmse.dims) == set(expected_out_dims)
