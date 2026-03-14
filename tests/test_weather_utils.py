import pytest
from make_test_data import DataGenerator

from svdrom.weather_utils import compute_rmse


@pytest.mark.parametrize(
    "dims",
    [
        ("time",),
        ("latitude", "longitude", "level"),
        ("latitude", "longitude", "level", "time"),
    ],
)
def test_compute_rmse(dims):
    """Test for the compute_rmse() weather utility function."""
    prediction_generator = DataGenerator(vars=["temperature"], seed=1234)
    groundtruth_generator = DataGenerator(vars=["temperature"], seed=1235)

    prediction_generator.generate_dataarray()
    prediction = prediction_generator.da
    groundtruth_generator.generate_dataarray()
    groundtruth = groundtruth_generator.da

    rename_dict = {"x": "latitude", "y": "longitude", "z": "level"}
    prediction = prediction.rename(rename_dict)
    groundtruth = groundtruth.rename(rename_dict)

    rmse = compute_rmse(groundtruth, prediction, dims=dims)

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

    assert set(rmse.dims) == set(expected_out_dims)
