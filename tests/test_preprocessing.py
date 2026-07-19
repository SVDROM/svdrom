import numpy as np
import pytest
import xarray as xr
from make_test_data import DataGenerator, SignalGenerator

import svdrom.config as config
from svdrom.preprocessing import (
    StandardScaler,
    hankel_preprocessing,
    variable_spatial_stack,
)

data_generator = DataGenerator()
data_generator.generate_dataarray()
data_generator.generate_dataset()

signal_generator = SignalGenerator()
signal_generator.add_sinusoid1()

hankel_coord_name = config.get("hankel_coord_name")
stack_coord_name = "space"
config.set(stack_coord_name=stack_coord_name)


@pytest.mark.parametrize("X", [data_generator.da, data_generator.ds])
def test_variable_spatial_stack(X: xr.DataArray | xr.Dataset):
    """Test for the variable_spatial_stack function."""
    # check that the output is a xarray DataArray
    X_stacked = variable_spatial_stack(X, dims=("x", "y", "z"))
    assert isinstance(X_stacked, xr.DataArray), (
        "The result of stacking should be a Xarray DataArray, "
        f"got {type(X_stacked).__name__}."
    )

    # check that the dimensions are "time" and stack_coord_name
    dims = sorted(map(str, X_stacked.sizes.keys()))
    expected_dims = sorted(["time", stack_coord_name])
    assert (
        dims == expected_dims
    ), f"Expected dimensions after stacking are {expected_dims}, got {dims}."

    # check if shape matches expected shape
    X_stacked = X_stacked.transpose(stack_coord_name, "time")
    len_x, len_y, len_z = len(X.x), len(X.y), len(X.z)
    if isinstance(X, xr.DataArray):
        expected_shape = (len_x * len_y * len_z, len(X.time))
        assert expected_shape == X_stacked.shape, (
            f"Expected shape for a DataArray after stacking is {expected_shape}, "
            f"got {X_stacked.shape}."
        )
    if isinstance(X, xr.Dataset):
        vars = list(X.keys())
        expected_shape = (len_x * len_y * len_z * len(vars), len(X.time))
        assert expected_shape == X_stacked.shape, (
            f"Expected shape for a Dataset after stacking is {expected_shape}, "
            f"got {X_stacked.shape}."
        )

    # check that by unstacking we can recover the original
    X_unstacked = X_stacked.unstack()
    if isinstance(X, xr.DataArray):
        xr.testing.assert_allclose(X, X_unstacked, check_dim_order=False)
    if isinstance(X, xr.Dataset):
        X_unstacked = X_unstacked.to_dataset(dim="variable")
        xr.testing.assert_allclose(X, X_unstacked, check_dim_order=False)


@pytest.mark.parametrize("X", [data_generator.da, data_generator.ds])
def test_standard_scaler(X: xr.DataArray | xr.Dataset):
    """Test for the StandardScaler class."""
    scaler = StandardScaler()
    X_scaled = scaler(X, with_std=True)

    assert hasattr(scaler, "mean")
    assert hasattr(scaler, "std")
    assert hasattr(scaler, "with_std")

    if isinstance(X, xr.DataArray):
        assert isinstance(
            X_scaled, xr.DataArray
        ), f"Expected a DataArray after scaling, got {type(X_scaled).__name__}."
        assert isinstance(
            scaler.mean, xr.DataArray
        ), f"Expected mean to be a DataArray, got {type(scaler.mean).__name__}."
        assert isinstance(
            scaler.std, xr.DataArray
        ), f"Expected std to be a DataArray, got {type(scaler.std).__name__}."
        if scaler.with_std:
            xr.testing.assert_allclose(X, X_scaled * scaler.std + scaler.mean)
        else:
            xr.testing.assert_allclose(X, X_scaled + scaler.mean)
    if isinstance(X, xr.Dataset):
        assert isinstance(
            X_scaled, xr.Dataset
        ), f"Expected a Dataset after scaling, got {type(X_scaled).__name__}."
        assert isinstance(
            scaler.mean, xr.Dataset
        ), f"Expected mean to be a Dataset, got {type(scaler.mean).__name__}."
        assert isinstance(
            scaler.std, xr.Dataset
        ), f"Expected std to be a Dataset, got {type(scaler.std).__name__}."
        if scaler.with_std:
            xr.testing.assert_allclose(X, X_scaled * scaler.std + scaler.mean)
        else:
            xr.testing.assert_allclose(X, X_scaled + scaler.mean)


@pytest.mark.parametrize("generator", [data_generator, signal_generator])
@pytest.mark.parametrize("d", [2, 3])
def test_hankel_preprocessing(generator: DataGenerator | SignalGenerator, d: int):
    """Test for the hankel_preprocessing function."""
    if isinstance(generator, DataGenerator):
        X = generator.da.chunk("auto")  # convert to Dask-backed DataArray
        # stack into single spatial dimension, called stack_coord_name
        X = variable_spatial_stack(X, dims=("x", "y", "z"))
    elif isinstance(generator, SignalGenerator):
        X = generator.da.rename({"x": stack_coord_name})
    else:
        msg = "Input must be an instance of DataGenerator or SignalGenerator."
        raise ValueError(msg)

    X = X.transpose(stack_coord_name, "time")
    n_samples, n_snapshots = X.shape
    X_delayed = hankel_preprocessing(X, d=d)

    assert (
        X_delayed.dims == X.dims
    ), f"Expected dimensions are {X.dims}, but got {X_delayed.dims}."

    expected_coords = list(X.coords)
    expected_coords.append(hankel_coord_name)
    actual_coords = list(X_delayed.coords)
    expected_coords, actual_coords = sorted(expected_coords), sorted(actual_coords)
    assert (
        actual_coords == expected_coords
    ), f"Expected coordinates are {expected_coords}, but got {actual_coords}."

    expected_shape = (d * n_samples, n_snapshots - d + 1)
    assert (
        X_delayed.shape == expected_shape
    ), f"Expected shape is {expected_shape}, but got {X_delayed.shape}."

    expected_lag_coord = np.repeat(np.arange(d), n_samples)
    assert np.array_equal(X_delayed[hankel_coord_name].values, expected_lag_coord), (
        f"Expected the lag coordinate to consist of {n_samples} repeats of "
        f"{np.arange(d)}, but got something else."
    )

    expected_time_coord = X.time.values[: -d + 1]
    assert np.array_equal(
        X_delayed.time.values,
        expected_time_coord,
    ), (
        f"Expected the time coordinate to be equal to t[:{-d + 1}], "
        "where t is the time coordinate of the original matrix."
    )

    original_time_vector = X.time.values
    hankel_time_mapping = X_delayed.attrs[config.get("hankel_time_mapping_attr")]
    assert np.array_equal(
        np.array(list(hankel_time_mapping.keys())),
        original_time_vector,
    ), (
        "Expected the original time vector to be saved in a dictionary as an "
        f"attribute with the name {config.get('hankel_time_mapping_attr')}."
    )
    expected = (original_time_vector[0], 0)
    assert hankel_time_mapping[original_time_vector[0]] == expected, (
        f"Expected hankel_time_mapping[{original_time_vector[0]}] to "
        f"be {expected}, but got "
        f"{hankel_time_mapping[original_time_vector[0]]}."
    )
    i = int(n_snapshots / 2)
    expected = (original_time_vector[i - d + 1], d - 1)
    assert hankel_time_mapping[original_time_vector[i]] == expected, (
        f"Expected hankel_time_mapping[{original_time_vector[i]}] to "
        f"be {expected}, but got "
        f"{hankel_time_mapping[original_time_vector[i]]}."
    )
    i = -1
    expected = (original_time_vector[n_snapshots - d], d - 1)
    assert hankel_time_mapping[original_time_vector[i]] == expected, (
        f"Expected hankel_time_mapping[{original_time_vector[i]}] to "
        f"be {expected}, but got "
        f"{hankel_time_mapping[original_time_vector[i]]}."
    )

    assert np.array_equal(
        X_delayed[(d - 1) * n_samples :, 0].compute().data,
        X_delayed[:n_samples, d - 1].compute().data,
    ), (
        "After Hankel pre-processing, expected "
        f"X[{(d - 1) * n_samples}:, 0] to equal X[:{n_samples}, {d - 1}]."
    )
