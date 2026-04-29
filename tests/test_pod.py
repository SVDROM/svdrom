import dask.array as da
import numpy as np
import pytest
import xarray as xr

from svdrom.pod import POD


def make_dataarray(matrix_type: str, time_dim_pos: int = 1) -> xr.DataArray:
    """Make a Dask-backed DataArray with random data of
    specified matrix type. The matrix type can be one of:
    - "tall-and-skinny": More samples than features.
    - "short-and-fat": More features than samples.
    - "square": Equal number of samples and features.

    """
    if matrix_type == "tall-and-skinny":
        n_space, n_time = 10_000, 100
        space_chunks, time_chunks = -1, int(n_time / 2)
    elif matrix_type == "short-and-fat":
        n_space, n_time = 100, 10_000
        space_chunks, time_chunks = int(n_space / 2), -1
    elif matrix_type == "square":
        n_space, n_time = 1_000, 1_000
        space_chunks, time_chunks = int(n_space / 2), int(n_time / 2)
    else:
        msg = (
            "Matrix type not supported. "
            "Must be one of: tall-and-skinny, short-and-fat, square."
        )
        raise ValueError(msg)

    if time_dim_pos == 1:
        shape = (n_space, n_time)
        chunks = (space_chunks, time_chunks)
        dims = ["space", "time"]
        coords = {"space": np.arange(n_space), "time": np.arange(n_time)}
    elif time_dim_pos == 0:
        shape = (n_time, n_space)
        chunks = (time_chunks, space_chunks)
        dims = ["time", "space"]
        coords = {"time": np.arange(n_time), "space": np.arange(n_space)}
    else:
        msg = "time_dim_pos must be 0 or 1."
        raise ValueError(msg)

    X = da.random.random(shape, chunks=chunks).astype("float32")
    return xr.DataArray(X, dims=dims, coords=coords)


@pytest.mark.parametrize("svd_algorithm", ["tsqr", "randomized"])
def test_pod_basic_attributes_and_aliases(svd_algorithm):
    """Test basic functionality and property aliases of POD."""
    n_modes = 10
    pod = POD(
        n_modes=n_modes,
        svd_algorithm=svd_algorithm,
        compute_energy_ratio=True,
    )

    expected_attrs = (
        "modes",
        "time_coeffs",
        "energy",
        "explained_energy_ratio",
    )
    for attr in expected_attrs:
        assert hasattr(pod, attr), f"POD should have attribute '{attr}'."

    X = make_dataarray("tall-and-skinny")
    pod.fit(X)

    assert isinstance(pod.modes, xr.DataArray)
    assert isinstance(pod.modes.data, np.ndarray)
    assert isinstance(pod.time_coeffs, xr.DataArray)
    assert isinstance(pod.time_coeffs.data, np.ndarray)
    assert isinstance(pod.energy, np.ndarray)
    assert isinstance(pod.explained_energy_ratio, np.ndarray)

    assert pod.modes is pod.u
    assert pod.time_coeffs is pod.v
    assert pod.explained_energy_ratio is pod.explained_var_ratio


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat", "square"])
def test_pod_shapes_and_dims(matrix_type):
    """Test that POD modes and time coefficients have the correct shapes and dims."""
    X = make_dataarray(matrix_type, time_dim_pos=1)
    n_space, n_time = X.shape
    n_modes = 10

    pod = POD(n_modes=n_modes)
    pod.fit(X)

    assert pod.modes.shape == (n_space, n_modes)
    assert pod.time_coeffs.shape == (n_modes, n_time)
    assert pod.energy.shape == (n_modes,)

    assert pod.modes.dims == ("space", "components")
    assert pod.time_coeffs.dims == ("components", "time")

    assert "space" in pod.modes.coords
    assert "components" in pod.modes.coords
    assert "time" not in pod.modes.coords

    assert "time" in pod.time_coeffs.coords
    assert "components" in pod.time_coeffs.coords
    assert "space" not in pod.time_coeffs.coords


def test_time_dimension_handling():
    """Test that POD correctly handles the time_dimension parameter
    by transposing if necessary."""
    X = make_dataarray("short-and-fat", time_dim_pos=0)
    assert X.dims == ("time", "space")
    n_time, n_space = X.shape
    n_modes = 15

    pod = POD(n_modes=n_modes, time_dimension="time")
    pod.fit(X)

    assert pod.modes.shape == (n_space, n_modes)
    assert pod.time_coeffs.shape == (n_modes, n_time)

    assert pod.modes.dims == ("space", "components")
    assert "space" in pod.modes.coords
    assert pod.time_coeffs.dims == ("components", "time")
    assert "time" in pod.time_coeffs.coords


def test_remove_mean():
    X = make_dataarray("tall-and-skinny")
    n_modes = 5

    pod = POD(n_modes=n_modes, remove_mean=True, time_dimension="time")
    pod.fit(X)

    # (U*S @ V).
    reconstructed_fluctuations = (pod.modes.data * pod.s) @ pod.time_coeffs.data

    # *The temporal mean of the reconstructed fluctuations should be close to zero
    mean_of_reconstruction = reconstructed_fluctuations.mean(axis=1)
    assert np.allclose(mean_of_reconstruction, 0, atol=1e-5)


def test_energy_calculation():
    """Test that the `energy` property is calculated correctly."""
    X = make_dataarray("square")
    n_modes = 20

    pod = POD(n_modes=n_modes, compute_energy_ratio=True, remove_mean=True)
    pod.fit(X)

    n_samples = pod.modes.shape[0]
    expected_energy = pod.s**2 / n_samples
    assert np.allclose(pod.energy, expected_energy, atol=1e-6)

    # The total variance is the total energy of the system
    X_processed = X - X.mean(dim="time")
    total_variance = (X_processed.data**2).sum().compute() / n_samples

    # The explained energy ratio of each mode should be its energy divided
    # by the total system energy (total variance)
    assert np.allclose(
        pod.explained_energy_ratio,
        pod.energy / total_variance,
        rtol=1e-2,
    )


def test_invalid_time_dimension_error():
    """Test that a ValueError is raised for a non-existent time dimension."""
    X = make_dataarray("tall-and-skinny")
    pod = POD(n_modes=5, time_dimension="non_existent_dim")

    with pytest.raises(ValueError, match="is not a dimension of the input array"):
        pod.fit(X)


def test_compute_methods():
    """Test that the `compute_*` convenience methods work."""
    n_modes = 5
    pod = POD(
        n_modes=n_modes,
        compute_modes=False,
        compute_time_coeffs=False,
        compute_energy_ratio=False,
    )

    X = make_dataarray("tall-and-skinny")
    pod.fit(X)

    assert isinstance(pod.modes.data, da.Array)
    assert isinstance(pod.time_coeffs.data, da.Array)
    assert isinstance(pod.explained_energy_ratio, da.Array)

    pod.compute_modes()
    assert isinstance(pod.modes.data, np.ndarray)

    pod.compute_time_coeffs()
    assert isinstance(pod.time_coeffs.data, np.ndarray)

    pod.compute_energy_ratio()
    assert isinstance(pod.explained_energy_ratio, np.ndarray)
