import dask.array as da
import numpy as np
import pytest
import xarray as xr

from svdrom.pod import POD

N_MODES = 10


def make_dataarray(matrix_type: str, time_dim_pos: int = 1) -> xr.DataArray:
    """Make a Dask-backed DataArray with random data of
    specified matrix type.

    Parameters
    ----------
    matrix_type: str
        Can be one of:
        - "tall-and-skinny": More spatial locations than snapshots.
        - "short-and-fat": More snapshots than spatial locations.
        - "square": Equal number of spatial locations and snapshots.
    time_dim_pos: int
        Set to 0 if you want the temporal dimension to be along the rows,
        or set to 1 if you want the temporal dimension to be along the
        columns. The default is 1.

    Returns
    -------
    xr.DataArray
        A Dask-backed DataArray of random data of the requested characteristics.
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
def test_basic(svd_algorithm):
    """Test basic functionality of POD, using the two backend
    SVD algorithms.
    """
    n_modes = N_MODES
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

    assert isinstance(
        pod.modes, xr.DataArray
    ), f"modes should be an xarray DataArray, got {type(pod.modes)}."
    assert isinstance(pod.modes.data, np.ndarray), (
        "modes should be a xarray DataArray with numpy ndarray data, "
        f"got {type(pod.modes.data)}."
    )
    assert isinstance(
        pod.time_coeffs, xr.DataArray
    ), f"time_coeffs should be an xarray DataArray, got {type(pod.time_coeffs)}."
    assert isinstance(pod.time_coeffs.data, np.ndarray), (
        "time_coeffs should be a xarray DataArray with numpy ndarray data, "
        f"got {type(pod.time_coeffs.data)}."
    )
    assert isinstance(
        pod.energy, np.ndarray
    ), f"energy should be a numpy ndarray, got {type(pod.energy)}."
    assert isinstance(pod.explained_energy_ratio, np.ndarray), (
        "explained_energy_ratio should be a numpy ndarray, "
        f"got {type(pod.explained_energy_ratio)}."
    )

    assert np.array_equal(
        pod.modes.data, pod.u.data
    ), "The POD spatial modes should equal the SVD left singular vectors."
    assert np.array_equal(pod.time_coeffs.data, np.diag(pod.s) @ pod.v.data), (
        "The POD time coefficients should equal the "
        "SVD right singular vectors scaled by the singular values."
    )
    assert np.array_equal(
        pod.energy, pod._s**2
    ), "The POD energy should equal the square of the SVD singular values."
    assert np.array_equal(
        pod.explained_energy_ratio.data, pod.explained_var_ratio.data
    ), (
        "The POD explained energy ratio should equal "
        "the SVD explained variance ratio."
    )


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat", "square"])
def test_pod_shapes_and_dims(matrix_type):
    """Test that POD modes and time coefficients have the correct shapes and dims."""
    X = make_dataarray(matrix_type)
    n_space, n_time = X.shape
    n_modes = N_MODES

    pod = POD(n_modes=n_modes)
    pod.fit(X)

    assert pod.modes is not None
    assert pod.time_coeffs is not None
    assert pod.energy is not None

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


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_orthogonality(algorithm):
    """Test orthogonality of POD modes and time coefficients."""
    X = make_dataarray("tall-and-skinny")
    n_modes = N_MODES
    pod = POD(n_modes=n_modes, svd_algorithm=algorithm)
    pod.fit(X)

    assert pod.modes is not None
    assert pod.time_coeffs is not None
    assert pod.energy is not None

    identity_k = np.eye(pod.n_components, dtype=np.float32)
    modes, time_coeffs = pod.modes.data, pod.time_coeffs.data
    modes_ortho = modes.T @ modes
    time_coeffs_ortho = time_coeffs @ time_coeffs.T

    assert np.allclose(
        modes_ortho, identity_k, atol=1e-4
    ), "modes.T @ modes is not close to identity."
    assert np.allclose(
        time_coeffs_ortho, np.diag(pod.energy), atol=1e-4
    ), "time_coeffs @ time_coeffs.T is not close to np.diag(energy)"


def test_time_dimension_handling():
    """Test that POD correctly handles the time_dimension parameter
    by transposing if necessary."""
    X = make_dataarray("short-and-fat", time_dim_pos=0)
    assert X.dims == ("time", "space")
    n_time, n_space = X.shape
    n_modes = N_MODES

    pod = POD(n_modes=n_modes, time_dimension="time")
    pod.fit(X)

    assert pod.modes is not None
    assert pod.time_coeffs is not None

    assert pod.modes.shape == (n_space, n_modes)
    assert pod.time_coeffs.shape == (n_modes, n_time)

    assert pod.modes.dims == ("space", "components")
    assert "space" in pod.modes.coords
    assert "time" not in pod.modes.coords
    assert pod.time_coeffs.dims == ("components", "time")
    assert "time" in pod.time_coeffs.coords
    assert "space" not in pod.time_coeffs.coords


def test_remove_mean():
    X = make_dataarray("tall-and-skinny")
    n_modes = N_MODES

    pod = POD(n_modes=n_modes)
    pod.fit(X)

    assert pod.modes is not None
    assert pod.time_coeffs is not None

    reconstructed_fluctuations = pod.modes @ pod.time_coeffs

    mean_of_reconstruction = reconstructed_fluctuations.mean("time")
    assert np.allclose(
        mean_of_reconstruction, 0, atol=1e-5
    ), "Expected the mean of the reconstructed fluctuations to be close to zero"


def test_energy_calculation():
    """Test that the 'energy' property returns the eigenvalues of the
    spatial covariance matrix, computed via the snapshot method."""
    X = make_dataarray("tall-and-skinny", time_dim_pos=1)
    n_snapshots = X.sizes["time"]
    n_modes = N_MODES

    pod = POD(n_modes=n_modes, svd_algorithm="tsqr")
    pod.fit(X)
    assert pod.energy is not None

    X_fluc = (X - X.mean(dim="time")).values  # (n_space, n_time)
    T = X_fluc.T @ X_fluc / n_snapshots  # (n_time, n_time) temporal correlation
    ref_eigenvalues = np.sort(np.linalg.eigvalsh(T))[::-1][:n_modes]

    assert np.allclose(pod.energy, ref_eigenvalues, rtol=1e-3)


def test_invalid_time_dimension_error():
    """Test that a ValueError is raised for a non-existent time dimension."""
    X = make_dataarray("tall-and-skinny")
    pod = POD(n_modes=5, time_dimension="non_existent_dim")

    with pytest.raises(ValueError, match="is not a dimension of the input array"):
        pod.fit(X)


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_n_modes_exceeds_max_rank_error(matrix_type):
    """Test that a ValueError is raised when n_modes is at least
    min(n_spatial_points, n_snapshots), the maximum possible rank."""
    X = make_dataarray(matrix_type)
    n_space, n_time = X.shape
    max_components = min(n_space, n_time)

    pod = POD(n_modes=max_components)
    with pytest.raises(ValueError, match="min\\(n_spatial_points, n_snapshots\\)"):
        pod.fit(X)

    # One less than the max rank should be accepted.
    pod = POD(n_modes=max_components - 1)
    pod.fit(X)


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_transform(matrix_type):
    """Test the transform method projects data onto POD modes correctly."""
    X = make_dataarray(matrix_type)
    n_modes = N_MODES
    n_time = X.sizes["time"]
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    # Transform projects onto modes: modes.T @ X -> shape (n_modes, n_snapshots)
    X_t = pod.transform(X)
    assert isinstance(
        X_t, xr.DataArray
    ), "Transformed data should be an xarray DataArray."
    assert isinstance(
        X_t.data, np.ndarray
    ), "Transformed data should be backed by a numpy array."
    assert X_t.shape == (n_modes, n_time), (
        f"Transformed data should have shape ({n_modes}, {n_time}), "
        f"but got {X_t.shape}."
    )
    # For training data, transform should match fitted time coefficients
    assert np.allclose(X_t.data, pod.time_coeffs.data, atol=1e-6)


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_transform_lazy(matrix_type):
    """Test the transform method with compute=False returns lazy Dask array."""
    X = make_dataarray(matrix_type)
    n_modes = N_MODES
    n_time = X.sizes["time"]
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    # Even though modes are NumPy-backed, compute=False should return Dask-backed
    X_t_lazy = pod.transform(X, compute=False)
    assert isinstance(X_t_lazy, xr.DataArray)
    assert isinstance(X_t_lazy.data, da.Array), (
        "Lazy transform should return Dask-backed DataArray, got "
        f"{type(X_t_lazy.data)}"
    )
    assert X_t_lazy.shape == (n_modes, n_time), (
        f"Transformed data should have shape ({n_modes}, {n_time}), "
        f"but got {X_t_lazy.shape}."
    )

    # Verify computation yields same result as eager transform
    result_lazy = X_t_lazy.compute()
    result_eager = pod.transform(X, compute=True)
    assert np.allclose(result_lazy, result_eager.data, atol=1e-6)


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_reconstruct(matrix_type):
    """Test the inherited reconstruct method on a POD model."""
    X = make_dataarray(matrix_type)
    n_modes = N_MODES
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    X_r = pod.reconstruct(0)
    assert isinstance(
        X_r, xr.DataArray
    ), f"Reconstructed snapshot should be an xarray DataArray, got {type(X_r)}."
    assert isinstance(X_r.data, np.ndarray), (
        "Reconstructed snapshot should have numpy ndarray as data, "
        f"got {type(X_r.data)}."
    )
    assert X_r.shape == (pod.modes.shape[0],), (
        f"Reconstructed snapshot should have shape ({pod.modes.shape[0]},), "
        f"got {X_r.shape}."
    )
    assert (
        "space" in X_r.dims
    ), "Reconstructed snapshot should have the spatial dimension."


@pytest.mark.parametrize("remove_mean", [True, False])
def test_reconstruct_original_scale(remove_mean):
    """Test that reconstruct() undoes the mean removal and 1/sqrt(N)
    scaling applied during fit(), recovering the original-scale data.

    Uses a rank-deficient input so that keeping n_modes = n_time - 1
    (the maximum allowed by TruncatedSVD) captures the full rank of the
    (possibly mean-removed) data, isolating the effect of the scale/mean
    postprocessing from ordinary truncation error.
    """
    rng = np.random.default_rng(0)
    n_space, n_time, rank = 20, 15, 10
    A = rng.standard_normal((n_space, rank))
    B = rng.standard_normal((rank, n_time))
    X_np = (A @ B).astype("float64")
    X = xr.DataArray(
        da.from_array(X_np, chunks=(n_space, n_time)),
        dims=["space", "time"],
        coords={"time": np.arange(n_time)},
    )

    pod = POD(n_modes=n_time - 1, remove_mean=remove_mean)
    pod.fit(X)

    for snapshot in [0, n_time // 2, n_time - 1]:
        reconstructed = pod.reconstruct(snapshot, snapshot_dim="time")
        assert np.allclose(reconstructed.values, X_np[:, snapshot], atol=1e-8), (
            f"Reconstructed snapshot {snapshot} should match the original-scale "
            f"data at (near-)full rank, with remove_mean={remove_mean}."
        )


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


# ──────────────────────────────────────────────────────────────
#  Extended POD tests
# ──────────────────────────────────────────────────────────────


def _make_secondary_dataarray(
    n_space_c: int,
    n_time: int,
    time_dim_pos: int = 1,
    space_name: str = "space_c",
) -> xr.DataArray:
    """Create a Dask-backed DataArray representing a simultaneously
    measured secondary quantity (e.g. temperature)."""
    if time_dim_pos == 1:
        shape = (n_space_c, n_time)
        chunks = (n_space_c, -1)
        dims = [space_name, "time"]
        coords = {space_name: np.arange(n_space_c), "time": np.arange(n_time)}
    else:
        shape = (n_time, n_space_c)
        chunks = (-1, n_space_c)
        dims = ["time", space_name]
        coords = {"time": np.arange(n_time), space_name: np.arange(n_space_c)}
    data = da.random.random(shape, chunks=chunks).astype("float32")
    return xr.DataArray(data, dims=dims, coords=coords)


def test_extended_pod_basic_shape():
    """Test that extended_pod returns the correct shape and type."""
    X = make_dataarray("tall-and-skinny")
    n_time = X.sizes["time"]
    n_modes = N_MODES
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    n_space_c = 500
    C = _make_secondary_dataarray(n_space_c, n_time)
    chi = pod.extended_pod(C)

    assert isinstance(chi, xr.DataArray), f"Expected xr.DataArray, got {type(chi)}."
    assert isinstance(
        chi.data, np.ndarray
    ), f"Expected numpy-backed DataArray, got {type(chi.data)}."
    assert chi.shape == (
        n_space_c,
        n_modes,
    ), f"Expected shape ({n_space_c}, {n_modes}), got {chi.shape}."
    assert chi.dims == (
        "space_c",
        "components",
    ), f"Expected dims ('space_c', 'components'), got {chi.dims}."
    assert (
        "space_c" in chi.coords
    ), "Expected 'space_c' in coords of extended POD modes."
    assert (
        "components" in chi.coords
    ), "Expected 'components' in coords of extended POD modes."


def test_extended_pod_same_spatial_dim():
    """Test extended_pod when the secondary field has the same spatial
    dimension name as the primary field."""
    X = make_dataarray("tall-and-skinny")
    n_time = X.sizes["time"]
    n_modes = N_MODES
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    n_space_c = 200
    C = _make_secondary_dataarray(n_space_c, n_time, space_name="space")
    chi = pod.extended_pod(C)

    assert chi.shape == (
        n_space_c,
        n_modes,
    ), f"Expected shape ({n_space_c}, {n_modes}), got {chi.shape}."
    assert chi.dims == (
        "space",
        "components",
    ), f"Expected dims ('space', 'components'), got {chi.dims}."


def test_extended_pod_transposed_input():
    """Test that extended_pod correctly handles an input array where
    the time dimension is along the rows."""
    X = make_dataarray("tall-and-skinny")
    n_time = X.sizes["time"]
    n_modes = N_MODES
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    n_space_c = 300
    C = _make_secondary_dataarray(n_space_c, n_time, time_dim_pos=0)
    chi = pod.extended_pod(C)

    assert chi.shape == (
        n_space_c,
        n_modes,
    ), f"Expected shape ({n_space_c}, {n_modes}), got {chi.shape}."
    assert chi.dims == (
        "space_c",
        "components",
    ), f"Expected dims ('space_c', 'components'), got {chi.dims}."


def test_extended_pod_lazy():
    """Test extended_pod with compute=False returns a lazy Dask-backed array."""
    X = make_dataarray("tall-and-skinny")
    n_time = X.sizes["time"]
    n_modes = N_MODES
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    n_space_c = 400
    C = _make_secondary_dataarray(n_space_c, n_time)
    chi_lazy = pod.extended_pod(C, compute=False)

    assert isinstance(
        chi_lazy, xr.DataArray
    ), f"Expected xr.DataArray, got {type(chi_lazy)}."
    assert isinstance(
        chi_lazy.data, da.Array
    ), f"Expected Dask-backed DataArray, got {type(chi_lazy.data)}."
    assert chi_lazy.shape == (
        n_space_c,
        n_modes,
    ), f"Expected shape ({n_space_c}, {n_modes}), got {chi_lazy.shape}."

    chi_eager = pod.extended_pod(C, compute=True)
    assert np.allclose(
        chi_lazy.compute().data, chi_eager.data, atol=1e-5
    ), "Lazy and eager extended POD results should match."


def test_extended_pod_formula():
    """Test the extended POD formula against a direct NumPy reference
    implementation following Boree (2003)."""
    np.random.seed(42)
    n_space = 1000
    n_time = 100
    n_modes = 5

    X_np = np.random.randn(n_space, n_time).astype("float32")
    X = xr.DataArray(
        da.from_array(X_np, chunks=(n_space, -1)),
        dims=["space", "time"],
        coords={"space": np.arange(n_space), "time": np.arange(n_time)},
    )

    pod = POD(n_modes=n_modes)
    pod.fit(X)

    n_space_c = 600
    C_np = np.random.randn(n_space_c, n_time).astype("float32")
    C = xr.DataArray(
        da.from_array(C_np, chunks=(n_space_c, -1)),
        dims=["space_c", "time"],
        coords={"space_c": np.arange(n_space_c), "time": np.arange(n_time)},
    )

    chi = pod.extended_pod(C)

    # Reference: chi_j = (1/(lambda_j * N)) * sum_i(a_ij * c_i')
    C_fluc = C_np - C_np.mean(axis=1, keepdims=True)
    # Reconstruct the actual (unscaled) time coefficients
    time_coeffs_stored = pod.time_coeffs.data  # (n_modes, n_time)
    scale_factor = pod._scale_factor
    a_actual = scale_factor * time_coeffs_stored  # (n_modes, n_time)
    energy = pod.energy  # lambda_j = sigma_j^2

    chi_ref = np.zeros((n_space_c, n_modes), dtype="float64")
    for j in range(n_modes):
        chi_ref[:, j] = (C_fluc @ a_actual[j, :]) / (energy[j] * n_time)

    assert np.allclose(
        chi.data, chi_ref, atol=1e-4
    ), "Extended POD modes do not match the reference implementation."


def test_extended_pod_not_fitted_error():
    """Test that calling extended_pod before fit raises RuntimeError."""
    pod = POD(n_modes=5)
    C = _make_secondary_dataarray(100, 50)
    with pytest.raises(RuntimeError, match="not fitted yet"):
        pod.extended_pod(C)


def test_extended_pod_wrong_time_dim_error():
    """Test that extended_pod raises ValueError when time dim is missing."""
    X = make_dataarray("tall-and-skinny")
    pod = POD(n_modes=N_MODES)
    pod.fit(X)

    C = xr.DataArray(
        da.random.random((100, 50), chunks=(100, -1)),
        dims=["space_c", "wrong_time"],
    )
    with pytest.raises(ValueError, match="is not a dimension of the input array"):
        pod.extended_pod(C)


def test_extended_pod_snapshot_mismatch_error():
    """Test that extended_pod raises ValueError for mismatched snapshot count."""
    X = make_dataarray("tall-and-skinny")
    n_time = X.sizes["time"]
    pod = POD(n_modes=N_MODES)
    pod.fit(X)

    C = _make_secondary_dataarray(200, n_time + 10)
    with pytest.raises(ValueError, match="Number of snapshots"):
        pod.extended_pod(C)


def test_extended_pod_time_coord_mismatch_error():
    """Test that extended_pod raises ValueError when time coordinates
    do not match (non-simultaneous measurement)."""
    X = make_dataarray("tall-and-skinny")
    n_time = X.sizes["time"]
    pod = POD(n_modes=N_MODES)
    pod.fit(X)

    # Create C with same number of snapshots but different time coords
    n_space_c = 200
    C = xr.DataArray(
        da.random.random((n_space_c, n_time), chunks=(n_space_c, -1)),
        dims=["space_c", "time"],
        coords={
            "space_c": np.arange(n_space_c),
            "time": np.arange(n_time) + 1000,
        },
    )
    with pytest.raises(ValueError, match="measured simultaneously"):
        pod.extended_pod(C)


def test_extended_pod_remove_mean_false():
    """Test extended_pod with remove_mean=False skips mean removal."""
    X = make_dataarray("tall-and-skinny")
    n_time = X.sizes["time"]
    n_modes = N_MODES
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    n_space_c = 300
    C = _make_secondary_dataarray(n_space_c, n_time)
    C_fluc = C - C.mean(dim="time")

    chi_with_mean_removal = pod.extended_pod(C, remove_mean=True)
    chi_pre_removed = pod.extended_pod(C_fluc, remove_mean=False)

    assert np.allclose(
        chi_with_mean_removal.data, chi_pre_removed.data, atol=1e-5
    ), "Results with remove_mean=True and pre-removed mean should match."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_extended_pod_different_matrix_types(matrix_type):
    """Test extended_pod works for different matrix geometries."""
    X = make_dataarray(matrix_type)
    n_time = X.sizes["time"]
    n_modes = N_MODES
    pod = POD(n_modes=n_modes)
    pod.fit(X)

    n_space_c = 250
    C = _make_secondary_dataarray(n_space_c, n_time)
    chi = pod.extended_pod(C)

    assert chi.shape == (
        n_space_c,
        n_modes,
    ), f"Expected shape ({n_space_c}, {n_modes}), got {chi.shape}."
    assert isinstance(
        chi.data, np.ndarray
    ), f"Expected numpy-backed DataArray, got {type(chi.data)}."
