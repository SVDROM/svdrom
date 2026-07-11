import dask.array as da
import numpy as np
import pytest
import xarray as xr

import svdrom.config as config
from svdrom.preprocessing import hankel_preprocessing
from svdrom.svd import TruncatedSVD

samples_coord_name = "samples"
time_coord_name = "time"
hankel_coord_name = config.get("hankel_coord_name")


def make_dataarray(matrix_type: str) -> xr.DataArray:
    """Make a Dask-backed DataArray with random data of
    specified matrix type. The matrix type can be one of:
    - "tall-and-skinny": More samples than features.
    - "short-and-fat": More features than samples.
    - "square": Equal number of samples and features.

    Chunks are set to test that the TruncatedSVD can handle
    them correctly.
    """
    if matrix_type == "tall-and-skinny":
        n_samples = 10_000
        n_features = 100
        X = da.random.random(
            (n_samples, n_features), chunks=(-1, int(n_features / 2))
        ).astype("float32")
    elif matrix_type == "short-and-fat":
        n_samples = 100
        n_features = 10_000
        X = da.random.random(
            (n_samples, n_features), chunks=(int(n_samples / 2), -1)
        ).astype("float32")
    elif matrix_type == "square":
        n_samples = 1_000
        n_features = 1_000
        X = da.random.random(
            (n_samples, n_features), chunks=(int(n_samples / 2), int(n_features / 2))
        ).astype("float32")
    else:
        msg = (
            "Matrix type not supported. "
            "Must be one of: tall-and-skinny, short-and-fat, square."
        )
        raise ValueError(msg)
    coords = {
        samples_coord_name: np.arange(n_samples),
        time_coord_name: np.arange(n_features),
    }
    dims = list(coords.keys())
    return xr.DataArray(X, dims=dims, coords=coords)


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_basic(algorithm):
    """Test basic functionality of TruncatedSVD."""
    n_components = 10
    tsvd = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
        compute_var_ratio=True,
    )
    expected_attrs = (
        "u",
        "s",
        "v",
        "explained_var_ratio",
    )
    for attr in expected_attrs:
        assert hasattr(tsvd, attr), f"TruncatedSVD should have attribute '{attr}'."

    X = make_dataarray("tall-and-skinny")
    tsvd.fit(X)
    assert isinstance(
        tsvd.u, xr.DataArray
    ), f"u should be an xarray DataArray, got {type(tsvd.u)}."
    assert isinstance(tsvd.u.data, np.ndarray), (
        "u should be a xarray DataArray with numpy ndarray data, "
        f"got {type(tsvd.u.data)}."
    )
    assert isinstance(
        tsvd.v, xr.DataArray
    ), f"v should be an xarray DataArray, got {type(tsvd.v)}."
    assert isinstance(tsvd.v.data, np.ndarray), (
        "v should be a xarray DataArray with numpy ndarray data, "
        f"got {type(tsvd.v.data)}."
    )
    assert isinstance(
        tsvd.s, np.ndarray
    ), f"s should be a numpy ndarray, got {type(tsvd.s)}."
    assert isinstance(tsvd.explained_var_ratio, np.ndarray), (
        "explained_var_ratio should be a numpy ndarray, "
        f"got {type(tsvd.explained_var_ratio)}."
    )
    assert np.all(
        tsvd.explained_var_ratio > 0
    ), "explained_var_ratio should contain values greater than 0."
    assert np.all(
        tsvd.explained_var_ratio < 1
    ), "explained_var_ratio should contain values less than 1."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat", "square"])
@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_matrix_types(matrix_type, algorithm):
    """Test TruncatedSVD with different matrix shapes."""
    X = make_dataarray(matrix_type)
    n_samples, n_features = X.shape
    n_components = 10
    tsvd = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
    )
    tsvd.fit(X)
    X_dims = list(X.dims)
    X_coords = list(X.coords)
    assert tsvd.u.shape == (
        X.shape[0],
        n_components,
    ), f"Shape of u should be ({n_samples}, {n_components}), got {tsvd.u.shape}."
    assert tsvd.v.shape == (
        n_components,
        X.shape[1],
    ), f"Shape of v should be ({n_components}, {n_features}), got {tsvd.v.shape}."
    assert tsvd.s.shape == (
        n_components,
    ), f"Shape of s should be ({n_components},), got {tsvd.s.shape}."
    assert tsvd.explained_var_ratio.shape == (n_components,), (
        f"Shape of explained_var_ratio should be ({n_components},), "
        f"got {tsvd.explained_var_ratio.shape}."
    )
    u_dims = tuple(tsvd.u.dims)
    u_coords = tuple(tsvd.u.coords)
    v_dims = tuple(tsvd.v.dims)
    v_coords = tuple(tsvd.v.coords)
    assert u_dims == (
        X_dims[0],
        "components",
    ), f"u should have dimensions ({X_dims[0]}, 'components'), got {u_dims}."
    assert all(
        u_coord in X_coords for u_coord in u_coords if u_coord != "components"
    ), f"u should have all coordinates from X except 'components', got {u_coords}."
    assert "components" in u_coords, "u should have 'components' coordinate."
    assert v_dims == (
        "components",
        X_dims[1],
    ), f"v should have dimensions ('components', {X_dims[1]}), got {v_dims}."
    assert all(
        v_coord in X_coords for v_coord in v_coords if v_coord != "components"
    ), f"v should have all coordinates from X except 'components', got {v_coords}."
    assert "components" in v_coords, "v should have 'components' coordinate."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat", "square"])
@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_n_components_exceeds_rank(matrix_type, algorithm):
    """n_components must be below the maximum theoretical rank,
    min(n_samples, n_features). Requesting n_components equal to the
    smaller dimension of a short-and-fat matrix used to mislabel the
    right singular vectors, so it must be rejected.
    """
    X = make_dataarray(matrix_type)
    max_rank = min(X.shape)
    tsvd = TruncatedSVD(n_components=max_rank, algorithm=algorithm)
    with pytest.raises(ValueError, match="maximum theoretical rank"):
        tsvd.fit(X)


def test_singular_vectors_labeled_by_kind():
    """The caller tells `_singular_vectors_to_dataarray` whether the
    vectors are left (`u`) or right (`v`), so an ambiguously-shaped
    (square) input is labeled correctly regardless of its shape.
    """
    n = 5
    X = xr.DataArray(
        da.random.random((n, n)),
        dims=[samples_coord_name, time_coord_name],
        coords={
            samples_coord_name: np.arange(n),
            time_coord_name: np.arange(n),
        },
    )
    tsvd = TruncatedSVD(n_components=2)
    vectors = np.random.default_rng().random((n, n)).astype("float32")

    u_da = tsvd._singular_vectors_to_dataarray(vectors, X, kind="u")
    assert u_da.name == "svd_u"
    assert tuple(u_da.dims) == (samples_coord_name, "components")

    v_da = tsvd._singular_vectors_to_dataarray(vectors, X, kind="v")
    assert v_da.name == "svd_v"
    assert tuple(v_da.dims) == ("components", time_coord_name)

    with pytest.raises(ValueError, match="Must be either 'u' or 'v'"):
        tsvd._singular_vectors_to_dataarray(vectors, X, kind="invalid")


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_orthogonality(algorithm):
    """Test orthogonality of u and v matrices."""
    X = make_dataarray("tall-and-skinny")
    n_components = 10
    tsvd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
    tsvd.fit(X)

    identity_k = np.eye(tsvd.n_components, dtype=np.float32)
    u, v = tsvd.u.data, tsvd.v.data
    u_ortho = u.T @ u
    v_ortho = v @ v.T

    assert np.allclose(
        u_ortho, identity_k, atol=1e-5
    ), "u.T @ u is not close to identity."
    assert np.allclose(
        v_ortho, identity_k, atol=1e-5
    ), "v @ v.T is not close to identity."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_transform(matrix_type):
    """Test the transform method of TruncatedSVD."""
    X = make_dataarray(matrix_type)
    n_components = 10
    tsvd = TruncatedSVD(n_components=n_components)
    tsvd.fit(X)

    X_t = tsvd.transform(X)
    assert isinstance(
        X_t, xr.DataArray
    ), "Transformed data should be an xarray DataArray."
    assert isinstance(
        X_t.data, np.ndarray
    ), f"Transformed data should have numpy ndarray as data, got {type(X_t.data)}."
    assert X_t.shape == (X.shape[0], n_components), (
        f"Transformed data should have shape ({X.shape[0]}, {n_components}), "
        f"but got {X_t.shape}."
    )
    assert np.allclose(
        tsvd.u * tsvd.s, X_t, atol=1e-5
    ), "Transformed data does not match u * s."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_reconstruct(matrix_type):
    """Test the reconstruct method of TruncatedSVD."""
    X = make_dataarray(matrix_type)
    n_components = 10
    tsvd = TruncatedSVD(n_components=n_components)
    tsvd.fit(X)

    X_r = tsvd.reconstruct(0)
    assert isinstance(
        X_r, xr.DataArray
    ), f"Reconstructed snapshot should be an xarray DataArray, got {type(X_r)}."
    assert isinstance(X_r.data, np.ndarray), (
        "Reconstructed snapshot should have numpy ndarray as data, "
        f"got {type(X_r.data)}."
    )
    assert X_r.shape == (
        tsvd.u.shape[0],
    ), f"Reconstructed snapshot should have shape ({tsvd.u.shape[0]}), got {X_r.shape}."
    assert (
        samples_coord_name in X_r.dims
    ), f"Reconstructed snapshot should have dimension {samples_coord_name}."


def test_svd_hankel():
    """Test that SVD can be performed on a matrix after
    Hankel pre-processing."""
    X = make_dataarray("tall-and-skinny")
    d = 2
    X_d = hankel_preprocessing(X, d=d)
    n_components = 10
    tsvd = TruncatedSVD(n_components=n_components)
    tsvd.fit(X_d)

    assert tsvd.u.shape == (X_d.shape[0], n_components), (
        "Expected the shape of the left singular vectors to be "
        f"{(X_d.shape[0], n_components)}, but got {tsvd.u.shape}."
    )
    assert tsvd.v.shape == (n_components, X_d.shape[1]), (
        "Expected the shape of the right singular vectors to be "
        f"{(n_components, X_d.shape[1])}, but got {tsvd.v.shape}."
    )
    assert len(tsvd.s) == n_components, (
        "Expected the length of the singular values to be "
        f"{n_components}, but got {len(tsvd.s)}."
    )
    assert hankel_coord_name in tsvd.u.coords, (
        "Expected the left singular vectors DataArray to contain "
        f"a coordinate called {hankel_coord_name}."
    )
    assert np.array_equal(
        tsvd.u[hankel_coord_name].values,
        X_d[hankel_coord_name].values,
    ), (
        f"Expected the {hankel_coord_name} coordinate of the left singular vectors "
        f"to match the {hankel_coord_name} coordinate of the Hankel-preprocessed "
        "data matrix."
    )
    assert (
        tsvd.v.attrs[config.get("hankel_time_mapping_attr")]
        == X_d.attrs[config.get("hankel_time_mapping_attr")]
    ), (
        f"The {config.get('hankel_time_mapping_attr')} attribute in the right "
        "singular vectors and Hankel-preprocessed data matrix should match."
    )
