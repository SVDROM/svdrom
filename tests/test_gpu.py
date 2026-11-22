"""Optional numerical and output-contract checks on a real CUDA device."""

from __future__ import annotations

from collections.abc import Iterator

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

import svdrom.config as config
from svdrom import POD, TruncatedSVD
from svdrom._backend import get_xp, to_device, to_host
from svdrom.preprocessing import StandardScaler

cp = pytest.importorskip("cupy")


@pytest.fixture(autouse=True)
def _cuda_device() -> Iterator[None]:
    """Skip installed-but-unusable CuPy and restore configuration after each test."""
    try:
        available = cp.is_available()
    except Exception as exc:
        pytest.skip(f"CUDA is unavailable: {exc}")
    if not available:
        pytest.skip("CuPy is installed but no CUDA device is available.")
    original = dict(config._editable_config)
    with dask.config.set({"array.backend": "numpy", "scheduler": "synchronous"}):
        yield
    config._editable_config.clear()
    config._editable_config.update(original)
    cp.get_default_memory_pool().free_all_blocks()


def weather_matrix(dtype: str) -> xr.DataArray:
    """Build a deterministic low-rank field independent of either backend RNG."""
    rng = np.random.default_rng(42)
    values = (rng.normal(size=(64, 3)) @ rng.normal(size=(3, 24))).astype(dtype)
    return xr.DataArray(
        da.from_array(values, chunks=(16, 6)),
        dims=("space", "time"),
        coords={"space": np.arange(64), "time": np.arange(24)},
        attrs={"units": "K"},
    )


def test_device_and_host_transfers_preserve_laziness() -> None:
    """CPU output chunks are downloaded lazily, even after configuration changes."""
    source = weather_matrix("float32")
    with dask.config.set({"array.backend": "cupy"}):
        assert get_xp() is cp, "Expected the CuPy array module."
        device = to_device(source)
        assert isinstance(device.data._meta, cp.ndarray), "Expected GPU chunk metadata."
        assert (
            to_device(device.data) is device.data
        ), "Expected idempotent GPU transfer."
        host = to_host(device)
        assert isinstance(
            host.data._meta, np.ndarray
        ), "Expected lazy CPU output chunks."
        eager = to_device(np.arange(4))
        assert isinstance(eager, cp.ndarray), "Expected eager GPU conversion."
    assert isinstance(host.compute().data, np.ndarray), "Expected computed CPU output."
    xr.testing.assert_allclose(host, source)
    np.testing.assert_array_equal(
        to_host(eager),
        np.arange(4),
        err_msg="Expected CPU conversion independent of configuration.",
    )


@pytest.mark.parametrize("staged", [None, False, True])
def test_randomized_gpu_staging_default(
    staged: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPU staging is memory-conscious by default and respects an explicit override."""
    original = da.linalg.svd_compressed
    settings = []

    def compressed(*args: object, **kwargs: object) -> tuple:
        settings.append(kwargs.get("compute"))
        return original(*args, **kwargs)

    monkeypatch.setattr(da.linalg, "svd_compressed", compressed)
    options = {} if staged is None else {"compute": staged}
    with dask.config.set({"array.backend": "cupy"}):
        model = TruncatedSVD(
            3, algorithm="randomized", compute_u=False, compute_v=False
        ).fit(weather_matrix("float32"), seed=42, **options)
    assert settings == [
        True if staged is None else staged
    ], "Expected default staging with explicit override support."
    assert isinstance(model.u.data, da.Array), "Expected deferred left vectors."
    assert isinstance(model.v.data, da.Array), "Expected deferred right vectors."
    assert isinstance(
        model.u.compute().data, np.ndarray
    ), "Expected deferred computation to produce CPU output."


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("eager", [False, True])
def test_svd_gpu_matches_cpu(algorithm: str, dtype: str, eager: bool) -> None:
    """GPU fitting agrees numerically while all public factors remain CPU-backed."""
    source = weather_matrix(dtype)
    options = {"seed": 42, "n_power_iter": 2} if algorithm == "randomized" else {}
    cpu = TruncatedSVD(3, algorithm=algorithm, compute_var_ratio=True).fit(
        source, **options
    )
    seen_gpu = []
    with (
        dask.config.set({"array.backend": "cupy"}),
        Callback(
            posttask=lambda _key, value, *_args: seen_gpu.append(
                isinstance(value, cp.ndarray)
            )
        ),
    ):
        gpu = TruncatedSVD(
            3,
            algorithm=algorithm,
            compute_u=eager,
            compute_v=eager,
            compute_var_ratio=eager,
        ).fit(source, **options)
    assert any(seen_gpu), "Expected real CuPy intermediates during decomposition."
    assert isinstance(gpu.s, np.ndarray), "Expected CPU singular values."
    if eager:
        assert isinstance(gpu.u.data, np.ndarray), "Expected eager CPU left vectors."
        assert isinstance(gpu.v.data, np.ndarray), "Expected eager CPU right vectors."
        assert isinstance(
            gpu.explained_var_ratio, np.ndarray
        ), "Expected eager CPU ratios."
    else:
        for result in (gpu.u.data, gpu.v.data, gpu.explained_var_ratio):
            assert isinstance(result, da.Array), "Expected a deferred result."
            assert isinstance(
                result._meta, np.ndarray
            ), "Expected deferred CPU output chunks."
    tolerance = 3e-4 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(
        gpu.s, cpu.s, rtol=tolerance, err_msg="Expected CPU/GPU singular-value parity."
    )
    gpu.compute_u()
    gpu.compute_v()
    gpu.compute_var_ratio()
    assert isinstance(
        gpu.u.data, np.ndarray
    ), "Expected deferred computation to return CPU left vectors."
    assert isinstance(
        gpu.v.data, np.ndarray
    ), "Expected deferred computation to return CPU right vectors."
    assert isinstance(
        gpu.explained_var_ratio, np.ndarray
    ), "Expected deferred computation to return CPU ratios."
    np.testing.assert_allclose(
        gpu.explained_var_ratio,
        cpu.explained_var_ratio,
        rtol=tolerance,
        err_msg="Expected explained-variance parity.",
    )
    np.testing.assert_allclose(
        gpu.reconstruct().data,
        source.compute().data,
        rtol=tolerance,
        atol=tolerance,
        err_msg="Expected the GPU factors to reconstruct the low-rank field.",
    )
    with dask.config.set({"array.backend": "cupy"}):
        projection = gpu.transform(source)
    assert isinstance(projection.data, np.ndarray), "Expected CPU projection output."


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized", "covariance"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("eager", [False, True])
def test_pod_gpu_matches_cpu(algorithm: str, dtype: str, eager: bool) -> None:
    """All POD algorithms preserve CPU outputs, lazy flags and reconstruction."""
    source = weather_matrix(dtype)
    options = {"seed": 42} if algorithm == "randomized" else {}
    cpu = POD(3, svd_algorithm=algorithm, compute_energy_ratio=True).fit(
        source, **options
    )
    with dask.config.set({"array.backend": "cupy"}):
        gpu = POD(
            3,
            svd_algorithm=algorithm,
            compute_modes=eager,
            compute_time_coeffs=eager,
            compute_energy_ratio=eager,
        ).fit(source, **options)
        projection = gpu.transform(source, compute=eager)
        extended = gpu.extended_pod(source, compute=eager)
    for result in (gpu.modes, gpu.time_coeffs, projection, extended):
        expected_type = np.ndarray if eager else da.Array
        assert isinstance(
            result.data, expected_type
        ), "Expected the requested eager/lazy result type."
        assert isinstance(
            result.compute().data, np.ndarray
        ), "Expected CPU data when materialized."
    tolerance = 3e-4 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(
        gpu.energy,
        cpu.energy,
        rtol=tolerance,
        err_msg="Expected CPU/GPU modal-energy parity.",
    )
    gpu.compute_modes()
    gpu.compute_time_coeffs()
    gpu.compute_energy_ratio()
    assert isinstance(
        gpu.explained_energy_ratio, np.ndarray
    ), "Expected CPU modal energy ratios."
    np.testing.assert_allclose(
        gpu.explained_energy_ratio,
        cpu.explained_energy_ratio,
        rtol=tolerance,
        err_msg="Expected CPU/GPU energy-ratio parity.",
    )
    np.testing.assert_allclose(
        gpu.reconstruct().data,
        source.compute().data,
        rtol=tolerance,
        atol=tolerance,
        err_msg="Expected reconstruction in original physical units.",
    )
    np.testing.assert_allclose(
        projection.compute().data,
        gpu.time_coeffs.data,
        rtol=tolerance,
        atol=tolerance,
        err_msg="Expected projection to recover temporal coefficients.",
    )


@pytest.mark.parametrize("as_dataset", [False, True])
@pytest.mark.parametrize("with_std", [False, True])
def test_scaler_gpu_matches_cpu(as_dataset: bool, with_std: bool) -> None:
    """GPU reductions preserve Dataset/DataArray metadata and CPU stored statistics."""
    source = weather_matrix("float64")
    if as_dataset:
        source = xr.Dataset(
            {"air": source, "other": source * 2}, attrs={"source": "test"}
        )
    cpu_scaler = StandardScaler()
    expected = cpu_scaler(source, with_std=with_std)
    gpu_scaler = StandardScaler()
    with dask.config.set({"array.backend": "cupy"}):
        actual = gpu_scaler(source, with_std=with_std)
    xr.testing.assert_allclose(actual, expected)
    xr.testing.assert_allclose(gpu_scaler.mean, cpu_scaler.mean)
    for variable in actual.data_vars.values() if as_dataset else [actual]:
        assert isinstance(variable.data, da.Array), "Expected lazy scaled output."
        assert isinstance(
            variable.data._meta, np.ndarray
        ), "Expected CPU scaled chunks."
    for variable in (
        gpu_scaler.mean.data_vars.values() if as_dataset else [gpu_scaler.mean]
    ):
        assert isinstance(variable.data, np.ndarray), "Expected CPU-stored means."
    if with_std:
        xr.testing.assert_allclose(gpu_scaler.std, cpu_scaler.std)


def test_cupy_config_and_numpy_dmd_handoff() -> None:
    """OptDMD explains its CPU fallback and accepts computed GPU SVD factors."""
    from svdrom import OptDMD

    time = np.linspace(0, 2, 24)
    source = xr.DataArray(
        da.from_array(
            np.column_stack((np.arange(1, 9), np.arange(1, 9) ** 2))
            @ np.stack((np.exp(-time), np.exp(-0.25 * time))),
            chunks=(4, 6),
        ),
        dims=("space", "time"),
        coords={"time": time},
    )
    with dask.config.set({"array.backend": "cupy"}):
        config.set(array_backend="cupy")
        svd = TruncatedSVD(2).fit(source)
        with pytest.warns(UserWarning, match="OptDMD runs on the CPU"):
            model = OptDMD(n_modes=2).fit(
                to_device(svd.u), to_device(svd.s), to_device(svd.v)
            )
    assert isinstance(model.modes.data, np.ndarray), "Expected CPU DMD modes."


@pytest.mark.parametrize("algorithm", ["tsqr", "covariance"])
def test_complex_gpu_pod_uses_hermitian_projection(algorithm: str) -> None:
    """GPU support preserves the newly added complex POD projection behavior."""
    rng = np.random.default_rng(12)
    spatial = rng.normal(size=(12, 3)) + 1j * rng.normal(size=(12, 3))
    temporal = rng.normal(size=(3, 32)) + 1j * rng.normal(size=(3, 32))
    values = spatial @ temporal
    source = xr.DataArray(da.from_array(values, chunks=(6, 8)), dims=("space", "time"))
    with dask.config.set({"array.backend": "cupy"}):
        model = POD(3, svd_algorithm=algorithm, compute_time_coeffs=False).fit(source)
        projected = model.transform(source)
    np.testing.assert_allclose(
        projected.data,
        model.time_coeffs.compute().data,
        atol=1e-10,
        err_msg="Expected complex GPU projection to recover fitted coefficients.",
    )
    np.testing.assert_allclose(
        model.reconstruct().data,
        values,
        atol=1e-10,
        err_msg="Expected complex GPU POD to reconstruct the low-rank field.",
    )
