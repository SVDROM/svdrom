"""Additional tests targeting error handling and less-common code paths
of the OptDMD class, to raise coverage of the dmd module above 80%."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from make_test_data import DataGenerator

import svdrom.config as config
from svdrom import OptDMD

dask.config.set(scheduler="single-threaded")


@pytest.fixture()
def svd_results():
    """Valid (u, s, v) SVD results with a datetime time vector."""
    generator = DataGenerator(seed=1234)
    generator.generate_svd_results(n_components=6)
    return generator.u, generator.s, generator.v


@pytest.fixture()
def fitted_optdmd(svd_results):
    """A fitted OptDMD instance (no bagging, no Hankel pre-processing)."""
    u, s, v = svd_results
    solver = OptDMD()
    solver.fit(u, s, v, varpro_opts_dict={"maxiter": 15})
    return solver


@pytest.fixture()
def fitted_optdmd_bagging(svd_results):
    """A fitted OptDMD instance with bagging enabled."""
    u, s, v = svd_results
    solver = OptDMD(num_trials=5, seed=1234)
    solver.fit(u, s, v, varpro_opts_dict={"maxiter": 15})
    return solver


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_modes": 0}, "'n_modes'"),
        ({"n_modes": -2}, "'n_modes'"),
        ({"time_units": "d"}, "'time_units'"),
        ({"input_time_units": "d"}, "'input_time_units'"),
        ({"num_trials": -1}, "'num_trials'"),
        ({"trial_size": 0}, "'trial_size'"),
        ({"trial_size": 1.5}, "'trial_size'"),
        ({"trial_size": -1.0}, "'trial_size'"),
    ],
)
def test_init_validation(kwargs, match):
    """The OptDMD constructor rejects invalid arguments."""
    with pytest.raises(ValueError, match=match):
        OptDMD(**kwargs)


# ---------------------------------------------------------------------------
# _check_svd_inputs
# ---------------------------------------------------------------------------


def test_check_inputs_u_not_computed(svd_results):
    """fit() raises when the left singular vectors are not numpy-backed."""
    u, s, v = svd_results
    u = u.copy(data=da.from_array(u.data))
    with pytest.raises(ValueError, match="left singular vectors"):
        OptDMD().fit(u, s, v)


def test_check_inputs_v_not_computed(svd_results):
    """fit() raises when the right singular vectors are not numpy-backed."""
    u, s, v = svd_results
    v = v.copy(data=da.from_array(v.data))
    with pytest.raises(ValueError, match="right singular vectors"):
        OptDMD().fit(u, s, v)


def test_check_inputs_mismatched_components(svd_results):
    """fit() raises when u, s and v have different numbers of components."""
    u, s, v = svd_results
    with pytest.raises(ValueError, match="same number of components"):
        OptDMD().fit(u, s[:-1], v)


def test_check_inputs_missing_time_dim(svd_results):
    """fit() raises when the time dimension is absent from v."""
    u, s, v = svd_results
    with pytest.raises(ValueError, match="not "):
        OptDMD(time_dimension="not_a_dim").fit(u, s, v)


def test_check_inputs_unsorted_time(svd_results):
    """fit() raises when the time dimension of v is not sorted."""
    u, s, v = svd_results
    v = v.isel(time=slice(None, None, -1))
    with pytest.raises(ValueError, match="not\n?.*sorted|sorted"):
        OptDMD().fit(u, s, v)


def _add_hankel_coord(u, values):
    """Attach a Hankel lag coordinate to the left singular vectors."""
    coord_name = config.get("hankel_coord_name")
    return u.assign_coords({coord_name: ("samples", values)})


def test_check_inputs_hankel_coord_wrong_dim(svd_results):
    """fit() raises when the Hankel coordinate is on the wrong dimension."""
    u, s, v = svd_results
    coord_name = config.get("hankel_coord_name")
    u = u.assign_coords({coord_name: ("components", np.zeros(u.sizes["components"]))})
    with pytest.raises(ValueError, match="same as the first dimension"):
        OptDMD().fit(u, s, v)


def test_check_inputs_hankel_coord_negative(svd_results):
    """fit() raises when the Hankel coordinate has negative values."""
    u, s, v = svd_results
    values = np.zeros(u.sizes["samples"])
    values[0] = -1
    u = _add_hankel_coord(u, values)
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        OptDMD().fit(u, s, v)


def test_check_inputs_hankel_missing_time_mapping(svd_results):
    """fit() raises when the Hankel coordinate is present but v lacks the
    Hankel time mapping attribute."""
    u, s, v = svd_results
    u = _add_hankel_coord(u, np.zeros(u.sizes["samples"]))
    with pytest.raises(ValueError, match="attribute"):
        OptDMD().fit(u, s, v)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_too_many_modes(svd_results):
    """fit() raises when more modes are requested than SVD components."""
    u, s, v = svd_results
    with pytest.raises(ValueError, match="exceeds the number"):
        OptDMD(n_modes=len(s) + 1).fit(u, s, v)


def test_fit_error_wrapped(svd_results, monkeypatch):
    """An error inside the BOPDMD fit is wrapped in a RuntimeError."""
    u, s, v = svd_results

    def _boom(*args, **kwargs):
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr("svdrom.dmd.BOPDMD.fit_econ", _boom)
    with pytest.raises(RuntimeError, match="Error computing the DMD fit"):
        OptDMD().fit(u, s, v)


# ---------------------------------------------------------------------------
# properties on a fitted model
# ---------------------------------------------------------------------------


def test_modes_averaged_without_hankel(fitted_optdmd):
    """modes_averaged returns the raw modes when no Hankel embedding is used."""
    xr.testing.assert_identical(fitted_optdmd.modes_averaged, fitted_optdmd.modes)


def test_modes_std_averaged_without_hankel(fitted_optdmd_bagging):
    """modes_std_averaged returns the raw modes_std without Hankel embedding."""
    xr.testing.assert_identical(
        fitted_optdmd_bagging.modes_std_averaged, fitted_optdmd_bagging.modes_std
    )


def test_modes_averaged_none_when_unfitted():
    """The averaged-mode properties are None before fitting."""
    solver = OptDMD()
    assert solver.modes_averaged is None
    assert solver.modes_std_averaged is None


def test_average_modes_across_lags_not_divisible(fitted_optdmd):
    """_average_modes_across_lags raises when rows aren't divisible by hankel_d."""
    modes = fitted_optdmd.modes
    with pytest.raises(ValueError, match="must be divisible"):
        OptDMD._average_modes_across_lags(modes, hankel_d=modes.shape[0] + 1)


# ---------------------------------------------------------------------------
# _generate_forecast_time_vector
# ---------------------------------------------------------------------------


def test_forecast_time_vector_uninitialized():
    """_generate_forecast_time_vector raises when the fit time vector is missing."""
    solver = OptDMD()
    with pytest.raises(ValueError, match="not initialized"):
        solver._generate_forecast_time_vector("10 s")


def test_forecast_span_bad_string(fitted_optdmd):
    """A malformed forecast_span string raises a ValueError."""
    with pytest.raises(ValueError, match="value units"):
        fitted_optdmd._generate_forecast_time_vector("10")


def test_forecast_dt_bad_string(fitted_optdmd):
    """A malformed dt string raises a ValueError."""
    with pytest.raises(ValueError, match="value units"):
        fitted_optdmd._generate_forecast_time_vector("10 s", dt="1")


def test_forecast_dt_non_positive(fitted_optdmd):
    """A non-positive integer dt raises a ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        fitted_optdmd._generate_forecast_time_vector("10 s", dt=0)


def test_forecast_time_step_too_small(fitted_optdmd):
    """A forecast time step that rounds to zero raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="not valid"):
        fitted_optdmd._generate_forecast_time_vector("1 s", dt="1 ms")


# ---------------------------------------------------------------------------
# forecast / reconstruct error wrapping and unfitted state
# ---------------------------------------------------------------------------


def test_forecast_time_vector_error_wrapped(fitted_optdmd):
    """forecast() wraps time-vector generation errors in a RuntimeError."""
    with pytest.raises(RuntimeError, match="forecast time vector"):
        fitted_optdmd.forecast("bad_span")


def test_estimate_array_size_unfitted():
    """_estimate_array_size raises before the model is fitted."""
    solver = OptDMD()
    with pytest.raises(RuntimeError, match="not available"):
        solver._estimate_array_size(np.arange(5.0))


def test_predict_unfitted():
    """_predict raises before the model is fitted."""
    solver = OptDMD()
    with pytest.raises(RuntimeError, match="not available"):
        solver._predict(np.arange(5.0))


def test_prediction_to_dataarray_unfitted():
    """_prediction_to_dataarray raises before the model is fitted."""
    solver = OptDMD()
    with pytest.raises(RuntimeError, match="modes have not been computed"):
        solver._prediction_to_dataarray(np.zeros((2, 2)), np.arange(2.0))


def test_rechunk_unfitted():
    """_rechunk_along_columns raises before the model is fitted."""
    solver = OptDMD()
    with pytest.raises(RuntimeError, match="not been fitted"):
        solver._rechunk_along_columns(da.zeros((4, 4)))


# ---------------------------------------------------------------------------
# label-based reconstruction paths
# ---------------------------------------------------------------------------


def test_reconstruct_label_slice(fitted_optdmd):
    """reconstruct() supports a label-based (datetime string) slice."""
    time_values = fitted_optdmd.time_fit
    start = np.datetime64(time_values[0], "s").astype(str)
    stop = np.datetime64(time_values[3], "s").astype(str)
    reconstruction = fitted_optdmd.reconstruct(slice(start, stop))
    assert isinstance(reconstruction, xr.DataArray)
    assert reconstruction.sizes[fitted_optdmd.time_dimension] == 4


def test_reconstruct_label_single(fitted_optdmd):
    """reconstruct() supports a single datetime-string label."""
    label = np.datetime64(fitted_optdmd.time_fit[2], "s").astype(str)
    reconstruction = fitted_optdmd.reconstruct(label)
    assert isinstance(reconstruction, xr.DataArray)
    assert reconstruction.sizes[fitted_optdmd.time_dimension] == 1


def test_reconstruct_mixed_slice_raises(fitted_optdmd):
    """reconstruct() raises for a slice mixing indices and labels."""
    with pytest.raises(RuntimeError, match="reconstruction time vector"):
        fitted_optdmd.reconstruct(slice(0, "2020-01-01"))


def test_reconstruct_invalid_type_raises(fitted_optdmd):
    """reconstruct() raises for an unsupported time selector type."""
    with pytest.raises(RuntimeError, match="reconstruction time vector"):
        fitted_optdmd.reconstruct(1.5)
