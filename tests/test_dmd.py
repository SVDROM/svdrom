import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from make_test_data import DataGenerator, SignalGenerator

from svdrom.dmd import OptDMD
from svdrom.preprocessing import hankel_preprocessing
from svdrom.svd import TruncatedSVD

# set the dask scheduler to single-threaded
dask.config.set(scheduler="single-threaded")


class BaseTestOptDMD:
    """Base test class for OptDMD containing all generic test methods."""

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_basic(self, solver):
        """Basic test to check attributes of the OptDMD class."""
        solver = getattr(self, solver)
        assert hasattr(
            solver, "modes"
        ), "OptDMD object is missing the 'modes' attribute."
        assert hasattr(solver, "eigs"), "OptDMD object is missing the 'eigs' attribute."
        assert hasattr(
            solver, "amplitudes"
        ), "OptDMD object is missing the 'amplitudes' attribute."
        assert hasattr(
            solver, "modes_std"
        ), "OptDMD object is missing the 'modes_std' attribute."
        assert hasattr(
            solver, "eigs_std"
        ), "OptDMD object is missing the 'eigs_std' attribute."
        assert hasattr(
            solver, "amplitudes_std"
        ), "OptDMD object is missing the 'amplitudes_std' attribute."
        assert hasattr(
            solver, "time_fit"
        ), "OptDMD object is missing the 'time_fit' attribute."
        assert hasattr(
            solver, "num_trials"
        ), "OptDMD object is missing the 'num_trials' attribute."
        assert hasattr(
            solver, "trial_size"
        ), "OptDMD object is missing the 'trial_size' attribute."
        assert hasattr(
            solver, "parallel_bagging"
        ), "OptDMD object is missing the 'parallel_bagging' attribute."
        assert hasattr(
            solver, "dynamics"
        ), "OptDMD object is missing the 'dynamics' attribute."
        assert hasattr(
            solver, "time_units"
        ), "OptDMD object is missing the 'time_units' attribute."
        assert hasattr(
            solver, "input_time_units"
        ), "OptDMD object is missing the 'input_time_units' attribute."
        assert hasattr(
            solver, "hankel_d"
        ), "OptDMD object is missing the 'hankel_d' attribute."
        assert hasattr(
            solver, "modes_averaged"
        ), "OptDMD object is missing the 'modes_averaged' attribute."
        assert hasattr(
            solver, "modes_std_averaged"
        ), "OptDMD object is missing the 'modes_std_averaged' attribute."

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_fit_basic(self, solver):
        """Test the fit() method of the OptDMD class."""
        solver = getattr(self, solver)
        solver.fit(
            self.u,
            self.s,
            self.v,
            varpro_opts_dict={"maxiter": 15},
            eig_sort="imag",
        )

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_fit_outputs(self, solver):
        """Test data types and shapes of attributes after
        calling the fit() method."""
        solver = getattr(self, solver)
        assert isinstance(solver.modes, xr.DataArray), (
            "Expected 'modes' to be of type 'xr.DataArray', "
            f"but got {type(solver.modes)} instead."
        )
        assert isinstance(solver.eigs, np.ndarray), (
            "Expected 'eigs' to be of type 'np.ndarray', "
            f"but got {type(solver.eigs)} instead."
        )
        assert isinstance(solver.amplitudes, np.ndarray), (
            "Expected 'amplitudes' to be of type 'np.ndarray', "
            f"but got {type(solver.amplitudes)} instead."
        )
        assert isinstance(solver.time_fit, np.ndarray), (
            "Expected 'time_fit' to be of type 'np.ndarray', "
            f"but got {type(solver.time_fit)} instead."
        )
        np.testing.assert_equal(
            solver.time_fit,
            self.v.time.values,
            strict=True,
            err_msg=(
                "Expected 'time_fit' vector to be strictly equal to 'v.time.values'."
            ),
        )
        assert isinstance(solver._t_fit, np.ndarray), (
            "Expected 't_fit' to be of type 'np.ndarray', "
            f"but got {type(solver._t_fit)} instead."
        )
        assert np.issubdtype(solver._t_fit.dtype, float), (
            f"Expected 't_fit' vector to have data type float, "
            f"but got {solver._t_fit.dtype.name}."
        )
        assert solver.modes.shape == self.u.shape, (
            f"Expected 'modes.shape' to be {self.u.shape}, "
            f"but got {solver.modes.shape} instead."
        )
        if self.hankel_preprocessing:
            expected_shape = (self.u.shape[0] // self.d, self.u.shape[1])
            assert solver.modes_averaged.shape == expected_shape, (
                f"For an input dataset with time-delay embedding of {self.d}, "
                f"expected 'modes_averaged.shape' to be {expected_shape}, "
                f"but got {solver.modes.shape} instead."
            )
        assert solver.eigs.shape == (solver.modes.shape[1],), (
            f"Expected 'eigs.shape' to be {(solver.modes.shape[1],)}, "
            f"but got {solver.eigs.shape} instead."
        )
        assert solver.amplitudes.shape == (solver.modes.shape[1],), (
            f"Expected 'amplitudes.shape' to be {(solver.modes.shape[1],)}, "
            f"but got {solver.amplitudes.shape} instead."
        )
        if solver.num_trials == 0:
            # no bagging
            assert solver.modes_std is None, (
                "Expected 'modes_std' to be None, "
                f"but got {solver.modes_std} instead."
            )
            assert solver.eigs_std is None, (
                "Expected 'eigs_std' to be None, " f"but got {solver.eigs_std} instead."
            )
            assert solver.amplitudes_std is None, (
                "Expected 'amplitudes_std' to be None, "
                f"but got {solver.amplitudes_std} instead."
            )
        else:
            # with bagging
            assert isinstance(solver.modes_std, xr.DataArray), (
                "Expected 'modes_std' to be xr.DataArray, "
                f"but got {solver.modes_std} instead."
            )
            assert solver.modes_std.shape == solver.modes.shape, (
                "Expected 'modes_std' and 'modes' to have the same shape, "
                f"but got shapes {solver.modes_std.shape} and {solver.modes.shape}, "
                "respectively."
            )
            if self.hankel_preprocessing:
                expected_shape = (self.u.shape[0] // self.d, self.u.shape[1])
                assert solver.modes_std_averaged.shape == expected_shape, (
                    f"For an input dataset with time-delay embedding of {self.d}, "
                    f"expected 'modes_std_averaged.shape' to be {expected_shape}, "
                    f"but got {solver.modes.shape} instead."
                )
            assert isinstance(solver.eigs_std, np.ndarray), (
                "Expected 'eigs_std' to be np.ndarray, "
                f"but got {solver.eigs_std} instead."
            )
            assert isinstance(solver.amplitudes_std, np.ndarray), (
                "Expected 'amplitudes_std' to be np.ndarray, "
                f"but got {solver.amplitudes_std} instead."
            )

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_dynamics_attr(self, solver):
        """Test the dynamics attribute."""
        solver = getattr(self, solver)
        dynamics = solver.dynamics
        assert isinstance(dynamics, xr.DataArray), (
            "Expected 'dynamics' to be of type 'xr.DataArray', "
            f"but got {type(dynamics)} instead."
        )
        assert isinstance(dynamics.data, np.ndarray), (
            "Expected the 'dynamics' DataArray to be backed by a "
            f"np.ndarray, but got a {type(dynamics.data)} instead."
        )
        assert dynamics.shape == (solver.n_modes, len(solver.time_fit))
        assert dynamics.dims == ("components", "time")

    @pytest.mark.parametrize("forecast_span", ["20 s", 20])
    @pytest.mark.parametrize("dt", ["2 s", 10, None])
    def test_generate_forecast_time_vector(self, forecast_span, dt):
        """Test the method to generate the forecast time vector
        with different inputs for the forecast span and forecast
        time step.
        """
        solver = self.optdmd
        if dt is not None:
            expected_delta_t = 2
            expected_delta_time = np.timedelta64(2, "s")
            expected_len = 10
        else:
            expected_delta_time = np.mean(np.diff(solver._time_fit))
            expected_delta_t = np.mean(np.diff(solver._t_fit))
            expected_len = 20 / expected_delta_t
        t_forecast, time_forecast = solver._generate_forecast_time_vector(
            forecast_span, dt
        )
        assert isinstance(t_forecast, np.ndarray), (
            "Expected 't_forecast' to be of type 'np.ndarray', "
            f"but got {type(t_forecast)} instead."
        )
        assert isinstance(time_forecast, np.ndarray), (
            "Expected 'time_forecast' to be of type 'np.ndarray', "
            f"but got {type(time_forecast)} instead."
        )
        assert np.unique(np.diff(t_forecast)) == expected_delta_t, (
            "Expected the difference between consecutive elements in "
            f"'t_forecast' to be {expected_delta_t}, but got "
            f"{np.unique(np.diff(t_forecast))} instead."
        )
        if np.issubdtype(self.t.dtype, float):
            assert np.unique(np.diff(time_forecast)) == expected_delta_t, (
                "Expected the difference between consecutive elements in "
                f"'time_forecast' to be {expected_delta_t}, but got "
                f"{np.unique(np.diff(time_forecast))} instead."
            )
        else:
            assert np.unique(np.diff(time_forecast)) == expected_delta_time, (
                "Expected the difference between consecutive elements in "
                f"'time_forecast' to be {expected_delta_time}, but got "
                f"{np.unique(np.diff(time_forecast))} instead."
            )
        assert len(t_forecast) == expected_len, (
            f"Expected the length of 't_forecast' to be {expected_len}, but got "
            f"{len(t_forecast)} instead."
        )
        assert len(time_forecast) == expected_len, (
            f"Expected the length of 'time_forecast' to be {expected_len}, but got "
            f"{len(time_forecast)} instead."
        )
        assert t_forecast[0] == solver._t_fit[-1] + expected_delta_t, (
            "Expected 't_forecast[0]' to be ahead of t_fit[-1] "
            f"by a value of {expected_delta_t},"
            f"but got a value of {t_forecast[0] - solver._t_fit[-1]} instead."
        )
        if np.issubdtype(self.t.dtype, float):
            if not self.hankel_preprocessing:
                assert time_forecast[0] == solver.time_fit[-1] + expected_delta_t, (
                    "Expected 'time_forecast[0]' to be ahead of time_fit[-1] "
                    f"by a value of {expected_delta_t} "
                    f"but got {time_forecast[0] - solver._time_fit[-1]} instead."
                )
            else:
                assert (
                    time_forecast[0] == solver.time_fit_original[-1] + expected_delta_t
                ), (
                    "Expected 'time_forecast[0]' to be ahead of time_fit_original[-1] "
                    f"by a value of {expected_delta_t} "
                    f"but got {time_forecast[0] - solver._time_fit[-1]} instead."
                )

        else:
            if not self.hankel_preprocessing:
                assert time_forecast[0] == solver.time_fit[-1] + expected_delta_time, (
                    "Expected 'time_forecast[0]' to be ahead of time_fit[-1] "
                    f"by a value of {expected_delta_time} "
                    f"but got {time_forecast[0] - solver._time_fit[-1]} instead."
                )
            else:
                assert (
                    time_forecast[0]
                    == solver.time_fit_original[-1] + expected_delta_time
                ), (
                    "Expected 'time_forecast[0]' to be ahead of time_fit_original[-1] "
                    f"by a value of {expected_delta_time} "
                    f"but got {time_forecast[0] - solver._time_fit[-1]} instead."
                )

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_predict(self, solver):
        """Test the private predict() method, ensuring it returns the
        same output as BOPDMD.forecast() from PyDMD.
        """
        solver = getattr(self, solver)
        t, _ = solver._generate_forecast_time_vector(
            forecast_span="20 s",
            dt="2 s",
        )
        if solver.num_trials == 0:
            # without bagging

            # no Dask
            forecast_np = solver._predict(t, use_dask=False)
            forecast_np_pydmd = solver.solver.forecast(t)
            assert isinstance(forecast_np, np.ndarray), (
                "Expected the forecast to be a np.ndarray, "
                f"but got {type(forecast_np)} instead."
            )
            np.testing.assert_allclose(
                forecast_np.real,
                forecast_np_pydmd.real,
                err_msg="Expected the forecast to match the output of PyDMD.",
            )

            # with Dask
            forecast_da = solver._predict(t, use_dask=True)
            assert isinstance(forecast_da, da.Array), (
                "Expected the forecast to be a da.Array, "
                f"but got {type(forecast_da)} instead."
            )
            forecast_np = forecast_da.compute()
            np.testing.assert_allclose(
                forecast_np.real,
                forecast_np_pydmd.real,
                err_msg="Expected the forecast to match the output of PyDMD.",
            )
        else:
            # with bagging

            # no Dask
            forecast_np, forecast_var_np = solver._predict(t, use_dask=False)
            forecast_np_pydmd, forecast_var_np_pydmd = solver.solver.forecast(t)
            assert isinstance(forecast_np, np.ndarray), (
                "Expected the mean forecast to be a np.ndarray, "
                f"but got {type(forecast_np)} instead."
            )
            assert isinstance(forecast_var_np, np.ndarray), (
                "Expected the forecast variance to be a np.ndarray, "
                f"but got {type(forecast_var_np)} instead."
            )
            np.testing.assert_allclose(
                forecast_np.real,
                forecast_np_pydmd.real,
                err_msg="Expected the mean forecast to match the output of PyDMD.",
            )
            np.testing.assert_allclose(
                forecast_var_np.real,
                forecast_var_np_pydmd.real,
                err_msg="Expected the forecast variance to match the output of PyDMD.",
            )

            # with Dask
            forecast_da, forecast_var_da = solver._predict(t, use_dask=True)
            assert isinstance(forecast_da, da.Array), (
                "Expected the mean forecast to be a da.Array, "
                f"but got {type(forecast_da)} instead."
            )
            assert isinstance(forecast_var_da, da.Array), (
                "Expected the forecast variance to be a da.Array, "
                f"but got {type(forecast_var_da)} instead."
            )
            forecast_np, forecast_var_np = (
                forecast_da.compute(),
                forecast_var_da.compute(),
            )
            np.testing.assert_allclose(
                forecast_np.real,
                forecast_np_pydmd.real,
                err_msg="Expected the mean forecast to match the output of PyDMD.",
            )
            np.testing.assert_allclose(
                forecast_var_np.real,
                forecast_var_np_pydmd.real,
                err_msg="Expected the forecast variance to match the output of PyDMD.",
            )

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_forecast(self, solver):
        """Test for the forecast() method."""
        solver = getattr(self, solver)
        forecast_span, dt = "10 s", "1 s"
        if self.hankel_preprocessing:
            expected_forecast_shape = (
                self.u.shape[0] // self.d,
                10,
            )  # 10: 10s span, 1s interval
        else:
            expected_forecast_shape = (self.u.shape[0], 10)  # 10: 10s span, 1s interval
        expected_forecast_dims = (self.u.dims[0], solver.time_dimension)
        _, expected_forecast_t_vector = solver._generate_forecast_time_vector(
            forecast_span=forecast_span,
            dt=dt,
        )
        if solver.num_trials == 0:
            # no bagging
            forecast = solver.forecast(forecast_span=forecast_span, dt=dt)
            assert isinstance(forecast, xr.DataArray), (
                "Expected 'forecast' to be of type 'xr.DataArray', "
                f"but got {type(forecast)} instead."
            )
            assert forecast.dims == expected_forecast_dims, (
                f"Expected 'forecast' to have dimensions {expected_forecast_dims}, "
                f"but got {forecast.dims} instead."
            )
            assert forecast.shape == expected_forecast_shape, (
                f"Expected 'forecast' to have shape {expected_forecast_shape}, "
                f"but got {forecast.shape}."
            )
            np.testing.assert_equal(
                forecast.time.values,
                expected_forecast_t_vector,
                err_msg=(
                    f"Expected the forecast time vector to be "
                    f"{expected_forecast_t_vector}, but got {forecast.time.values}."
                ),
            )
        else:
            # with bagging
            forecast, forecast_var = solver.forecast(forecast_span=forecast_span, dt=dt)
            assert isinstance(forecast, xr.DataArray), (
                "Expected 'forecast' to be of type 'xr.DataArray', "
                f"but got {type(forecast)} instead."
            )
            assert forecast.dims == expected_forecast_dims, (
                f"Expected 'forecast' to have dimensions {expected_forecast_dims}, "
                f"but got {forecast.dims} instead."
            )
            assert forecast.shape == expected_forecast_shape, (
                f"Expected 'forecast' to have shape {expected_forecast_shape}, "
                f"but got {forecast.shape}."
            )
            np.testing.assert_equal(
                forecast.time.values,
                expected_forecast_t_vector,
                err_msg=(
                    f"Expected the forecast time vector to be "
                    f"{expected_forecast_t_vector}, but got {forecast.time.values}."
                ),
            )
            assert isinstance(forecast_var, xr.DataArray), (
                "Expected 'forecast_var' to be of type 'xr.DataArray', "
                f"but got {type(forecast)} instead."
            )
            assert forecast_var.dims == expected_forecast_dims, (
                f"Expected 'forecast_var' to have dimensions {expected_forecast_dims}, "
                f"but got {forecast.dims} instead."
            )
            assert forecast_var.shape == expected_forecast_shape, (
                f"Expected 'forecast_var' to have shape {expected_forecast_shape}, "
                f"but got {forecast.shape}."
            )
            np.testing.assert_equal(
                forecast_var.time.values,
                expected_forecast_t_vector,
                err_msg=(
                    "Expected the forecast variance time vector to be "
                    f"{expected_forecast_t_vector}, but got {forecast.time.values}."
                ),
            )

    @pytest.mark.parametrize("t", [slice(5), slice(5, 10), 10])
    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_reconstruct(self, solver, t):
        """Test for the reconstruct() method."""
        solver = getattr(self, solver)
        reconstruction = solver.reconstruct(t)
        expected_reconstruct_dims = (self.u.dims[0], solver.time_dimension)

        def check_reconstruction_shape(reconstruction: xr.DataArray):
            if isinstance(t, slice):
                if self.hankel_preprocessing:
                    expected_reconstruction_shape = (
                        solver._modes.shape[0] // self.d,
                        5,
                    )  # 5: len of slice
                else:
                    expected_reconstruction_shape = (
                        solver._modes.shape[0],
                        5,
                    )  # 5: len of slice
                assert reconstruction.shape == expected_reconstruction_shape, (
                    "Expected 'reconstruction' to have shape "
                    f"{expected_reconstruction_shape}, "
                    f"but got {reconstruction.shape} instead."
                )
            else:
                if self.hankel_preprocessing:
                    expected_reconstruction_shape = (
                        solver._modes.shape[0] // self.d,
                        1,
                    )  # 1: single snapshot
                else:
                    expected_reconstruction_shape = (
                        solver._modes.shape[0],
                        1,
                    )  # 1: single snapshot
                assert reconstruction.shape == expected_reconstruction_shape, (
                    "Expected 'reconstruction' to have shape "
                    f"{expected_reconstruction_shape}, "
                    f"but got {reconstruction.shape} instead."
                )

        if solver.num_trials == 0:
            # no bagging
            assert isinstance(reconstruction, xr.DataArray), (
                "Expected 'reconstruction' to be of type 'xr.DataArray', "
                f"but got {type(reconstruction)} instead."
            )
            assert reconstruction.dims == expected_reconstruct_dims, (
                "Expected 'reconstruction' to have dimensions "
                f"{expected_reconstruct_dims}, but got {reconstruction.dims} instead."
            )
            check_reconstruction_shape(reconstruction)
            assert np.array_equal(
                reconstruction[solver.time_dimension].values,
                solver.time_fit[t],
            ), (
                "Expected the reconstruction time vector to be: "
                f"{solver.time_fit[t]}, "
                f"but got {reconstruction[solver.time_dimension].values} instead."
            )
        else:
            # with bagging
            reconstruction_mean, reconstruction_var = reconstruction
            assert isinstance(reconstruction_mean, xr.DataArray), (
                "Expected 'reconstruction_mean' to be of type 'xr.DataArray', "
                f"but got {type(reconstruction_mean)} instead."
            )
            assert isinstance(reconstruction_var, xr.DataArray), (
                "Expected 'reconstruction_var' to be of type 'xr.DataArray', "
                f"but got {type(reconstruction_var)} instead."
            )
            for array in (reconstruction_mean, reconstruction_var):
                assert array.dims == expected_reconstruct_dims, (
                    "Expected 'reconstruction' to have dimensions "
                    f"{expected_reconstruct_dims}, but got {array.dims} instead."
                )
                check_reconstruction_shape(array)
                assert np.array_equal(
                    array[solver.time_dimension].values, solver.time_fit[t]
                ), (
                    "Expected the reconstruction time vector to be: "
                    f"{solver.time_fit[t]}, "
                    f"but got {array[solver.time_dimension].values} instead."
                )


class TestOptDMDRandomData(BaseTestOptDMD):
    """Tests for the OptDMD class using a DataGenerator instance
    to generate random input data, with a time vector containing
    datetimes.
    """

    @classmethod
    def setup_class(cls):
        generator = DataGenerator(seed=1234)
        generator.generate_svd_results(n_components=10)
        cls.u, cls.s, cls.v, cls.t = generator.u, generator.s, generator.v, generator.t
        cls.optdmd = OptDMD()
        cls.optdmd_bagging = OptDMD(num_trials=5, seed=1234)
        cls.hankel_preprocessing = False


class TestOptDMDCoherentSignal(BaseTestOptDMD):
    """Tests for the OptDMD class using a SignalGenerator instance
    to generate coherent spatio-temporal input data, with a time
    vector containing floats.
    """

    @classmethod
    def setup_class(cls):
        generator = SignalGenerator()
        generator.generate_svd_results(random_seed=1234)
        cls.u, cls.s, cls.v = generator.u, generator.s, generator.v
        cls.t, cls.components = generator.t, generator.components
        cls.optdmd = OptDMD(input_time_units="s")
        cls.optdmd_bagging = OptDMD(
            input_time_units="s", num_trials=10, trial_size=0.9, seed=1234
        )
        cls.hankel_preprocessing = False

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_correct_eigs(self, solver):
        """Test that OptDMD can find the correct frequencies of oscillation
        in the data.
        """
        solver = getattr(self, solver)
        omegas = np.sort(
            [component["omega"] for component in self.components],
        )[::-1]  # get the temporal frequencies of oscillation from the data generator
        eigs_imag = [
            np.abs(eig.imag) for eig in solver.eigs
        ]  # get the imaginary DMD eigenvalues
        eigs_imag = np.sort(np.unique(np.round(eigs_imag, decimals=2)))[::-1]
        np.testing.assert_array_almost_equal(
            omegas,
            eigs_imag,
            decimal=2,
            err_msg=(
                f"The expected imaginary eigenvalues are: "
                f"{np.round(omegas, decimals=2)}, "
                "while the computed imaginary eigenvalues are: "
                f"{np.round(eigs_imag, decimals=2)}."
            ),
        )


class TestOptDMDHankelMatrix(TestOptDMDCoherentSignal):
    """Tests for the OptDMD class using a SignalGenerator instance
    to generate coherent spatio-temporal input data, pre-processed
    with the Hankel pre-processor prior to SVD and DMD to apply
    time-delay embedding.
    """

    @classmethod
    def setup_class(cls):
        generator = SignalGenerator()
        generator.generate_signal(random_seed=1234)
        cls.components = generator.components
        X = generator.da.transpose("x", "time")
        cls.d = 2
        X_d = hankel_preprocessing(X, d=cls.d)
        cls.hankel_preprocessing = True
        # convert to Dask-backed Xarray as TruncatedSVD currently only
        # supports Dask arrays
        X_d = X_d.copy(data=da.from_array(X_d.data))
        n_components = len(cls.components) * cls.d
        tsvd = TruncatedSVD(n_components=n_components)
        tsvd.fit(X_d)
        cls.u, cls.s, cls.v = tsvd.u, tsvd.s, tsvd.v
        cls.t = X_d.time
        cls.optdmd = OptDMD()
        cls.optdmd_bagging = OptDMD(num_trials=5, seed=1234)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    @pytest.mark.parametrize("hankel_d", [2, 3])
    def test_extract_hankel_prediction(self, array_type, hankel_d):
        """Test for the extract_hankel_prediction method."""
        arr = np.random.randn(100, 10)
        if array_type == "dask":
            arr = da.from_array(arr)
        # need to transform array into DataArray because
        # hankel_preprocessing() only takes DataArrays
        X = xr.DataArray(
            data=arr,
            dims=("x", "t"),
            coords={
                "x": ("x", np.arange(arr.shape[0])),
                "t": ("t", np.arange(arr.shape[1])),
            },
        )
        X_d = hankel_preprocessing(X, hankel_d)
        X_extracted = self.optdmd._extract_hankel_prediction(X_d.data, hankel_d)
        if isinstance(X_extracted, da.Array):
            assert np.array_equal(X_extracted.compute(), X.values), (
                "Expected the output of extract_hankel_preprocessing() "
                "to match the input of hankel_preprocessing()."
            )
        else:
            assert np.array_equal(X_extracted, X.values), (
                "Expected the output of extract_hankel_preprocessing() "
                "to match the input of hankel_preprocessing()."
            )

    @pytest.mark.parametrize("t", [0, 1, 5, 10, -2, -1])
    def test_extract_hankel_time(self, t):
        """Test for the extract_hankel_time method."""
        t_hankel, lag = self.optdmd._extract_hankel_time(t)
        if t < 0:
            expected_t_hankel = t
            expected_lag = self.d - 1
        elif t - self.d < 0:
            expected_t_hankel = 0
            expected_lag = t
        else:
            expected_t_hankel = t - self.d + 1
            expected_lag = self.d - 1
        assert t_hankel == expected_t_hankel, (
            f"Expected t_hankel to be {expected_t_hankel}, "
            f"but got {t_hankel} instead."
        )
        assert (
            lag == expected_lag
        ), f"Expected lag to be {expected_lag}, but got {lag} instead."
