import warnings
from typing import Literal, cast

import dask
import dask.array as da
import numpy as np
import xarray as xr
from dask.utils import parse_bytes
from pydmd import BOPDMD

import svdrom.config as config
from svdrom.logger import setup_logger

logger = setup_logger("DMD", "dmd.log")


class OptDMD:
    def __init__(
        self,
        n_modes: int = -1,
        time_dimension: str = "time",
        time_units: Literal["s", "h"] = "s",
        input_time_units: Literal["s", "h"] | None = None,
        num_trials: int = 0,
        trial_size: int | float = 0.6,
        parallel_bagging: bool = False,
        seed: int | None = None,
    ) -> None:
        """Optimized Dynamic Mode Decomposition (DMD) via variable
        projection method for nonlinear least squares, with optional
        bootstrap aggregation (bagging) for uncertainty quantification.

        This class makes use of the BOPDMD class from the PyDMD library
        (https://pydmd.github.io/PyDMD/bopdmd.html).

        Parameters
        ----------
        n_modes: int
            Number of DMD modes to compute (must be positive or -1 for all
            available modes).
        time_dimension: str
            Name of the time dimension in the input data. Default is "time".
        time_units: str
            Units in which you wish to treat the time dimension. Default is "s".
            Must be one of {"s", "h"}, where "s" is seconds and "h" is hours.
        input_time_units: str | None
            Units of the time vector of the input data. Default is None, which
            means same as "time_units". If you provide a string, it must be one
            of {"s", "h"}. Only used if the time vector of the input data consists
            of floats instead of datetimes, and hence the units cannot be inferred.
        num_trials: int
            Number of bagging trials to perform during the OptDMD fit.
            Default is 0 (no bagging).
        trial_size: int | float
            Size of the randomly selected subset of snapshots to use for
            each trial of bagged optimized DMD. If it's a positive integer,
            "trial_size" many snapshots will be used per trial. If it's a
            float between 0 and 1, then "trial_size" denotes the fraction of
            snapshots to be used per trial. Default is 0.6.
        parallel_bagging: bool
            Whether to perform bagging sequentially (False) or in parallel
            (True). The default is False. Only active when 'num_trials' is
            greater than zero. Note that, for parallel bagging to actually take
            place, a Dask multi-processing or distributed scheduler must be employed.
        seed: int | None
            Random seed used to initialize the random number generator.
            Default is None, which initializes the random number generator using
            system entropy. Used only if `num_trials` is a positive integer, i.e.
            when performing bagging. Set seed to a fixed integer value to obtain
            reproducible results.

        Notes
        -----
        This class is a wrapper of the `BOPDMD.fit_econ()` method, which fits
        an approximate Optimized DMD on an array X by operating on the SVD of X.
        """
        if n_modes != -1 and n_modes < 1:
            msg = "'n_modes' must be a positive integer or -1."
            logger.exception(msg)
            raise ValueError(msg)
        if time_units not in {"s", "h"}:
            msg = "'time_units' must be one of {'s', 'h'}."
            logger.exception(msg)
            raise ValueError(msg)
        if input_time_units is not None and input_time_units not in {"s", "h"}:
            msg = "'input_time_units' must be one of {'s', 'h'}."
            logger.exception(msg)
            raise ValueError(msg)
        if num_trials < 0 or not isinstance(num_trials, int):
            msg = "'num_trials' must be an integer greater than zero."
            logger.exception(msg)
            raise ValueError(msg)
        if (
            isinstance(trial_size, float)
            and (trial_size <= 0 or trial_size >= 1)
            or isinstance(trial_size, int)
            and trial_size <= 0
        ):
            msg = "'trial_size' must be a positive integer or a float between 0 and 1."
            logger.exception(msg)
            raise ValueError(msg)

        self._n_modes = n_modes
        self._time_dimension = time_dimension
        self._time_units = time_units
        self._input_time_units = input_time_units or time_units
        self._num_trials = num_trials
        self._trial_size = trial_size
        self._parallel_bagging = parallel_bagging
        self._seed = seed
        self._eigs: np.ndarray | None = None
        self._eigs_std: np.ndarray | None = None
        self._amplitudes: np.ndarray | None = None
        self._amplitudes_std: np.ndarray | None = None
        self._modes: xr.DataArray | None = None
        self._modes_std: xr.DataArray | None = None
        self._solver: BOPDMD | None = None
        self._time_fit: np.ndarray | None = None
        self._time_fit_original: np.ndarray | None = None
        self._t_fit: np.ndarray | None = None  # internal use only
        self._t_fit_original: np.ndarray | None = None  # internal use only
        self._is_datetime: bool = False  # internal use only
        self._dynamics: xr.DataArray | None = None
        self._hankel_d: int = 1
        self._hankel_time_mapping: dict | None = None  # internal use only

    @property
    def n_modes(self) -> int:
        """Number of DMD modes (read-only)."""
        return self._n_modes

    @property
    def time_dimension(self) -> str:
        """Name of the time dimension (read-only)."""
        return self._time_dimension

    @property
    def eigs(self) -> np.ndarray | None:
        """The DMD eigenvalues (read-only)."""
        return self._eigs

    @property
    def modes(self) -> xr.DataArray | None:
        """The DMD modes (read-only)."""
        return self._modes

    @property
    def modes_averaged(self) -> xr.DataArray | None:
        """The DMD modes, averaged across time lags if Hankel
        pre-processing has been used (read-only). You should
        use this attribute instead of 'modes' for visualization
        when applying Hankel pre-processing. If Hankel pre-processing
        has not been used it just returns the standard DMD modes.
        """
        if not isinstance(self._modes, xr.DataArray):
            return None
        if self._hankel_d == 1:
            return self._modes
        return self._average_modes_across_lags(self._modes, self._hankel_d)

    @property
    def amplitudes(self) -> np.ndarray | None:
        """The DMD amplitudes (read-only)."""
        return self._amplitudes

    @property
    def eigs_std(self) -> np.ndarray | None:
        """The standard deviation of the DMD eigenvalues,
        when using bagging (read-only).
        """
        return self._eigs_std

    @property
    def modes_std(self) -> xr.DataArray | None:
        """The standard deviation of the DMD modes,
        when using bagging (read-only)."""
        return self._modes_std

    @property
    def modes_std_averaged(self) -> xr.DataArray | None:
        """The STD of the DMD modes, averaged across time lags if
        Hankel pre-processing has been used (read-only). You should
        use this attribute instead of 'modes_std' for visualization
        when applying Hankel pre-processing. If Hankel pre-processing
        has not been used it just returns the STD of the DMD modes
        without any further processing.
        """
        if not isinstance(self._modes_std, xr.DataArray):
            return None
        if self._hankel_d == 1:
            return self._modes_std
        return self._average_modes_across_lags(self._modes_std, self._hankel_d)

    @property
    def amplitudes_std(self) -> np.ndarray | None:
        """The standard deviation of the DMD amplitudes,
        when using bagging (read-only)."""
        return self._amplitudes_std

    @property
    def num_trials(self) -> int:
        """The number of bagging trials (read-only)."""
        return self._num_trials

    @property
    def trial_size(self) -> int | float:
        """The bagging trial size (read-only)."""
        return self._trial_size

    @property
    def time_units(self) -> str:
        """The time units to use in the DMD fit and forecast (read-only)."""
        return self._time_units

    @property
    def input_time_units(self) -> str | None:
        """The units of the time vector of the input data (read-only)."""
        return self._input_time_units

    @property
    def solver(self) -> BOPDMD | None:
        """The DMD solver instance (read-only)."""
        return self._solver

    @property
    def time_fit(self) -> np.ndarray | None:
        """The time vector for the DMD fit (read-only).
        If Hankel pre-processing has been applied, this corresponds
        to the time vector of the preprocessed data, not the original
        time vector.
        """
        return self._time_fit

    @property
    def time_fit_original(self) -> np.ndarray | None:
        """The original time vector of the training data when
        Hankel pre-processing has been applied (read-only).
        """
        return self._time_fit_original

    @property
    def parallel_bagging(self) -> bool:
        """Whether bagging is performed sequentially or in parallel (read-only)."""
        return self._parallel_bagging

    @property
    def dynamics(self) -> xr.DataArray | None:
        """The time evolution of each DMD mode,
        scaled by the mode amplitude (read-only).
        """
        return self._dynamics

    @property
    def hankel_d(self) -> int:
        """Hankel matrix rank if time-delay embedding has
        been applied via Hankel pre-processing (read-only).
        Note that if time-delay embedding has not been applied,
        this parameter will have a value of 1.
        """
        return self._hankel_d

    @staticmethod
    def _average_modes_across_lags(modes: xr.DataArray, hankel_d: int) -> xr.DataArray:
        """Given a matrix of modes or modes STDs, split each column/mode
        into hankel_d equal parts and average together, eliminating the
        coordinate of Hankel lags.
        """
        if modes.values.shape[0] % hankel_d != 0:
            msg = (
                f"The number of rows ({modes.values.shape[0]}) in the matrix of modes "
                f"must be divisible by the hankel rank ({hankel_d})."
            )
            raise ValueError(msg)
        modes_averaged = modes.values
        modes_averaged = np.average(
            modes_averaged.reshape(
                hankel_d,
                modes_averaged.shape[0] // hankel_d,
                modes_averaged.shape[1],
            ),
            axis=0,
        )
        dim0 = modes.dims[0]
        modes = modes.sel(
            {dim0: modes[config.get("hankel_coord_name")] == 0}
        )  # keep only the samples associated with the zero time-lag
        modes = modes.drop_vars(config.get("hankel_coord_name"))
        # return same DataArray structure with new data
        return modes.copy(data=modes_averaged)

    def _check_svd_inputs(self, u: xr.DataArray, s: np.ndarray, v: xr.DataArray):
        """Check that the passed SVD results are valid."""
        if not isinstance(u.data, np.ndarray):
            msg = "The left singular vectors have not been computed."
            logger.exception(msg)
            raise ValueError(msg)
        if not isinstance(v.data, np.ndarray):
            msg = "The right singular vectors have not been computed."
            logger.exception(msg)
            raise ValueError(msg)
        if len(u.components) != len(v.components) or len(u.components) != len(s):
            msg = "'u', 's' and 'v' must have the same number of components."
            logger.exception(msg)
            raise ValueError(msg)
        if self._time_dimension not in v.dims:
            msg = (
                f"Specified time dimension '{self._time_dimension}' not "
                "a dimension of the right singular vectors 'v'."
            )
            logger.exception(msg)
            raise ValueError(msg)

        def is_sorted(x):
            return np.all(x[:-1] <= x[1:])

        if not is_sorted(v[self._time_dimension].values):
            msg = (
                f"Time dimension '{self._time_dimension}' is not "
                "sorted in the right singular vectors 'v'."
            )
            logger.exception(msg)
            raise ValueError(msg)

        if config.get("hankel_coord_name") in u.coords:
            if u[config.get("hankel_coord_name")].dims[0] != u.dims[0]:
                msg = (
                    f"The dimension of the {config.get('hankel_coord_name')} "
                    "coordinate should be the same as the first dimension of the "
                    "left singular vectors 'u'."
                )
                logger.exception(msg)
                raise ValueError(msg)
            if np.any(np.unique(u[config.get("hankel_coord_name")].values) < 0):
                msg = (
                    f"The {config.get('hankel_coord_name')} coordinate should only "
                    "contain values greater than or equal to zero."
                )
                logger.exception(msg)
                raise ValueError(msg)
            if config.get("hankel_time_mapping_attr") not in v.attrs:
                msg = (
                    "Given the left singular vectors (u) contain a "
                    f"{config.get('hankel_coord_name')} coordinate, expected "
                    "right singular vectors (v) to contain a "
                    f"{config.get('hankel_time_mapping_attr')} attribute."
                )
                logger.exception(msg)
                raise ValueError(msg)

    def _get_time_conversion_factor(self, from_units: str, to_units: str) -> float:
        """Get the time conversion factor from one unit of time to another.
        Currently, only seconds ('s') and hours ('h') are supported.
        """
        to_seconds = {
            "s": 1.0,
            "h": 3600.0,
        }
        return to_seconds[from_units] / to_seconds[to_units]

    def _generate_fit_time_vector(
        self, v: xr.DataArray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Given the right singular vectors containing the temporal
        information, generate the time vector(s) for the DMD fit.
        When Hankel pre-processing has been applied, it also calculates
        the fit time vectors of the original data, although these are
        not used for the DMD fit.

        Parameters
        ----------
        v: xr.DataArray
            The right singular vectors containing the temporal information.

        Returns
        -------
        t_fit: np.ndarray[float]
            Vector to be fed to the DMD fit call.
        time_fit: np.ndarray[float] | np.ndarray[np.timedelta64]
            Vector representing the true time of the fitted data,
            which may consist of floats or timedeltas, depending
            on the input data.
        """

        def calculate_time_deltas_float(time_deltas: np.ndarray) -> np.ndarray:
            if np.issubdtype(time_deltas.dtype, np.timedelta64):
                # if the input time vector contains datetimes
                self._is_datetime = True
                time_deltas_td = time_deltas.astype(f"timedelta64[{self._time_units}]")
                time_deltas_float = time_deltas_td / np.timedelta64(1, self._time_units)
            else:
                # if the input time vector contains floats
                self._is_datetime = False
                time_deltas_float = time_deltas.astype(np.float64)
                conversion_factor = self._get_time_conversion_factor(
                    self._input_time_units,
                    self._time_units,
                )
                time_deltas_float *= conversion_factor
            return time_deltas_float

        time_fit = v[self._time_dimension].values
        time_deltas = np.diff(time_fit)
        time_deltas_float = calculate_time_deltas_float(time_deltas)
        time_vector = np.cumsum(time_deltas_float)
        self._time_fit = time_fit  # vector representing true time of the training data
        start_time = np.array([0], dtype=np.float64)
        self._t_fit = np.concatenate(
            (start_time, time_vector)
        )  # vector to be fed to the `fit()` call

        if self._hankel_d > 1:
            # If Hankel pre-processing has been applied, do the same
            # for the data's original time vector
            if not self._hankel_time_mapping:
                msg = "The Hankel time mapping dictionary is not available."
                raise RuntimeError(msg)
            time_fit = np.sort(list(self._hankel_time_mapping.keys()))
            time_deltas = np.diff(time_fit)
            time_deltas_float = calculate_time_deltas_float(time_deltas)
            time_vector = np.cumsum(time_deltas_float)
            self._time_fit_original = time_fit
            start_time = np.array([0], dtype=np.float64)
            self._t_fit_original = np.concatenate((start_time, time_vector))

        return self._t_fit, self._time_fit

    def _generate_forecast_time_vector(
        self, forecast_span: str | int, dt: str | int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Given a forecast time span and time step, generate the time vector
        for the DMD forecast.

        Parameters
        ----------
        forecast_span: str | int
            The total time span for the forecast. If int, interpreted as time
            in the model's time units. If str, should be in the format "value units",
            e.g. "30 D" for 30 days.
        dt: str | int | None, optional
            The time step or number of points for the forecast. If str, should be in
            the format "value units", e.g. "1 h" for 1 hour. If int, interpreted as
            number of forecast points. If None, uses the average time step of the
            training data.

        Returns
        -------
        t_forecast: np.ndarray[float]
            The time vector to be fed to the `forecast()` call.
        time_forecast: np.ndarray[float] | np.ndarray[np.timedelta64]
            The time vector representing the true time of the forecast,
            which may consist of floats or timedeltas, depending on the
            input data.
        """
        if self._t_fit is None or self._time_fit is None:
            msg = "The DMD fit time vector is not initialized."
            raise ValueError(msg)

        # parse forecast_span
        if isinstance(forecast_span, int):
            # forecast span is an integer
            span_td = np.timedelta64(forecast_span, self._time_units)
        else:
            # forecast_span is a string
            span_parts = forecast_span.split(" ")
            if len(span_parts) != 2:
                msg = (
                    "String forecast_span must be in the format 'value units', "
                    "e.g. '30 D'."
                )
                raise ValueError(msg)
            span_td = np.timedelta64(int(span_parts[0]), span_parts[1])  # type: ignore[call-overload]
        span_float = span_td / np.timedelta64(1, self._time_units)

        # determine time step
        if dt is None:
            # use average time step of training data
            if len(self._t_fit) > 1:
                time_step_float = np.mean(np.diff(self._t_fit))
            else:
                msg = (
                    "Cannot determine time step from training data "
                    "with only one time point."
                )
                raise ValueError(msg)
        elif isinstance(dt, str):
            # parse time step string
            dt_parts = dt.split(" ")
            if len(dt_parts) != 2:
                msg = "String dt must be in the format 'value units', e.g. '1 h'."
                raise ValueError(msg)
            time_step_td = np.timedelta64(int(dt_parts[0]), dt_parts[1])  # type: ignore[call-overload]
            time_step_float = time_step_td / np.timedelta64(1, self._time_units)
        else:
            # dt is an int and specifies number of points
            if dt <= 0:
                msg = "Number of forecast points must be positive."
                raise ValueError(msg)
            # calculate time step to get the requested number of points
            time_step_float = span_float / dt
        time_step_td = time_step_float.astype(f"timedelta64[{self._time_units}]")

        if (
            self._is_datetime
            and time_step_td / np.timedelta64(1, self._time_units) == 0
        ):
            msg = (
                f"The calculated forecast time step is not valid: {time_step_td}. "
                "Ensure the requested forecast time step is larger than the time step "
                "of the fitted data."
            )
            logger.exception(msg)
            raise RuntimeError(msg)

        # generate the time vector to be fed to the `forecast()` call
        forecast_start = self._t_fit[-1]
        forecast_end = forecast_start + span_float
        t_forecast = np.arange(
            forecast_start + time_step_float,
            forecast_end + time_step_float,
            time_step_float,
        )

        # generate the time vector representing true time of the forecast
        if self._hankel_d > 1:
            if self._time_fit_original is None:
                msg = (
                    "Required time vector information "
                    "for Hankel post-processing is missing."
                )
                raise RuntimeError(msg)
            forecast_start = self._time_fit_original[-1]
        else:
            forecast_start = self._time_fit[-1]
        forecast_end = (
            (forecast_start + span_td)
            if self._is_datetime
            else (forecast_start + span_float)
        )
        time_step = time_step_td if self._is_datetime else time_step_float
        time_forecast = np.arange(
            forecast_start + time_step,
            forecast_end + time_step,
            time_step,
        )

        return t_forecast, time_forecast

    def _extract_results(
        self, bopdmd: BOPDMD, u: xr.DataArray, v: xr.DataArray
    ) -> "OptDMD":
        """Given the fitted BOPDMD instance, the left singular vectors
        containing the spatial information, and the right singular vectors
        containing the temporal information, store them in the instance attributes.
        """
        self._solver = bopdmd
        self._modes = u.copy(data=bopdmd.modes)  # use new data with structure from u
        self._modes.name = "dmd_modes"
        self._eigs = bopdmd.eigs
        self._amplitudes = bopdmd.amplitudes
        try:
            self._dynamics = v.copy(
                data=bopdmd.dynamics
            )  # use new data with original structure
        except Exception:
            msg = "Could not retrieve dynamics from BOPDMD solver."
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if self.num_trials > 0:
            self._modes_std = u.copy(
                data=bopdmd.modes_std
            )  # use new data with structure from u
            self._modes_std.name = "dmd_modes_std"
            self._eigs_std = bopdmd.eigenvalues_std
            self._amplitudes_std = bopdmd.amplitudes_std

        return self

    def fit(
        self, u: xr.DataArray, s: np.ndarray, v: xr.DataArray, **kwargs
    ) -> "OptDMD":
        """Fit the OptDMD model to the results of a Singular Value Decomposition (SVD).
        If you have requested bagging when instantiating the model, "num_trials" OptDMD
        trials will be fitted by randomly subsampling snapshots of the data, and the
        trials will be ensemble averaged. If you have requested to perform bagging in
        parallel and you have a Dask multi-processing or distributed scheduler running,
        bagging will be performed in parallel.

        Parameters
        ----------
        u: xarray.DataArray, shape (n_spatial_points, n_components)
            The left singular vectors, containing the spatial information.
        s: np.ndarray, shape (n_components,)
            The singular values.
        v: xarray.DataArray, shape (n_components, n_timesteps)
            The right singular vectors, containing the temporal information.
        **kwargs:
            Additional keyword arguments to pass to PyDMD's BOPDMD constructor.
            See https://pydmd.github.io/PyDMD/bopdmd.html for more info.

        Notes
        -----
        To perform a DMD fit on a Hankel pre-processed (a.k.a. time-delay embedding)
        matrix, 'u', 's' and 'v' must come from the singular value decomposition
        of the Hankel pre-processed matrix via the TruncatedSVD class.
        """
        self._check_svd_inputs(u, s, v)
        if self._n_modes > len(s):
            msg = (
                "The requested number of DMD modes exceeds the number "
                "of available SVD components."
            )
            logger.exception(msg)
            raise ValueError(msg)
        if self._n_modes == -1:
            self._n_modes = len(s)
        u, s, v = u[:, : self._n_modes], s[: self._n_modes], v[: self._n_modes, :]
        if config.get("hankel_coord_name") in u.coords:
            self._hankel_d = len(np.unique(u[config.get("hankel_coord_name")].values))
            self._hankel_time_mapping = v.attrs[config.get("hankel_time_mapping_attr")]

        bopdmd = BOPDMD(
            svd_rank=self._n_modes,
            use_proj=True,
            proj_basis=u.data,
            num_trials=self._num_trials,
            trial_size=self._trial_size,
            parallel_bagging=self._parallel_bagging,
            seed=self._seed,
            **kwargs,
        )  # type: ignore[call-arg]
        t_fit, _ = self._generate_fit_time_vector(v)
        logger.info("Computing the DMD fit...")
        try:
            if self._num_trials > 0:
                if self._parallel_bagging:
                    logger.info("Bagging will be performed in parallel.")
                else:
                    logger.info("Bagging will be performed sequentially.")
            bopdmd.fit_econ(
                s,
                v.data,
                t_fit,
            )
        except Exception as e:
            msg = "Error computing the DMD fit."
            logger.exception(msg)
            raise RuntimeError(msg) from e
        logger.info("Done.")
        self._extract_results(bopdmd, u, v)

        return self

    def _rechunk_along_columns(self, arr: da.Array) -> da.Array:
        """Rechunk a Dask matrix along the columns (i.e. the time dimension),
        which allows to more efficiently unstack the spatial dimension
        (the rows) if necessary. The target chunk size is based on the Dask
        config default.
        """
        if self._modes is None:
            msg = "The model has not been fitted. Can't calculate bytes per snapshot."
            raise RuntimeError(msg)
        target_chunk_bytes = parse_bytes(dask.config.get("array.chunk-size"))
        bytes_per_snapshot = self._modes[:, 0].nbytes
        time_chunk_size = round(target_chunk_bytes / bytes_per_snapshot)
        return arr.rechunk((-1, time_chunk_size))

    def _prediction_to_dataarray(
        self,
        prediction: np.ndarray
        | da.Array
        | tuple[da.Array, da.Array]
        | tuple[np.ndarray, np.ndarray],
        time_prediction: np.ndarray,
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """Convert the DMD prediction (i.e. a forecast or a reconstruction),
        deterministic or probabilistic, into formatted xarray.DataArray(s).

        Parameters
        ----------
        prediction: (
            np.ndarray | da.Array
            | tuple[np.ndarray, np.ndarray]
            | tuple[da.Array, da.Array]
        )
            A DMD forecast or reconstruction, deterministic (single array) or
            probabilistic (two arrays - ensemble mean and variance). The arrays can be
            NumPy or Dask arrays. Dimensions should be (space/samples, time).
        time_prediction: np.ndarray[float] | np.ndarray[np.timedelta64]
            The time vector representing the true prediction time. Must have the same
            length as the second dimension of 'prediction'.

        Returns
        -------
        xr.DataArray | tuple[xr.DataArray, xr.DataArray]
            The forecast or reconstruction formatted as xr.DataArray(s).
            If two arrays are passed for a probabilistic prediction, two
            DataArrays are returned, the first corresponding to the mean
            and the second to the variance. The DataArrays are NumPy- or
            Dask-backed, depending on the input arrays.
        """
        if self._modes is None:
            msg = "The DMD modes have not been computed."
            raise RuntimeError(msg)

        dims = (self._modes.dims[0], self._time_dimension)
        # if Hankel pre-processing has been applied, drop the
        # duplicate dimension values along the first dimension,
        # i.e. keep only coordinates associated with lag=0,
        # and drop the Hankel lag coordinate
        if self._hankel_d > 1:
            modes = self._modes.drop_duplicates(dims[0], keep="first").drop_vars(
                config.get("hankel_coord_name")
            )
        else:
            modes = self._modes
        coords = {k: v for k, v in modes.coords.items() if k != "components"}
        coords[self._time_dimension] = time_prediction.squeeze()
        if isinstance(prediction, (np.ndarray | da.Array)):
            return xr.DataArray(
                prediction, dims=dims, coords=coords, name="dmd_prediction"
            )
        prediction_mean = xr.DataArray(
            prediction[0], dims=dims, coords=coords, name="dmd_prediction_mean"
        )
        prediction_var = xr.DataArray(
            prediction[1], dims=dims, coords=coords, name="dmd_prediction_var"
        )
        return prediction_mean, prediction_var

    def _predict(
        self, t: np.ndarray, use_dask: bool = False
    ) -> (
        np.ndarray
        | da.Array
        | tuple[np.ndarray, np.ndarray]
        | tuple[da.Array, da.Array]
    ):
        """Given a time vector and the fitted DMD model, compute a forecast or a
        reconstruction (i.e. a prediction). If bagging has been used, Monte-Carlo
        uncertainty propagation is employed to produce multiple prediction realizations
        and a tuple consisting of the mean prediction and the prediction variance
        is returned. Set the flag `use_dask=True` to compute the prediction as a
        Dask array. Otherwise the prediction is computed as a NumPy array."""
        if self._modes is None or self._eigs is None or self._amplitudes is None:
            msg = "Results of the OptDMD fit are not available."
            raise RuntimeError(msg)

        def draw_from_rand_distr(
            rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Draw eigenvalues and amplitudes from a normal distribution
            scaled by the computed standard deviation.
            """
            if not isinstance(self._eigs, np.ndarray) or not isinstance(
                self._amplitudes, np.ndarray
            ):
                msg = "The DMD model has not been fitted."
                raise RuntimeError(msg)
            if not isinstance(self._eigs_std, np.ndarray) or not isinstance(
                self._amplitudes_std, np.ndarray
            ):
                msg = "The DMD bagging statistics have not been computed."
                raise RuntimeError(msg)
            eigs = self._eigs + np.multiply(
                rng.standard_normal(*self._eigs.shape),
                self._eigs_std,
            )
            amps = self._amplitudes + np.multiply(
                rng.standard_normal(*self._amplitudes.shape),
                self._amplitudes_std,
            )
            return eigs, amps

        rng = np.random.default_rng(self._seed)
        predictions = []

        if use_dask:
            # use Dask to compute the prediction
            modes_da = da.from_array(self._modes, chunks=(-1, -1))
            if self._eigs_std is not None and self._amplitudes_std is not None:
                # when bagging has been used, use Monte-Carlo uncertainty propagation
                # to produce multiple prediction realizations
                for _ in range(self._num_trials):
                    # draw eigenvalues and amplitudes from random distribution
                    eigs, amps = draw_from_rand_distr(rng)
                    amps_da = da.from_array(np.diag(amps), chunks=(-1, -1))
                    exp_da = da.from_array(np.exp(np.outer(eigs, t)))
                    exp_da = self._rechunk_along_columns(exp_da)
                    # compute a prediction realization
                    prediction_da = da.dot(modes_da, da.dot(amps_da, exp_da))
                    predictions.append(prediction_da)

                # stack predictions along new axis and compute mean and variance
                predictions_da = da.stack(predictions, axis=0)
                return predictions_da.mean(axis=0), predictions_da.var(axis=0)

            amps_da = da.from_array(
                np.diag(self._amplitudes),
                chunks=(-1, -1),
            )
            exp_da = da.from_array(
                np.exp(np.outer(self._eigs, t)),
            )
            exp_da = self._rechunk_along_columns(exp_da)
            return da.dot(modes_da, da.dot(amps_da, exp_da))

        # use NumPy to compute the prediction
        if self._eigs_std is not None and self._amplitudes_std is not None:
            # if bagging has been used, use Monte-Carlo uncertainty propagation
            # to produce multiple prediction realizations
            for _ in range(self._num_trials):
                eigs, amps = draw_from_rand_distr(rng)
                forecast = np.linalg.multi_dot(
                    [self._modes, np.diag(amps), np.exp(np.outer(eigs, t))]
                )
                predictions.append(forecast)

            predictions_np = np.stack(predictions, axis=0)
            return predictions_np.mean(axis=0), predictions_np.var(axis=0)

        return np.linalg.multi_dot(
            [
                self._modes,
                np.diag(self._amplitudes),
                np.exp(np.outer(self._eigs, t)),
            ]
        )

    @staticmethod
    def _extract_hankel_prediction(
        prediction: np.ndarray | da.Array,
        hankel_d: int,
        lags: tuple[int,] | tuple[int, int] | None = None,
        rechunk: bool = True,
    ) -> np.ndarray | da.Array:
        """Given a DMD prediction of a Hankel pre-processed (i.e.
        time-delay embedded) matrix, extract the first occurrence of
        each snapshot available in the given matrix.

        Parameters
        ----------
        prediction: np.ndarray | da.Array
            The DMD prediction (a forecast or a reconstruction).
        hankel_d: int
            Hankel rank of the input matrix. It should be an integer
            greater than or equal to 2.
        lags: tuple[int] | tuple[int, int] | None
            A tuple containing the starting and ending lag indices of the
            Hankel matrix to extract, in the form (lag_start, lag_end).
            If lag_start equals lag_end, then pass lags as a single-element
            tuple (lag,). If None, all lags are extracted. Default is None.
        rechunk: bool
            Whether to rechunk the input array into column blocks
            of n rows, where n is the length of a single snapshot (i.e.
            the portion of a Hankel matrix column with the same lag index).
            Default is True.

        Returns
        -------
        np.ndarray | da.Array
            A matrix containing the first occurrence of each snapshot.
            If the input matrix has dimensions (n * hankel_d, t), the
            output matrix will have dimensions (n, t + hankel_d - 1).
            The output matrix is built by extracting the first column
            and last row of the input matrix.
        """
        n = prediction.shape[0] // hankel_d
        t = prediction.shape[1]
        if rechunk and isinstance(prediction, da.Array):
            prediction = prediction.rechunk({0: n, 1: 1})
        snapshots = prediction[:, 0].reshape(hankel_d, -1).T
        if lags is not None:
            if len(lags) == 1:
                snapshots = snapshots[:, lags[0] : lags[0] + 1]
            else:
                snapshots = snapshots[:, lags[0] : lags[1] + 1]
        if t > 1:
            if isinstance(prediction, da.Array):
                snapshots = da.concatenate([snapshots, prediction[-n:, 1:]], axis=1)
            else:
                snapshots = np.concatenate([snapshots, prediction[-n:, 1:]], axis=1)
        return snapshots

    def _extract_hankel_time(
        self,
        t: str | int,
    ) -> tuple[str, int] | tuple[int, int]:
        """Given a single snapshot (either a string label or a
        integer index) in the original time vector, extract the
        corresponding snapshot in the Hankel time vector, and the
        associated lag index of the Hankel pre-processed matrix.

        Parameters
        ----------
        t: str | int
            A single snapshot in the original time vector, either
            as a string label or an integer index.

        Returns
        -------
        str | int
            The corresponding snapshot in the Hankel time vector,
            either as a string label or an integer index. Negative
            indexing is supported.
            The first snapshot available in the Hankel time vector that
            corresponds to the given snapshot in the original time
            vector is returned.
        int
            The lag index in the Hankel pre-processed matrix corresponding
            to the requested snapshot.

        Notes
        -----
        For example, if the original data consisted of t n-long snapshots,
        the original data matrix would have shape (n, t). If we performed
        Hankel pre-processing with Hankel rank d = 2, the resulting matrix
        would have shape (2*n, t-1), where the first row block [:n, :]
        corresponds to lag=0 and the second row block [n:, :] corresponds
        to lag=1. If we call _extract_hankel_time() passing t = 5,
        the function should return the tuple (4, 1), because the first
        occurrence of the 5th original snapshot is in the second block (lag=1)
        of the 4th snapshot of the Hankel pre-processed matrix.
        """
        if self._hankel_time_mapping is None:
            msg = "The Hankel time mapping dictionary is not available."
            raise RuntimeError(msg)
        if isinstance(t, str):
            try:
                first_snapshot: tuple = self._hankel_time_mapping[np.datetime64(t)]
            except Exception as e:
                msg = f"Can't find snapshot {t} in the data's original time vector."
                logger.exception(msg)
                raise ValueError(msg) from e
            return first_snapshot[0].astype(str), first_snapshot[1]
        if t < 0:
            if abs(t) > len(self._hankel_time_mapping) - self._hankel_d + 1:
                msg = "Negative index is out of bounds. Try using a positive index."
                logger.exception(msg)
                raise ValueError(msg)
            return t, self._hankel_d - 1
        if t - self._hankel_d < 0:
            return 0, t
        return t - self._hankel_d + 1, self._hankel_d - 1

    def _estimate_array_size(self, t: np.ndarray) -> int:
        """Given an input time vector, estimate the size of the array
        resulting from the corresponding DMD reconstruction or forecast."""
        if self._modes is None or self._eigs is None or self._amplitudes is None:
            msg = "Results of the OptDMD fit are not available."
            logger.exception(msg)
            raise RuntimeError(msg)
        array_shape = (self._modes.shape[0], len(t))
        dtype = np.result_type(
            self._modes,
            self._amplitudes,
            np.exp(np.outer(self._eigs, t)),
        )
        return (np.prod(array_shape) * np.dtype(dtype).itemsize).item()

    def _apply_hankel_postprocessing(
        self,
        prediction: np.ndarray
        | da.Array
        | tuple[np.ndarray, np.ndarray]
        | tuple[da.Array, da.Array],
        lags: tuple[int] | tuple[int, int] | None = None,
    ) -> (
        np.ndarray
        | da.Array
        | tuple[np.ndarray, np.ndarray]
        | tuple[da.Array, da.Array]
    ):
        """Helper function to call _extract_hankel_prediction() on a DMD
        prediction (a reconstruction or a forecast), deterministic or
        probabilistic (an array or a tuple of arrays), on which Hankel
        pre-processing has been applied. The function will perform rechunking
        along the columns (i.e. the temporal dimension) if the prediction is
        a Dask array.
        """
        if isinstance(prediction, tuple):
            preds = []
            for p in prediction:
                pred = self._extract_hankel_prediction(
                    p,
                    self._hankel_d,
                    lags,
                )
                pred = (
                    self._rechunk_along_columns(pred)
                    if isinstance(pred, da.Array)
                    else pred
                )
                preds.append(pred)
            return cast(
                tuple[da.Array, da.Array] | tuple[np.ndarray, np.ndarray], tuple(preds)
            )
        pred = self._extract_hankel_prediction(
            prediction,
            self._hankel_d,
            lags,
        )
        return self._rechunk_along_columns(pred) if isinstance(pred, da.Array) else pred

    def forecast(
        self,
        forecast_span: str | int,
        dt: str | int | None = None,
        memory_limit_bytes: float = 1e9,
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """Generates a forecast using the fitted OptDMD model over a specified
        time span. The model must be fitted before calling this method.

        Parameters
        ----------
        forecast_span: str | int
            The total time span for the forecast. If int, interpreted as time
            in the model's time units. If str, should be in the format
            "value units", e.g. "30 D" for 30 days.
        dt: str | int | None, optional
            The time step or number of time points for the forecast. If str,
            should be in the format "value units", e.g. "1 h" for 1 hour.
            If int, interpreted as number of forecast points. If None, uses
            the average time step of the training data on which the DMD model
            was fitted. The default is None.
        memory_limit_bytes: float, optional
            The memory threshold that decides whether to compute the forecast
            using NumPy or Dask. If the estimated array size is below
            memory_limit_bytes, NumPy is used and the forecast is returned as
            a NumPy-backed Xarray. If the estimated array size is above
            memory_limit_bytes, Dask is used and the forecast is returned as
            a Dask-backed Xarray. The default value is 1e9 (1 GB).

        Returns
        -------
        xarray.DataArray | tuple[xarray.DataArray, xarray.DataArray]
            The forecasted data as an xarray.DataArray. If bagging is used,
            two xarray.DataArray are returned where the first one is the
            ensemble mean and the second one is the ensemble variance. The
            Xarrays are NumPy-backed or Dask-backed depending on the
            'memory_limit_bytes' parameter.
        """
        if self._solver is None:
            msg = "The OptDMD model must be fitted before forecasting."
            logger.exception(msg)
            raise RuntimeError(msg)
        try:
            t_forecast, time_forecast = self._generate_forecast_time_vector(
                forecast_span, dt
            )
        except Exception as e:
            msg = "Error trying to generate the forecast time vector."
            logger.exception(msg)
            raise RuntimeError(msg) from e

        estimated_size = self._estimate_array_size(t_forecast)
        msg = f"Estimated forecast size is {estimated_size/1e3:.3f} KB."
        logger.info(msg)

        if estimated_size > memory_limit_bytes:
            logger.info("Will use Dask to compute the forecast.")
            use_dask = True
        else:
            logger.info("Will not use Dask to compute the forecast.")
            use_dask = False

        logger.info("Computing the DMD forecast...")
        try:
            forecast = self._predict(t_forecast, use_dask)
        except Exception as e:
            msg = "Error computing the DMD forecast."
            logger.exception(msg)
            raise RuntimeError(msg) from e
        logger.info("Done.")

        if self._hankel_d > 1:
            try:
                # note that when Hankel preprocessing has been applied,
                # we always extract the bottom row block of the forecast
                # Hankel matrix, i.e. the rows with lag index == hankel_d - 1.
                # This is because all previous lags of the first matrix column
                # were already part of the fitted dataset.
                forecast = self._apply_hankel_postprocessing(
                    forecast, lags=(self._hankel_d - 1,)
                )
            except Exception as e:
                msg = "Error extracting the forecast from the Hankel matrix."
                logger.exception(msg)
                raise RuntimeError(msg) from e
        try:
            return self._prediction_to_dataarray(forecast, time_forecast)
        except Exception as e:
            msg = "Error trying to convert forecast into xarray.DataArray."
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def _generate_reconstruct_time_vector(
        self,
        t: slice | int | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, (tuple | None)]:
        """Given the user-provided time span over which to produce
        the reconstruction, generate the time vector required to compute
        the reconstruction, the true time vector of this reconstruction,
        and the corresponding lag indices of the Hankel matrix (if
        Hankel pre-processing has been performed).
        """
        if self._time_fit is None or self._t_fit is None:
            msg = "The OptDMD fit time vector is not available."
            raise RuntimeError(msg)

        def normalize_datetime_string(s: slice | str) -> slice | str:
            """Reformat datetime strings to contain second-level precision."""
            if isinstance(s, slice):
                start = np.datetime64(s.start, "s").astype(str) if s.start else None
                stop = np.datetime64(s.stop, "s").astype(str)
                return slice(start, stop)
            return np.datetime64(s, "s").astype(str)

        def check_slice_type(t: slice) -> str:
            """Check if the slice is index or label based."""
            if (isinstance(t.start, int) or t.start is None) and isinstance(
                t.stop, int
            ):
                return "index"
            if (isinstance(t.start, str) or t.start is None) and isinstance(
                t.stop, str
            ):
                return "label"
            msg = "Slice must contain time coordinate indices or labels."
            logger.exception(msg)
            raise ValueError(msg)

        # if the requested time span is label based, reformat it to second-level
        # precision
        if (isinstance(t, slice) and check_slice_type(t) == "label") or isinstance(
            t, str
        ):
            t = normalize_datetime_string(t)

        # If Hankel pre-processing has been applied:
        # convert the requested time span in the original time vector frame of
        # reference to the corresponding span in the Hankel time vector, and
        # obtain the associated lag indices of the Hankel pre-processed matrix
        lags: tuple[int, int] | tuple[int] | None = None
        t_original: slice | int | str | None = None
        if self._hankel_d > 1:
            t_original = t
            if isinstance(t, slice):
                t_start, lag_start = (
                    self._extract_hankel_time(t.start) if t.start else (None, 0)
                )
                t_stop, lag_stop = self._extract_hankel_time(t.stop)
                t = slice(t_start, t_stop)
                lags = (lag_start, lag_stop) if lag_start != lag_stop else (lag_start,)
            elif t is not None:
                t, lag_single = self._extract_hankel_time(t)
                lags = (lag_single,)

        # now we can compute the reconstruction time vector
        time_reconstruct_original: np.ndarray = np.empty(0)
        if (isinstance(t, slice) and check_slice_type(t) == "label") or isinstance(
            t, str
        ):
            time_fit = xr.DataArray(
                self._time_fit, dims="time", coords={"time": self._time_fit}
            )
            time_reconstruct = time_fit.sel(time=t).values
            if self._hankel_d > 1:
                if self._time_fit_original is None or t_original is None:
                    msg = (
                        "Required time vector information "
                        "for Hankel post-processing is missing."
                    )
                    raise RuntimeError(msg)
                time_fit_original = xr.DataArray(
                    self._time_fit_original,
                    dims="time",
                    coords={"time": self._time_fit_original},
                )
                time_reconstruct_original = time_fit_original.sel(
                    time=t_original
                ).values
        elif (isinstance(t, slice) and check_slice_type(t) == "index") or isinstance(
            t, int
        ):
            time_reconstruct = self._time_fit[t]
            if self._hankel_d > 1:
                if self._time_fit_original is None or t_original is None:
                    msg = (
                        "Required time vector information "
                        "for Hankel post-processing is missing."
                    )
                    raise RuntimeError(msg)
                time_reconstruct_original = self._time_fit_original[t_original]
        elif t is None:
            time_reconstruct = self._time_fit
            if self._hankel_d > 1:
                if self._time_fit_original is None:
                    msg = (
                        "Required time vector information "
                        "for Hankel post-processing is missing."
                    )
                    raise RuntimeError(msg)
                time_reconstruct_original = self._time_fit_original
        else:
            msg = "Parameter 't' must be a slice, an integer, a string or None."
            logger.exception(msg)
            raise ValueError(msg)
        _, ind, _ = np.intersect1d(
            self._time_fit, time_reconstruct, return_indices=True
        )
        t_reconstruct = self._t_fit[ind]

        if self._hankel_d > 1:
            return (
                np.atleast_1d(t_reconstruct),
                np.atleast_1d(time_reconstruct_original),
                lags,
            )
        return np.atleast_1d(t_reconstruct), np.atleast_1d(time_reconstruct), lags

    def reconstruct(
        self,
        t: slice | int | str | None = None,
        memory_limit_bytes: float = 1e9,
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """Produces a reconstruction of the training data using the fitted
        OptDMD model over a specified time span. The model must be fitted
        before calling this method.

        Parameters
        ----------
        t: slice | int | str | None, optional
            The time span over which to perform the DMD reconstruction. If you
            have performed Hankel pre-processing, this time should nevertheless
            be expressed in the original data's time vector frame of reference,
            not the Hankel pre-processed data.
            If 't' is a slice, it may contain integers or strings, where integers
            and strings are interpreted as starting and stopping indices or labels
            of the fit time vector, respectively. To reconstruct a single snapshot,
            't' should be an integer or string specifying the index or label of the
            desired snapshot. If 't' is None, a reconstruction of the whole training
            dataset is produced, which could potentially be very large. The default is
            None.
        memory_limit_bytes: float, optional
            The memory threshold that decides whether to compute the reconstruction
            using NumPy or Dask. If the estimated array size is below
            memory_limit_bytes, NumPy is used and the reconstruction is returned as
            a NumPy-backed Xarray. If the estimated array size is above
            memory_limit_bytes, Dask is used and the reconstruction is returned as
            a Dask-backed Xarray. The default value is 1e9 (1 GB).

        Returns
        -------
        xarray.DataArray | tuple[xarray.DataArray, xarray.DataArray]
            The reconstructed data as an xarray.DataArray. If bagging is used,
            two xarray.DataArray are returned where the first one is the
            ensemble mean and the second one is the ensemble variance. The
            arrays are NumPy-backed or Dask-backed depending on the
            'memory_limit_bytes' parameter.

        Examples
        --------
        Given optdmd, a fitted OptDMD instance:

        Produce a reconstruction of the training data for Dec 2020:
        >>> optdmd.reconstruct(slice("2020-12-01", "2020-12-31"))

        Produce a reconstruction of the 10th snapshot of the training data.
        If you had performed Hankel pre-processing, this would still refer
        to the 10th snapshot of the original data:
        >>> optdmd.reconstruct(10)

        Reconstruct the whole training dataset, which could be huge:
        >>> optdmd.reconstruct()
        """
        if self._solver is None:
            msg = "The OptDMD model must be fitted before reconstructing."
            logger.exception(msg)
            raise RuntimeError(msg)

        try:
            t_reconstruct, time_reconstruct, lags = (
                self._generate_reconstruct_time_vector(t)
            )
        except Exception as e:
            msg = "Error trying to generate the reconstruction time vector."
            logger.exception(msg)
            raise RuntimeError(msg) from e

        estimated_size = self._estimate_array_size(t_reconstruct)
        msg = f"Estimated reconstruction size is {estimated_size/1e3:.3f} KB."
        logger.info(msg)

        # compute the reconstruction, using Dask or directly with NumPy
        if estimated_size > memory_limit_bytes:
            logger.info("Will use Dask to compute the reconstruction.")
            use_dask = True
        else:
            logger.info("Will not use Dask to compute the reconstruction.")
            use_dask = False

        logger.info("Computing the DMD reconstruction...")
        try:
            reconstruction = self._predict(t_reconstruct, use_dask)
        except Exception as e:
            msg = "Error computing the DMD reconstruction."
            logger.exception(msg)
            raise RuntimeError(msg) from e
        logger.info("Done.")

        # if Hankel preprocessing has been used, extract the
        # relevant snapshots from the computed reconstruction matrix
        if self._hankel_d > 1:
            try:
                reconstruction = self._apply_hankel_postprocessing(
                    reconstruction,
                    lags,
                )
            except Exception as e:
                msg = "Error extracting the reconstuction from the Hankel matrix."
                logger.exception(msg)
                raise RuntimeError(msg) from e

        # finally, transform the computed reconstruction into a labelled DataArray
        try:
            return self._prediction_to_dataarray(reconstruction, time_reconstruct)
        except Exception as e:
            msg = "Error trying to convert reconstruction into xarray.DataArray."
            logger.exception(msg)
            raise RuntimeError(msg) from e
