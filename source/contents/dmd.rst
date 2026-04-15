DMD Module
==========

The DMD module provides access to the ``OptDMD`` class, which provides an API for computing the Optimized Dynamic Mode Decomposition.

.. py:module:: dmd

.. py:class:: OptDMD(n_modes=-1, time_dimension='time', time_units='s', input_time_units=None, num_trials=0, trial_size=0.6, parallel_bagging=False, seed=None)

    Optimized Dynamic Mode Decomposition (DMD) via variable projection method for nonlinear least squares, with optional bootstrap aggregation (bagging) for uncertainty quantification.

    This class makes use of the ``BOPDMD`` class from the `PyDMD library <https://pydmd.github.io/PyDMD/bopdmd.html>`_.

    .. note::
        This class is a wrapper of the ``BOPDMD.fit_econ()`` method, which fits
        an approximate Optimized DMD on an array X by operating on the SVD of X.

    :param n_modes: Number of DMD modes to compute. Must be a positive integer
        or ``-1`` to use all available modes. Default is ``-1``.
    :type n_modes: int
    :param time_dimension: Name of the time dimension in the input data.
        Default is ``"time"``.
    :type time_dimension: str
    :param time_units: Units in which to treat the time dimension. Must be one
        of ``{"s", "h"}``, where ``"s"`` is seconds and ``"h"`` is hours.
        Default is ``"s"``.
    :type time_units: str
    :param input_time_units: Units of the time vector of the input data. If
        ``None``, defaults to the value of *time_units*. If provided, must be
        one of ``{"s", "h"}``. Only used when the input time vector consists
        of floats rather than datetimes. Default is ``None``.
    :type input_time_units: str or None
    :param num_trials: Number of bagging trials to perform during the OptDMD
        fit. Default is ``0`` (no bagging).
    :type num_trials: int
    :param trial_size: Size of the randomly selected snapshot subset used for
        each bagging trial. A positive integer specifies the exact number of
        snapshots; a float in ``(0, 1)`` specifies the fraction of snapshots.
        Default is ``0.6``.
    :type trial_size: int or float
    :param parallel_bagging: Whether to perform bagging in parallel (``True``)
        or sequentially (``False``). Only active when *num_trials* is greater
        than zero. Parallel bagging requires a Dask multi-processing or
        distributed scheduler. Default is ``False``.
    :type parallel_bagging: bool
    :param seed: Random seed for the random number generator. Set to a fixed
        integer for reproducible bagging results. ``None`` initialises the
        generator from system entropy. Default is ``None``.
    :type seed: int or None

    .. rubric:: Attributes

    .. py:property:: n_modes
        :type: int

        Number of DMD modes (read-only).

    .. py:property:: time_dimension
        :type: str

        Name of the time dimension (read-only).

    .. py:property:: time_units
        :type: str

        Time units used in the DMD fit and forecast (read-only).

    .. py:property:: input_time_units
        :type: str or None

        Units of the time vector of the input data (read-only).

    .. py:property:: num_trials
        :type: int

        Number of bagging trials (read-only).

    .. py:property:: trial_size
        :type: int or float

        Bagging trial size (read-only).

    .. py:property:: parallel_bagging
        :type: bool

        Whether bagging is performed in parallel or sequentially (read-only).

    .. py:property:: eigs
        :type: numpy.ndarray or None

        DMD eigenvalues (read-only). ``None`` before the model is fitted.

    .. py:property:: eigs_std
        :type: numpy.ndarray or None

        Standard deviation of the DMD eigenvalues when using bagging
        (read-only). ``None`` before the model is fitted or when bagging is
        not used.

    .. py:property:: modes
        :type: xarray.DataArray or None

        DMD modes (read-only). ``None`` before the model is fitted. When
        Hankel pre-processing has been applied the modes span the stacked
        Hankel space; use :py:attr:`modes_averaged` for a spatially
        interpretable result.

    .. py:property:: modes_averaged
        :type: xarray.DataArray or None

        DMD modes averaged across Hankel time lags (read-only). Equivalent
        to :py:attr:`modes` when no Hankel pre-processing has been applied.
        Prefer this attribute over :py:attr:`modes` for visualisation when
        Hankel pre-processing is used. ``None`` before the model is fitted.

    .. py:property:: modes_std
        :type: xarray.DataArray or None

        Standard deviation of the DMD modes when using bagging (read-only).
        ``None`` before the model is fitted or when bagging is not used.

    .. py:property:: modes_std_averaged
        :type: xarray.DataArray or None

        Standard deviation of the DMD modes averaged across Hankel time lags
        (read-only). Equivalent to :py:attr:`modes_std` when no Hankel
        pre-processing has been applied. ``None`` before the model is fitted
        or when bagging is not used.

    .. py:property:: amplitudes
        :type: numpy.ndarray or None

        DMD mode amplitudes (read-only). ``None`` before the model is fitted.

    .. py:property:: amplitudes_std
        :type: numpy.ndarray or None

        Standard deviation of the DMD mode amplitudes when using bagging
        (read-only). ``None`` before the model is fitted or when bagging is
        not used.

    .. py:property:: dynamics
        :type: xarray.DataArray or None

        Time evolution of each DMD mode, scaled by the mode amplitude
        (read-only). ``None`` before the model is fitted.

    .. py:property:: solver
        :type: pydmd.BOPDMD or None

        Underlying ``BOPDMD`` solver instance (read-only). ``None`` before
        the model is fitted.

    .. py:property:: time_fit
        :type: numpy.ndarray or None

        Time vector of the fitted (training) data (read-only). When Hankel
        pre-processing has been applied this corresponds to the time vector of
        the preprocessed data, not the original time vector. ``None`` before
        the model is fitted.

    .. py:property:: time_fit_original
        :type: numpy.ndarray or None

        Original time vector of the training data when Hankel pre-processing
        has been applied (read-only). ``None`` before the model is fitted or
        when no Hankel pre-processing has been used.

    .. py:property:: hankel_d
        :type: int

        Hankel matrix rank used during time-delay embedding pre-processing
        (read-only). Equals ``1`` when no Hankel pre-processing has been
        applied.

    .. rubric:: Methods

    .. py:method:: fit(u, s, v, **kwargs)

        Fit the OptDMD model to the results of a Singular Value Decomposition
        (SVD). When bagging has been requested, *num_trials* OptDMD trials are
        fitted on randomly subsampled snapshots and ensemble averaged. If
        parallel bagging has been requested and a Dask multi-processing or
        distributed scheduler is active, the trials are computed in parallel.

        To perform a DMD fit on a Hankel pre-processed (time-delay embedding)
        matrix, *u*, *s*, and *v* must come from the SVD of the Hankel
        pre-processed matrix produced by the ``TruncatedSVD`` class.

        :param u: Left singular vectors containing the spatial information,
            with shape ``(n_spatial_points, n_components)``.
        :type u: xarray.DataArray
        :param s: Singular values, with shape ``(n_components,)``.
        :type s: numpy.ndarray
        :param v: Right singular vectors containing the temporal information,
            with shape ``(n_components, n_timesteps)``.
        :type v: xarray.DataArray
        :param \**kwargs: Additional keyword arguments forwarded to PyDMD's
            ``BOPDMD`` constructor. See
            `BOPDMD docs <https://pydmd.github.io/PyDMD/bopdmd.html>`_ for
            details.
        :returns: The fitted ``OptDMD`` instance (``self``).
        :rtype: OptDMD

    .. py:method:: forecast(forecast_span, dt=None, memory_limit_bytes=1e9)

        Generate a forecast using the fitted OptDMD model over a specified
        time span. The model must be fitted before calling this method.

        :param forecast_span: Total time span for the forecast. If an
            ``int``, interpreted as time in the model's *time_units*. If a
            ``str``, must be in the format ``"value units"``, e.g.
            ``"30 D"`` for 30 days.
        :type forecast_span: str or int
        :param dt: Time step (or number of time points) for the forecast.
            If a ``str``, must be in the format ``"value units"``, e.g.
            ``"1 h"`` for 1 hour. If an ``int``, interpreted as the number
            of forecast points. If ``None``, the average training data time
            step is used. Default is ``None``.
        :type dt: str, int, or None
        :param memory_limit_bytes: Memory threshold (bytes) that determines
            whether the forecast is computed with NumPy or Dask. Arrays
            estimated to exceed this limit are computed with Dask and
            returned as Dask-backed Xarrays; otherwise NumPy is used.
            Default is ``1e9`` (1 GB).
        :type memory_limit_bytes: float
        :returns: The forecasted data. When bagging is used, a tuple of two
            ``xarray.DataArray`` is returned — the first is the ensemble mean
            and the second is the ensemble variance.
        :rtype: xarray.DataArray or tuple[xarray.DataArray, xarray.DataArray]

        .. rubric:: Examples

        Given ``optdmd``, a fitted ``OptDMD`` instance:

        Produce a 45 day forecast using the time step of the training data:

        .. code-block:: python

            optdmd.forecast("45 D")

        Produce a 45 day forecast using a 12 hour time step:

        .. code-block:: python

            optdmd.forecast("45 D", dt="12 h")

    .. py:method:: reconstruct(t=None, memory_limit_bytes=1e9)

        Produce a reconstruction of the training data using the fitted OptDMD
        model over a specified time span. The model must be fitted before
        calling this method.

        When Hankel pre-processing has been applied, *t* should still be
        expressed in the original data's time vector frame of reference, not
        the Hankel pre-processed frame.

        :param t: Time span over which to perform the reconstruction.

            * ``slice`` — start and stop may be integer indices or string
              labels of the fit time vector.
            * ``int`` or ``str`` — reconstructs a single snapshot by index
              or label.
            * ``None`` — reconstructs the entire training dataset (can be
              very large).

            Default is ``None``.
        :type t: slice, int, str, or None
        :param memory_limit_bytes: Memory threshold (bytes) that determines
            whether the reconstruction is computed with NumPy or Dask. Arrays
            estimated to exceed this limit are computed with Dask and returned
            as Dask-backed Xarrays; otherwise NumPy is used. Default is
            ``1e9`` (1 GB).
        :type memory_limit_bytes: float
        :returns: The reconstructed data. When bagging is used, a tuple of
            two ``xarray.DataArray`` is returned — the first is the ensemble
            mean and the second is the ensemble variance.
        :rtype: xarray.DataArray or tuple[xarray.DataArray, xarray.DataArray]

        .. rubric:: Examples

        Given ``optdmd``, a fitted ``OptDMD`` instance:

        Reconstruct the training data for December 2020:

        .. code-block:: python

            optdmd.reconstruct(slice("2020-12-01", "2020-12-31"))

        Reconstruct the 10th snapshot of the training data (still refers to
        the original time vector even when Hankel pre-processing was used):

        .. code-block:: python

            optdmd.reconstruct(10)

        Reconstruct the entire training dataset:

        .. code-block:: python

            optdmd.reconstruct()
