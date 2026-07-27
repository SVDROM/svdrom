Weather Utilities
=================

The weather utilities module provides metrics and tools commonly used in weather and climate analysis, following conventions from `WeatherBench2 <https://github.com/google-research/weatherbench2>`_.

.. py:module:: weather_utils

.. py:function:: compute_rmse(ground_truth, prediction, lat_weighting=True, dims=('latitude', 'longitude'))

    Compute the Root Mean Squared Error (RMSE) between a ground truth and a prediction.

    :param ground_truth: The ground truth data.
    :type ground_truth: xarray.DataArray
    :param prediction: The prediction (a reconstruction or a forecast). Must be defined on the same spatio-temporal grid as *ground_truth*.
    :type prediction: xarray.DataArray
    :param lat_weighting: Whether to apply latitude-based weighting so that spatial locations closer to the Equator receive a larger weight than those closer to the poles. Default is ``True``.
    :type lat_weighting: bool
    :param dims: Dimensions along which to average the RMSE. Default is ``("latitude", "longitude")``, which returns the RMSE as a function of prediction time (assuming a single pressure level).
    :type dims: str or tuple[str, ...]
    :returns: The RMSE as a DataArray.
    :rtype: xarray.DataArray
    :raises ValueError: If the input arrays are not defined on the same spatio-temporal grid, or if *lat_weighting* is ``True`` but the input data has no ``"latitude"`` dimension.

    .. rubric:: Notes

    If the inputs are Dask-backed ``DataArray`` objects, the function builds the task graph lazily. Trigger computation manually (e.g. by calling ``.compute()``). It is recommended to set up a multi-threading Dask cluster before calling the function.

.. py:function:: compute_mae(ground_truth, prediction, lat_weighting=True, dims=('latitude', 'longitude'))

    Compute the Mean Absolute Error (MAE) between a ground truth and a prediction.

    :param ground_truth: The ground truth data.
    :type ground_truth: xarray.DataArray
    :param prediction: The prediction (a reconstruction or a forecast). Must be defined on the same spatio-temporal grid as *ground_truth*.
    :type prediction: xarray.DataArray
    :param lat_weighting: Whether to apply latitude-based weighting so that spatial locations closer to the Equator receive a larger weight than those closer to the poles. Default is ``True``.
    :type lat_weighting: bool
    :param dims: Dimensions along which to average the MAE. Default is ``("latitude", "longitude")``, which returns the MAE as a function of prediction time.
    :type dims: str or tuple[str, ...]
    :returns: The MAE as a DataArray.
    :rtype: xarray.DataArray
    :raises ValueError: If the input arrays are not defined on the same spatio-temporal grid, or if *lat_weighting* is ``True`` but the input data has no ``"latitude"`` dimension.

    .. rubric:: Notes

    If the inputs are Dask-backed ``DataArray`` objects, the function builds the task graph lazily. Trigger computation manually (e.g. by calling ``.compute()``). It is recommended to set up a multi-threading Dask cluster before calling the function.

.. py:function:: compute_climatology(data, smooth_window=61, probabilistic=False)

    Compute the climatology as a function of day of year and hour of day.

    :param data: A Dask- or NumPy-backed DataArray containing the observed data. Must have a ``"time"`` dimension.
    :type data: xarray.DataArray
    :param smooth_window: The size (in days) of a sliding window around each day-of-year/hour combination, with weights linearly decaying to zero from the centre, used to compute a rolling weighted mean. This removes sample noise and results in a smoother climatology. Should be an odd number. Set to ``None`` to skip smoothing. Default is ``61`` (same as WeatherBench2).
    :type smooth_window: int or None
    :param probabilistic: If ``True``, also compute and return the standard deviation of the climatology (across years, for each day-of-year/hour-of-day group). The same smoothing is applied to the standard deviation as to the mean. Default is ``False``.
    :type probabilistic: bool
    :returns: A DataArray containing the calculated climatology as a function of day of year and hour of day. If ``probabilistic=True``, returns a tuple ``(mean_climatology, std_climatology)``.
    :rtype: xarray.DataArray or tuple[xarray.DataArray, xarray.DataArray]

    .. rubric:: Notes

    If the input data is Dask-backed, it is recommended to set up a multi-threading Dask cluster before calling the function. Performing the smoothing via the rolling weighted average is considerably more expensive and a large amount of disk spillage from Dask is expected.

    .. rubric:: Examples

    .. code-block:: python

        from dask.distributed import Client
        from svdrom.weather_utils import compute_climatology

        client = Client(processes=False)
        climatology = compute_climatology(era5)
        climatology = climatology.sel(dayofyear=slice(1, 60))
        climatology = climatology.compute()
        client.close()

.. py:function:: expand_time_climatology(climatology, year)

    Convert a climatology indexed by day of year and hour of day into a climatology indexed by a single time dimension.

    :param climatology: Climatology as a function of day of year and hour of day (e.g. as returned by :py:func:`compute_climatology`). Must contain dimensions ``"dayofyear"`` and ``"hour"``.
    :type climatology: xarray.DataArray
    :param year: The year used to construct the new time dimension.
    :type year: int
    :returns: The climatology with the ``dayofyear`` and ``hour`` dimensions converted into a single ``time`` dimension. The output DataArray is NumPy-backed.
    :rtype: xarray.DataArray

    .. rubric:: Notes

    If the input DataArray is Dask-backed, the function will trigger the computation of the underlying task graph and load the data into memory.

    .. rubric:: Examples

    .. code-block:: python

        from svdrom.weather_utils import compute_climatology, expand_time_climatology

        climatology = compute_climatology(era5)
        climatology_2020 = expand_time_climatology(climatology, year=2020)

.. py:function:: compute_energy_spectrum(data, lat_range=(30, 60))

    Compute the zonal energy spectrum along lines of constant latitude as a function of wavenumber, frequency, and wavelength. The spectrum is computed using WeatherBench2.

    .. note::
        This function requires the optional ``weather`` dependencies. Install them with: ``pip install "svdrom[weather]"``.

    :param data: The variable on a lat/lon grid for which to compute the energy spectrum. Must be backed by a NumPy array.
    :type data: xarray.DataArray
    :param lat_range: Latitude range over which to perform an average of the energy spectrum. Default is ``(30, 60)``, meaning the output is the average for :math:`30° < |\text{lat}| < 60°`. If ``None``, no averaging is performed and the output is given as a function of wavenumber.
    :type lat_range: tuple[int, int] or None
    :returns: A NumPy-backed DataArray containing the computed zonal energy spectrum. If *lat_range* is ``None``, the spectrum is returned as a function of latitude and wavenumber. If *lat_range* is a tuple, the spectrum is returned as a function of frequency (1/m).
    :rtype: xarray.DataArray

    .. rubric:: Notes

    The input DataArray must not be backed by a Dask array. Select a relatively small slice and load it into memory before calling.

    .. rubric:: Examples

    .. code-block:: python

        from svdrom.weather_utils import compute_energy_spectrum

        temperature = era5["temperature"].sel(time="2020-01-01")
        temperature.load()
        spectrum = compute_energy_spectrum(temperature)

.. py:function:: compute_crps_gaussian(ground_truth, prediction_mean, prediction_std, lat_weighting=True, dims=('latitude', 'longitude'))

    Compute the Continuous Ranked Probability Score (CRPS) for a probabilistic forecast, assuming a Gaussian distribution.

    .. note::
        This function requires the optional ``weather`` dependencies. Install them with: ``pip install "svdrom[weather]"``.

    :param ground_truth: The ground truth data. Must be a NumPy-backed array.
    :type ground_truth: xarray.DataArray
    :param prediction_mean: The mean of the predictions ensemble. Must be a NumPy-backed array.
    :type prediction_mean: xarray.DataArray
    :param prediction_std: The standard deviation of the predictions ensemble. Must be a NumPy-backed array with only positive values.
    :type prediction_std: xarray.DataArray
    :param lat_weighting: Whether to apply latitude-based weighting. Default is ``True``.
    :type lat_weighting: bool
    :param dims: Dimensions along which to average the CRPS. Default is ``("latitude", "longitude")``. Set to ``None`` to skip averaging.
    :type dims: str or tuple[str, ...] or None
    :returns: The CRPS as a NumPy-backed DataArray.
    :rtype: xarray.DataArray

    .. rubric:: Notes

    All input arrays must be NumPy-backed. CRPS is calculated using ``crps_gaussian()`` from the `properscoring <https://github.com/properscoring/properscoring>`_ library.

.. py:function:: compute_acc(ground_truth, prediction, climatology)

    Compute the Anomaly Correlation Coefficient (ACC).

    :param ground_truth: The ground truth data. Can be Dask-backed or NumPy-backed.
    :type ground_truth: xarray.DataArray
    :param prediction: The prediction (a reconstruction or a forecast). Must be defined on the same spatio-temporal grid as *ground_truth*. Can be Dask-backed or NumPy-backed.
    :type prediction: xarray.DataArray
    :param climatology: The climatology, defined on the same spatio-temporal grid as the ground truth and prediction. Must have a ``time`` dimension matching the other arrays. Can be Dask-backed or NumPy-backed.
    :type climatology: xarray.DataArray
    :returns: The Anomaly Correlation Coefficient as a latitude-weighted normalized cross-correlation, reduced over latitude and longitude and returned as a function of time.
    :rtype: xarray.DataArray

    .. rubric:: Notes

    If your climatology is defined as a function of day of year and hour of day (such as the one returned by :py:func:`compute_climatology`), you can use :py:func:`expand_time_climatology` to convert it into a function of time.

    .. rubric:: Examples

    .. code-block:: python

        from svdrom.weather_utils import compute_acc, compute_climatology, expand_time_climatology

        climatology = compute_climatology(era5_train)
        climatology_2020 = expand_time_climatology(climatology, year=2020)
        acc = compute_acc(era5_test, forecast, climatology_2020)
