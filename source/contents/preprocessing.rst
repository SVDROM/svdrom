Preprocessing Module
====================

The preprocessing module provides utilities for preparing data before SVD or DMD analysis, including spatial stacking, standard scaling, and Hankel (time-delay) embedding.

.. py:module:: preprocessing

.. py:function:: variable_spatial_stack(X, dims)

    Stack multiple dimensions of an input Xarray Dataset or DataArray into a single new dimension. This is typically used to collapse spatial dimensions (e.g. latitude and longitude) into a single ``"samples"`` dimension suitable for SVD.

    If the input is a ``Dataset`` containing multiple variables, all variables are first converted to a ``DataArray`` with an extra ``"variable"`` dimension, and then stacked together.

    You can change the default name of the resulting stacked dimension using :py:func:`config.set`:

    .. code-block:: python

        import svdrom.config
        svdrom.config.set(stack_coord_name="space")

    :param X: The input data to stack.
    :type X: xarray.Dataset or xarray.DataArray
    :param dims: The dimensions to stack together into a single spatial dimension (e.g. ``("latitude", "longitude")``).
    :type dims: Sequence[str]
    :returns: The array with the specified dimensions stacked into a single spatial dimension.
    :rtype: xarray.DataArray

    .. rubric:: Notes

    The returned xarray object is Dask-backed and lazy if the input is Dask-backed.

.. py:class:: StandardScaler

    Preprocessing class for scaling Xarray Datasets or DataArrays by removing the mean and optionally scaling by the standard deviation along a specified dimension.

    .. rubric:: Attributes

    .. py:property:: mean
        :type: xarray.DataArray or xarray.Dataset or None

        The mean values computed along the specified dimension (read-only).

    .. py:property:: std
        :type: xarray.DataArray or xarray.Dataset or None

        The standard deviation values computed along the specified dimension (read-only).

    .. py:property:: with_std
        :type: bool

        Whether the data has been scaled to unit variance (read-only).

    .. py:method:: __call__(X, dim='time', with_std=False)

        Scale the input xarray object by removing the mean and optionally dividing by the standard deviation.

        :param X: The input xarray object to be scaled.
        :type X: xarray.Dataset or xarray.DataArray
        :param dim: The dimension along which to compute the mean and standard deviation. Default is ``"time"``.
        :type dim: str
        :param with_std: If ``True``, scales the data by dividing by the standard deviation after subtracting the mean. Default is ``False``.
        :type with_std: bool
        :returns: The scaled xarray object with the mean removed, and optionally divided by the standard deviation.
        :rtype: xarray.Dataset or xarray.DataArray

        .. rubric:: Notes

        The mean and standard deviation are computed eagerly and stored as NumPy-backed xarray objects. The returned xarray object is Dask-backed and lazy if the input is Dask-backed.

        .. rubric:: Examples

        .. code-block:: python

            from svdrom.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler(X, dim="time", with_std=True)

.. py:function:: hankel_preprocessing(X, d=2)

    Perform Hankel (time-delay) embedding on a 2-D DataArray.

    Given a matrix with dimensions :math:`(m \times n)`, where :math:`m` is the number of samples (e.g. spatial observations) and :math:`n` is the number of snapshots or temporal observations, this function augments the data matrix by appending time-shifted copies of itself. This can help unveil hidden or latent variables from the data matrix.

    :param X: The input array with dimensions :math:`(m \times n)`. The DataArray can be NumPy- or Dask-backed.
    :type X: xarray.DataArray
    :param d: Hankel matrix rank. Must be an integer :math:`\geq 2`. Default is ``2``.
    :type d: int
    :returns: The augmented data matrix with dimensions :math:`((m \cdot d) \times (n - d + 1))`. The returned DataArray has a new coordinate indicating the time-delay lag relative to the current timestamp. It also contains a new attribute (a dictionary) that maps each original snapshot to a tuple consisting of the first time-delay embedded snapshot in which it appears and the corresponding lag index.
    :rtype: xarray.DataArray

    .. rubric:: Notes

    - If the input DataArray is NumPy-backed, the returned DataArray is also NumPy-backed. If it is Dask-backed, the returned DataArray is also Dask-backed.
    - The name of the lag coordinate is controlled by the ``"hankel_coord_name"`` config key (default: ``"hankel_lag"``).
    - The attribute storing the time mapping is named according to the ``"hankel_time_mapping_attr"`` config key (default: ``"hankel_time_mapping"``).

    .. rubric:: Examples

    .. code-block:: python

        from svdrom.preprocessing import hankel_preprocessing

        # Augment with one time-delay (d=2): (m x n) -> (2m x n-1)
        X_hankel = hankel_preprocessing(X, d=2)

        # Augment with three time-delays (d=4): (m x n) -> (4m x n-3)
        X_hankel = hankel_preprocessing(X, d=4)
