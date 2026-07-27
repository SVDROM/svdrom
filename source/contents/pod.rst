POD Module
==========

The POD module provides access to the ``POD`` class, which computes the Proper Orthogonal Decomposition of a spatio-temporal field and supports the Extended POD of a secondary, simultaneously measured field.

.. py:module:: pod

.. py:class:: POD(n_modes, svd_algorithm='tsqr', compute_modes=True, compute_time_coeffs=True, compute_energy_ratio=False, rechunk=False, remove_mean=True, time_dimension='time')

    Proper Orthogonal Decomposition (POD) of a spatio-temporal field, operating directly on Dask-backed Xarray ``DataArray`` objects.

    Given a 2-D snapshot matrix :math:`\mathbf{X} \in \mathbb{R}^{m \times n}` with :math:`m` spatial points and :math:`n` temporal snapshots, POD decomposes the (optionally mean-removed) fluctuating field into a set of orthonormal spatial modes :math:`\boldsymbol{\phi}_j`, an energy associated with each mode, and time coefficients :math:`a_j(t)` describing how each mode evolves in time:

    .. math::

        \mathbf{X}'(\mathbf{x}, t) \approx \sum_{j=1}^{k} \boldsymbol{\phi}_j(\mathbf{x})\, a_j(t),

    where :math:`\mathbf{X}'` is the fluctuating field. Internally the decomposition is computed via a truncated SVD of :math:`\mathbf{X}'` scaled by :math:`1/\sqrt{n}` so that the modal energy does not depend on the number of snapshots. See :doc:`background` for the broader theoretical context.

    ``POD`` subclasses :py:class:`svd.TruncatedSVD` and therefore inherits its SVD back-ends and lazy/eager compute behaviour. The POD spatial modes correspond to the left singular vectors, the mode energies to the squared singular values, and the time coefficients to the singular-value-scaled right singular vectors.

    :param n_modes: Number of POD modes to compute (the truncation rank :math:`k`). Must be strictly less than ``min(n_spatial_points, n_snapshots)``.
    :type n_modes: int
    :param svd_algorithm: SVD back-end used internally. Must be one of ``{"tsqr", "randomized"}``. See :py:class:`svd.TruncatedSVD` for the trade-offs between the two algorithms. Default is ``"tsqr"``.
    :type svd_algorithm: str
    :param compute_modes: Whether to eagerly compute the POD spatial modes during :py:meth:`fit`. If ``False``, :py:attr:`modes` is returned as a lazy Dask collection until :py:meth:`compute_modes` is called. Default is ``True``.
    :type compute_modes: bool
    :param compute_time_coeffs: Whether to eagerly compute the POD time coefficients during :py:meth:`fit`. If ``False``, :py:attr:`time_coeffs` is returned as a lazy Dask collection until :py:meth:`compute_time_coeffs` is called. Default is ``True``.
    :type compute_time_coeffs: bool
    :param compute_energy_ratio: Whether to eagerly compute the ratio of total energy explained by each mode. If ``False``, :py:attr:`explained_energy_ratio` is returned as a lazy Dask collection until :py:meth:`compute_energy_ratio` is called. Default is ``False``.
    :type compute_energy_ratio: bool
    :param rechunk: If ``True`` and *svd_algorithm* is ``"randomized"``, the input array is rechunked to a single chunk along its smallest dimension before the SVD is computed. See :py:class:`svd.TruncatedSVD` for details. Default is ``False``.
    :type rechunk: bool
    :param remove_mean: Whether to remove the temporal mean from the input array before computing the POD modes. The mean is stored and added back automatically in :py:meth:`reconstruct`, and subtracted automatically in :py:meth:`transform`. Default is ``True``.
    :type remove_mean: bool
    :param time_dimension: Name of the dimension in the input array that represents time. The input array is transposed automatically if this dimension is not already along the columns. Default is ``"time"``.
    :type time_dimension: str

    .. rubric:: Attributes

    .. py:property:: modes
        :type: xarray.DataArray or None

        POD spatial modes with shape ``(n_spatial_points, n_modes)`` (read-only). NumPy-backed when *compute_modes* was ``True``, otherwise Dask-backed. ``None`` before the model is fitted.

    .. py:property:: time_coeffs
        :type: xarray.DataArray or None

        POD time coefficients with shape ``(n_modes, n_snapshots)`` (read-only). NumPy-backed when *compute_time_coeffs* was ``True``, otherwise Dask-backed. ``None`` before the model is fitted.

    .. py:property:: energy
        :type: numpy.ndarray or None

        Energy (variance) explained by each POD mode with shape ``(n_modes,)`` (read-only). Equal to the squared singular values. ``None`` before the model is fitted.

    .. py:property:: explained_energy_ratio
        :type: numpy.ndarray, dask.array.Array, or None

        Ratio of total energy explained by each POD mode with shape ``(n_modes,)`` (read-only). NumPy-backed when *compute_energy_ratio* was ``True``, otherwise Dask-backed. ``None`` before the model is fitted.

    .. py:property:: n_components
        :type: int

        Number of POD modes retained in the truncation (read-only). Inherited from :py:class:`svd.TruncatedSVD`.

    .. rubric:: Methods

    .. py:method:: fit(X, **kwargs)

        Fit the POD model to the input data array. The method validates that the requested number of modes is strictly less than ``min(n_spatial_points, n_snapshots)``, transposes *X* so that *time_dimension* lies along the columns, optionally removes the temporal mean, scales by :math:`1/\sqrt{n}`, and dispatches to the inherited truncated SVD. The time coefficients are formed by scaling the right singular vectors by the singular values.

        :param X: Input snapshot matrix with shape ``(n_spatial_points, n_snapshots)``. Must be a Dask-backed ``xarray.DataArray`` containing *time_dimension*.
        :type X: xarray.DataArray
        :param \**kwargs: Additional keyword arguments forwarded to ``dask.array.linalg.svd_compressed`` when *svd_algorithm* is ``"randomized"`` (e.g. ``n_power_iter``, ``n_oversamples``, ``seed``). Ignored when *svd_algorithm* is ``"tsqr"``.
        :returns: The fitted instance (``self``), allowing method chaining.
        :rtype: POD
        :raises ValueError: If *time_dimension* is not a dimension of *X*, or if *n_modes* is not strictly less than ``min(n_spatial_points, n_snapshots)``.

    .. py:method:: transform(X, compute=True)

        Project an input array onto the fitted POD modes, computing the temporal coefficients :math:`\mathbf{\Phi}^{\top} \mathbf{X}'`. The same preprocessing applied during :py:meth:`fit` (mean removal and scaling) is applied automatically, and the input array is transposed if needed so that *time_dimension* lies along the columns.

        :param X: Array to project with shape ``(n_spatial_points, n_snapshots)`` or ``(n_snapshots, n_spatial_points)``. Must have the same spatial dimension as the array on which POD was fitted. Do not remove the mean beforehand; the training-data mean is subtracted automatically.
        :type X: xarray.DataArray
        :param compute: If ``True``, eagerly compute and return a NumPy-backed ``DataArray``. If ``False``, return a lazy Dask-backed ``DataArray`` without triggering computation. Default is ``True``.
        :type compute: bool
        :returns: The temporal coefficients with shape ``(n_modes, n_snapshots)``.
        :rtype: xarray.DataArray

    .. py:method:: reconstruct(snapshot=None, snapshot_dim=None, memory_limit_bytes=1e9)

        Reconstruct one, many, or all snapshots from the truncated POD decomposition. The reconstruction is delegated to :py:meth:`svd.TruncatedSVD.reconstruct` and the POD preprocessing is then undone (the result is rescaled by :math:`\sqrt{n}` and, when *remove_mean* was ``True``, the stored temporal mean is added back).

        :param snapshot: Selection along *snapshot_dim*. ``None`` reconstructs the entire training dataset; an ``int`` selects a single snapshot by positional index; a ``str`` selects all snapshots whose coordinate label matches; a ``slice`` selects a range (integer or ``None`` bounds are positional, string bounds are label-based). Default is ``None``.
        :type snapshot: slice, int, str, or None
        :param snapshot_dim: Name of the dimension along which snapshots are indexed. If ``None``, the model's *time_dimension* is used. Default is ``None``.
        :type snapshot_dim: str or None
        :param memory_limit_bytes: Memory threshold (bytes) that determines whether the reconstruction is computed eagerly with NumPy or lazily with Dask. Arrays estimated to exceed this limit are returned as Dask-backed Xarrays; otherwise NumPy is used. Default is ``1e9`` (1 GB).
        :type memory_limit_bytes: float
        :returns: The reconstructed data. NumPy-backed or Dask-backed depending on *memory_limit_bytes*.
        :rtype: xarray.DataArray

    .. py:method:: extended_pod(C, remove_mean=True, compute=True)

        Compute the Extended POD modes as proposed by Borée (2003). Extended POD finds the correlation between the pre-computed POD modes of the primary field (e.g. a velocity field) and another quantity :math:`C` measured simultaneously in time, such as temperature or pressure. The extended POD modes are

        .. math::

            \boldsymbol{\chi}_j = \frac{1}{\lambda_j\, N} \sum_i a_{ij}\, \mathbf{c}_i',

        where :math:`N` is the number of snapshots, :math:`\lambda_j` is the energy of the :math:`j`-th POD mode, :math:`a_{ij}` are the (unscaled) time coefficients, and :math:`\mathbf{c}_i'` is the fluctuating part of :math:`C`. Unlike the POD spatial modes, the extended POD modes are not unit-norm; their norm quantifies the spatial energy in :math:`C'` that is linearly correlated with mode :math:`j`.

        :param C: The simultaneously measured spatio-temporal field. Must have the same *time_dimension* (with matching size and coordinates) as the array on which POD was fitted, but may have a different number of spatial points. Must be Dask-backed.
        :type C: xarray.DataArray
        :param remove_mean: Whether to remove the temporal mean from *C* before computing the extended POD modes. Set to ``False`` if *C* is already a fluctuating quantity. Default is ``True``.
        :type remove_mean: bool
        :param compute: If ``True``, eagerly compute and return a NumPy-backed ``DataArray``. If ``False``, return a lazy Dask-backed ``DataArray``. Default is ``True``.
        :type compute: bool
        :returns: The extended POD modes with shape ``(n_space_C, n_modes)``, returned in the same format as the POD spatial modes.
        :rtype: xarray.DataArray
        :raises ValueError: If *time_dimension* is not a dimension of *C*, if the number of snapshots does not match the fitted data, or if the time coordinates of *C* do not match those of the training data.

    .. py:method:: compute_modes()

        Materialise :py:attr:`modes` into a NumPy-backed ``DataArray`` when it is still a lazy Dask collection. Intended to be called after :py:meth:`fit` when ``compute_modes=False`` was set at construction time.

        :returns: ``None``.
        :rtype: None

    .. py:method:: compute_time_coeffs()

        Materialise :py:attr:`time_coeffs` into a NumPy-backed ``DataArray`` when it is still a lazy Dask collection. Intended to be called after :py:meth:`fit` when ``compute_time_coeffs=False`` was set at construction time.

        :returns: ``None``.
        :rtype: None

    .. py:method:: compute_energy_ratio()

        Materialise :py:attr:`explained_energy_ratio` into a NumPy array when it is still a lazy Dask collection. Intended to be called after :py:meth:`fit` when ``compute_energy_ratio=False`` was set at construction time.

        :returns: ``None``.
        :rtype: None

    .. rubric:: Examples

    Fit a POD model and inspect the leading spatial modes and their energy:

    .. code-block:: python

        from svdrom.pod import POD

        pod = POD(n_modes=10)
        pod.fit(X)

        modes = pod.modes             # spatial modes, (n_spatial_points, n_modes)
        coeffs = pod.time_coeffs      # time coefficients, (n_modes, n_snapshots)
        energy = pod.energy           # modal energy, (n_modes,)

    Reconstruct a range of snapshots by label:

    .. code-block:: python

        reconstruction = pod.reconstruct(slice("2020-12-01", "2020-12-31"))

    Correlate the POD modes of the primary field with a simultaneously measured field ``C``:

    .. code-block:: python

        chi = pod.extended_pod(C)
