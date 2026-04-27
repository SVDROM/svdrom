SVD Module
==========

The SVD module provides access to the ``TruncatedSVD`` class, which provides an API for computing a rank-:math:`k` truncated Singular Value Decomposition of a Dask-backed Xarray ``DataArray``.

.. py:module:: svd

.. py:class:: TruncatedSVD(n_components, algorithm='tsqr', compute_u=True, compute_v=True, compute_var_ratio=False, rechunk=False)

    Linear dimensionality reduction via truncated Singular Value Decomposition (SVD), operating directly on Dask-backed Xarray ``DataArray`` objects.

    Given a 2-D snapshot matrix :math:`\mathbf{X} \in \mathbb{R}^{m \times n}` with :math:`m` spatial points and :math:`n` temporal snapshots, the full SVD factorises :math:`\mathbf{X}` as :math:`\mathbf{X} = \mathbf{U}\, \mathbf{\Sigma}\, \mathbf{V}^{*}`, where :math:`\mathbf{U}` and :math:`\mathbf{V}` are orthonormal and :math:`\mathbf{\Sigma}` is a diagonal matrix of singular values :math:`\sigma_1 \geq \sigma_2 \geq \cdots \geq 0`. The rank-:math:`k` truncation retains only the :math:`k` leading singular triplets:

    .. math::

        \mathbf{X}_k
        = \mathbf{U}_k\, \mathbf{\Sigma}_k\, \mathbf{V}_k^{*}
        = \sum_{j=1}^{k} \sigma_j\, \mathbf{u}_j\, \mathbf{v}_j^{*},

    which by the Eckart–Young theorem is the best rank-:math:`k` approximation of :math:`\mathbf{X}` in the spectral and Frobenius norms. See :doc:`background` for the broader theoretical context, including the randomised projection used by the scalable back-end.

    The class API is intentionally close to ``dask_ml.decomposition.TruncatedSVD``, but fit/transform operate on ``xarray.DataArray`` so that named dimensions and coordinates (e.g. ``"time"``) are preserved through the decomposition and can be consumed by downstream tools such as :py:class:`dmd.OptDMD`.

    .. note::
        Two back-ends are available. ``"tsqr"`` performs a direct SVD via Dask's ``dask.array.linalg.svd``, which uses the Tall-and-Skinny QR factorisation internally. It is numerically stable and highly accurate, but only supports chunking along a single axis (tall-and-skinny or short-and-fat arrays) and is slower on very large problems. ``"randomized"`` performs an approximate SVD via Dask's ``dask.array.linalg.svd_compressed``, combining a randomised range finder with TSQR on the compressed matrix. It supports chunking along both axes, scales to much larger problems and is substantially faster, at the cost of a small, controllable approximation error that is acceptable when :math:`\mathbf{X}` is effectively low-rank.

    :param n_components: Number of singular triplets to retain (the truncation rank :math:`k`). Must be strictly less than the number of features of the input array.
    :type n_components: int
    :param algorithm: SVD back-end. Must be one of ``{"tsqr", "randomized"}``. Default is ``"tsqr"``.
    :type algorithm: str
    :param compute_u: Whether to eagerly compute the left singular vectors :math:`\mathbf{U}_k` during :py:meth:`fit`. If ``False``, the computation graph is built but not executed and :py:attr:`u` is returned as a lazy Dask collection. Default is ``True``.
    :type compute_u: bool
    :param compute_v: Whether to eagerly compute the right singular vectors :math:`\mathbf{V}_k` during :py:meth:`fit`. If ``False``, the computation graph is built but not executed and :py:attr:`v` is returned as a lazy Dask collection. Default is ``True``.
    :type compute_v: bool
    :param compute_var_ratio: Whether to eagerly compute the ratio of variance explained by each retained component. If ``False``, the computation graph is built but not executed and :py:attr:`explained_var_ratio` is returned as a lazy Dask collection. Default is ``False``.
    :type compute_var_ratio: bool
    :param rechunk: If ``True`` and *algorithm* is ``"randomized"``, the input array is rechunked to a single chunk along its smallest dimension before the SVD is computed. Rechunking is always performed when *algorithm* is ``"tsqr"``, regardless of this flag, because TSQR requires chunking along a single axis. Default is ``False``.
    :type rechunk: bool

    .. note::
        Although rechunking is not a hard requirement for the ``"randomized"`` back-end (unlike ``"tsqr"``), in our experimentation we found that setting ``rechunk=True`` noticeably improved the accuracy of the randomised SVD.
        Enabling it is therefore recommended when accuracy matters more than the computational savings of preserving the original chunking.

    .. rubric:: Attributes

    .. py:property:: n_components
        :type: int

        Number of singular triplets retained in the truncation (read-only).

    .. py:property:: algorithm
        :type: str

        SVD back-end in use, either ``"tsqr"`` or ``"randomized"`` (read-only).

    .. py:property:: u
        :type: xarray.DataArray or None

        Left singular vectors :math:`\mathbf{U}_k` with shape ``(n_samples, n_components)`` (read-only). NumPy-backed when *compute_u* was ``True``, otherwise Dask-backed. ``None`` before the model is fitted.

    .. py:property:: s
        :type: numpy.ndarray or None

        Singular values :math:`\sigma_1 \geq \ldots \geq \sigma_k` with shape ``(n_components,)`` (read-only). ``None`` before the model is fitted.

    .. py:property:: v
        :type: xarray.DataArray or None

        Right singular vectors :math:`\mathbf{V}_k^{*}` with shape ``(n_components, n_features)`` (read-only). NumPy-backed when *compute_v* was ``True``, otherwise Dask-backed. ``None`` before the model is fitted.

    .. py:property:: explained_var_ratio
        :type: numpy.ndarray, dask.array.Array, or None

        Ratio of variance explained by each retained component with shape ``(n_components,)`` (read-only). NumPy-backed when *compute_var_ratio* was ``True``, otherwise Dask-backed. ``None`` before the model is fitted.

    .. rubric:: Methods

    .. py:method:: fit(X, **kwargs)

        Fit the truncated SVD to the input data array. The method validates that *X* is a 2-D Dask-backed ``DataArray`` with strictly more features than *n_components*, rechunks when required by the selected algorithm, dispatches to ``dask.array.linalg.svd`` or ``dask.array.linalg.svd_compressed``, and materialises the decomposition according to the *compute_u*, *compute_v* and *compute_var_ratio* flags set at construction time. The singular vectors are wrapped in ``xarray.DataArray`` objects that preserve the coordinate of the non-reduced axis of *X* and introduce a new ``"components"`` axis indexed by ``0, 1, ..., n_components - 1``.

        :param X: Input snapshot matrix with shape ``(n_samples, n_features)``. Must be a Dask-backed ``xarray.DataArray``.
        :type X: xarray.DataArray
        :param \**kwargs: Additional keyword arguments forwarded to ``dask.array.linalg.svd_compressed`` when *algorithm* is ``"randomized"`` (e.g. ``n_power_iter``, ``n_oversamples``, ``seed``). See `svd_compressed docs <https://docs.dask.org/en/stable/generated/dask.array.linalg.svd_compressed.html>`_ for details. Ignored when *algorithm* is ``"tsqr"``.
        :returns: ``None``. The decomposition is stored on the instance and exposed through :py:attr:`u`, :py:attr:`s`, :py:attr:`v` and :py:attr:`explained_var_ratio`.
        :rtype: None

    .. py:method:: transform(X)

        Project an input array onto the fitted right singular vectors, computing :math:`\tilde{\mathbf{X}} = \mathbf{X}\, \mathbf{V}_k^{*\top}`. The input must have the same number of features as the array on which the model was fitted. The returned ``DataArray`` replaces the feature dimension with the ``"components"`` axis.

        :param X: Array to project with shape ``(n_samples, n_features)``. Must be a Dask-backed ``xarray.DataArray``.
        :type X: xarray.DataArray
        :returns: The projected array with shape ``(n_samples, n_components)``.
        :rtype: xarray.DataArray

    .. py:method:: reconstruct_snapshot(snapshot, snapshot_dim='time')

        Reconstruct one or more snapshots from the truncated decomposition using :math:`\hat{\mathbf{x}}_j = \mathbf{U}_k\, \mathbf{\Sigma}_k\, \mathbf{v}_j`. The method indexes along the snapshot dimension of the fitted decomposition and handles both positional (``int``) and label-based (``str``) selection. When a label matches multiple entries (e.g. a date label resolving to several timestamps) all matching snapshots are returned stacked along the snapshot dimension.

        :param snapshot: Index or coordinate label of the snapshot to reconstruct. Integers are interpreted as positional indices; strings are interpreted as labels along *snapshot_dim*.
        :type snapshot: int or str
        :param snapshot_dim: Name of the dimension along which snapshots are indexed. Must match one of the dimensions of :py:attr:`u` or :py:attr:`v`. Default is ``"time"``.
        :type snapshot_dim: str
        :returns: The reconstructed snapshot(s).
        :rtype: xarray.DataArray

        .. rubric:: Examples

        Given ``tsvd``, a fitted ``TruncatedSVD`` instance whose training data had a ``"time"`` dimension:

        Reconstruct the first snapshot by positional index:

        .. code-block:: python

            tsvd.reconstruct_snapshot(0)

        Reconstruct every snapshot that falls on the date label ``"2017-01-01"``:

        .. code-block:: python

            tsvd.reconstruct_snapshot("2017-01-01")

        Reconstruct a snapshot from an array whose snapshot dimension is not named ``"time"``:

        .. code-block:: python

            tsvd.reconstruct_snapshot(0, snapshot_dim="samples")

    .. py:method:: compute_u()

        Materialise :py:attr:`u` into a NumPy-backed ``DataArray`` when it is still a lazy Dask collection. Intended to be called after :py:meth:`fit` when ``compute_u=False`` was set at construction time.

        :returns: ``None``.
        :rtype: None

    .. py:method:: compute_v()

        Materialise :py:attr:`v` into a NumPy-backed ``DataArray`` when it is still a lazy Dask collection. Intended to be called after :py:meth:`fit` when ``compute_v=False`` was set at construction time.

        :returns: ``None``.
        :rtype: None

    .. py:method:: compute_var_ratio()

        Materialise :py:attr:`explained_var_ratio` into a NumPy array when it is still a lazy Dask collection. Intended to be called after :py:meth:`fit` when ``compute_var_ratio=False`` was set at construction time.

        :returns: ``None``.
        :rtype: None
