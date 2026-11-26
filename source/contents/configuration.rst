Configuration Module
====================

The configuration module provides a simple key-value interface for controlling global behaviour of SVD-ROM at runtime.

.. py:module:: config

.. py:function:: get(key=None)

    Retrieve the current configuration.

    :param key: Name of the configuration key to retrieve. If ``None``, returns the entire configuration dictionary.
    :type key: str or None
    :returns: The value for the given key, or the full configuration dictionary if *key* is ``None``.
    :rtype: str or dict

    .. rubric:: Examples

    .. code-block:: python

        import svdrom.config

        # Get all configuration
        svdrom.config.get()

        # Get a specific key
        svdrom.config.get("stack_coord_name")

.. py:function:: set(**kwargs)

    Update editable configuration values for the current session.

    :param kwargs: Key-value pairs of configuration values to update.
    :raises KeyError: If an unknown editable config key is provided.
    :raises ValueError: If ``stack_coord_name`` is not a non-empty string.
    :raises ValueError: If ``array_backend`` is not ``"numpy"`` or ``"cupy"``.
    :raises ImportError: If CuPy is selected but not installed.
    :raises RuntimeError: If CuPy cannot access an NVIDIA GPU.

    .. rubric:: Examples

    .. code-block:: python

        import svdrom.config

        # Change the name of the stacked coordinate dimension
        svdrom.config.set(stack_coord_name="space")

.. rubric:: Configuration Keys

The following configuration keys are available:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Key
     - Description
     - Editable
   * - ``stack_coord_name``
     - Name of the coordinate created when stacking spatial dimensions (used by :py:func:`preprocessing.variable_spatial_stack`).
     - Yes (default: ``"samples"``)
   * - ``array_backend``
     - Numerical backend, delegated to Dask's ``array.backend`` configuration.
       Computed public outputs remain NumPy-backed, including deferred results.
     - Yes (default: ``"numpy"``; alternative: ``"cupy"``)
   * - ``hankel_coord_name``
     - Name of the lag coordinate added during Hankel preprocessing.
     - No (fixed: ``"hankel_lag"``)
   * - ``hankel_time_mapping_attr``
     - Name of the DataArray attribute storing the Hankel time mapping dictionary.
     - No (fixed: ``"hankel_time_mapping"``)

Experimental GPU support
------------------------

Install the ``gpu`` extra with ``uv sync --extra gpu`` on a supported Linux or
Windows NVIDIA system. CuPy requires a compatible driver and CUDA 12 toolkit
components. If the toolkit libraries and headers are missing, install the CUDA
component wheels with ``uv pip install 'cupy-cuda12x[ctk]'`` and use
``uv run --no-sync`` to keep them in the environment.

``svdrom.config.set(array_backend="cupy")`` changes Dask's process-wide backend,
including array creation in user code. Use ``array_backend="numpy"`` to restore
CPU computation. To select the backend for a block of code, use Dask's context
manager:

.. code-block:: python

    import dask
    from svdrom import TruncatedSVD

    with dask.config.set({"array.backend": "cupy"}):
        model = TruncatedSVD(20, algorithm="randomized", compute_u=False).fit(X)

    modes = model.u.compute()  # NumPy-backed output from GPU computations

Randomized SVD and POD default to staged ``compute=True`` execution on CuPy to
lower peak device memory during fitting. Staging eagerly computes reduced
intermediates and can reread input between passes. When any output is deferred,
the compression basis is cached in host memory and uploaded by chunk as needed.
``fit(X, compute=False)`` disables staging. The output flags still control when
the final factors and statistics materialize. Deferred results from a GPU fit
can still execute GPU tasks after a backend change. Graphs that refer to
user-supplied GPU arrays continue to reference that device data.

Covariance POD requires a full spatial covariance matrix and eigensolver
workspace on the GPU. Use randomized POD for large spatial dimensions.
StandardScaler uses the CPU for variables whose dtypes CuPy cannot represent.
OptDMD continues to run on the CPU and warns when CuPy is selected.
