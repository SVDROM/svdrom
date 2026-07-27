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
   * - ``hankel_coord_name``
     - Name of the lag coordinate added during Hankel preprocessing.
     - No (fixed: ``"hankel_lag"``)
   * - ``hankel_time_mapping_attr``
     - Name of the DataArray attribute storing the Hankel time mapping dictionary.
     - No (fixed: ``"hankel_time_mapping"``)
