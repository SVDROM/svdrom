.. SVD-ROM documentation master file, created by
   sphinx-quickstart on Wed Apr  1 20:30:44 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SVD-ROM documentation
=====================

Welcome to the SVD-ROM documentation.
SVD-ROM is a Python package for Singular Value Decomposition (SVD)-based Reduced Order Modeling (ROM).
It aims to bring together in one place algorithms typically used in dynamical system analysis, particularly in the fields of fluid dynamics and weather & climate.
The main advantage of SVD-ROM is that it allows you to work seamlessly with massive arrays, directly on your laptop, without running into out of memory errors.
This is achieved by using scalable implementations of the ROM algorithms, and by using [Dask](https://www.dask.org/) as a backend for out-of-core parallel computing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/installation
