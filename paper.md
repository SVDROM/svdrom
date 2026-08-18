---
title: 'SVD-ROM: A Python package for Singular Value Decomposition Reduced Order Modeling of large datasets.'
tags:
  - Python
  - fluid dynamics
  - weather
  - climate
  - reduced order modeling
  - big data
authors:
  - name: David I. Salvador-Jasin
    orcid: 0000-0001-8892-2410
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
    address: Cognition, London, UK
  - name: Robert Vava
    affiliation: 2
  - name: Oliver Strickson
    affiliation: 1
  - name: Louisa van Zeeland
    affiliation: 1
  - name: Lydia France
    affiliation: 1
  - name: Peter Yatsyshin
    affiliation: 1
  - name: Scott Hosking
    affiliation: 1
affiliations:
  - name: The Alan Turing Institute, London, United Kingdom
    index: 1
  - name: Independent Researcher, London, United Kingdom
    index: 2
date: 13 August 2026
bibliography: paper.bib
---

## Summary

Despite their high dimensionality, datasets in fluid dynamics, weather, and climate often exhibit low-rank structure, where a small number of dominant patterns explain most of the variability.
Singular Value Decomposition (SVD)-based methods, such as the Proper Orthogonal Decomposition (POD) [@Berkooz:1993], provide efficient and interpretable tools for dimensionality reduction in such systems.
Dynamic Mode Decomposition (DMD) [@Schmid:2022] extends this framework to time-resolved data by extracting coherent spatio-temporal structures and their associated dynamics, which enables the construction of low-dimensional, interpretable emulators of complex dynamical systems.
However, the wide-scale adoption of these methods in some domains, for example for weather and climate applications, has been limited by the challenges of applying them to large data volumes.
We present `SVD-ROM`, an open-source Python package for SVD-based Reduced Order Modeling (ROM), designed to operate efficiently on large datasets using parallel and out-of-core computation on standard hardware.

## Statement of need

Modern datasets in fields such as fluid dynamics, weather, and climate are characterized by high spatial and temporal resolution, typically leading to very large multi-dimensional data arrays that exceed the memory capacity of standard computing resources.
Traditional SVD-based algorithms struggle to process such data due to memory constraints and computational bottlenecks.
`SVD-ROM` addresses this challenge by providing a scalable framework that enables efficient computation of SVD-based reduced order models on datasets that would otherwise be intractable.
This is achieved through a combination of scalable algorithms and efficient memory management and parallelization.
In contrast to other packages, `SVD-ROM` is designed to operate on standard computing resources, such as laptop computers.
It is purely written in Python and does not require specialized configuration or compilation.
It is designed to be accessible to researchers and practitioners without requiring deep expertise in high-performance computing, making it a practical choice for ROM of large multi-dimensional arrays.

## Related Work

Several open-source packages exist for ROM of dynamical systems, but most are either not designed for large-scale data or require specialized hardware.
This section provides a brief overview of some notable packages in the field.

`PyLOM` [@Eiximeno:2025] is a Python library for ROM in fluid dynamics with algorithms tailored for supercomputers, where parallel operations are implemented using Message Passing Interface (MPI).
`PySPOD` [@Mengaldo:2021] is a package for Spectral Proper Orthogonal Decomposition (SPOD) [@Schmidt:2020], a frequency-domain extension of POD for statistically stationary time-resolved data, closely related to DMD [@Towne:2018].
Similarly to `PyLOM`, `PySPOD` is designed for high-performance computing environments using MPI for parallelization.
`PyDMD` [@Ichinaga:2024] is a Python package that implements DMD and several of its major variants.
While it includes a number of cutting-edge methods developed to handle real-world data, it is not specifically designed to handle datasets larger than memory.
`SVD-ROM` uses `PyDMD` internally to perform DMD at scale.
`pyMOR` [@Milk:2016] is a library of ROM techniques for reducing the computational complexity of solving parametrized partial differential equation problems.
While `SVD-ROM` is a data-centric/scalable ROM library, `pyMOR` is a model-centric ROM framework.
Finally, `EZyRB` [@Demo:2018] is a general-purpose Python ROM framework focused on POD/reduced-basis methods and parametric model reduction, including interpolation of reduced-order solutions.
While `EZyRB` provides a broader ROM toolbox than `SVD-ROM`, it is not specifically optimized for large-scale data processing.

## Methods

Spatio-temporal data is arranged into an $(m \times n)$ snapshot matrix $\mathbf{X}$ whose rows index spatial locations and whose columns index time.
Typical fluid dynamics applications have $m \gg n$ (the "tall-and-skinny" case), while weather and climate problems also have $m > n$ but with $m$ exceeding $n$ by a more moderate margin.
Its SVD is $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^{*}$, where $\mathbf{U}$ and $\mathbf{V}$ hold the left and right singular vectors and $\boldsymbol{\Sigma}$ is the rectangular diagonal matrix whose entries are the singular values $\sigma_j$, ordered by decreasing magnitude.
ROM proceeds by retaining only the leading $k$ singular values, giving the rank $k$ approximation $\mathbf{X}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_{k}^{*}$, which captures most of the variance of $\mathbf{X}$ when $k \ll n$.

POD is the direct application of the truncated SVD to the (optionally mean-removed) fluctuating field $\mathbf{X}'$, scaled by $1/\sqrt{n}$ so that modal energies do not depend on the number of snapshots:

$$
\mathbf{X}'(\mathbf{x}, t) \approx \sum_{j=1}^{k} \boldsymbol{\phi}_j(\mathbf{x})\, a_j(t),
$$

where the orthonormal spatial modes $\boldsymbol{\phi}_j$ are the left singular vectors $\mathbf{U}_k$, the modal energies are $\lambda_j = \sigma_j^2$, and the time coefficients $a_j(t)$ follow from $\boldsymbol{\Sigma}_k \mathbf{V}_{k}^{*}$.
Extended POD [@Boree:2003] correlates these modes with a second field $\mathbf{C}$ measured simultaneously in time, by projecting its fluctuating part onto the time coefficients as $\boldsymbol{\chi}_j = (\lambda_j n)^{-1} \sum_{i} a_{ij}\, \mathbf{c}_i'$, whose norm measures the energy in $\mathbf{C}'$ that is linearly correlated with mode $j$.

DMD [@Schmid:2022] complements this purely spatial reduction by also extracting temporal dynamics, seeking a decomposition $\mathbf{X} \approx \boldsymbol{\Phi} \mathbf{B} \mathbf{T}(\boldsymbol{\omega})$ into DMD modes $\boldsymbol{\Phi}$, amplitudes $\mathbf{B}$ and temporal dynamics $\mathbf{T}(\boldsymbol{\omega})$, whose $j$-th row is of the form $e^{\omega_j t}$ with $\omega_j$ the complex frequency (oscillation rate and growth or decay) governing the evolution of the $j$-th mode.
Because these dynamics are continuous in time, they can be extrapolated beyond the training window to produce forecasts.
`SVD-ROM` uses Optimized DMD [@Askham:2018], which solves the exponential fitting problem directly by variable projection and is therefore robust to noise and to unevenly sampled snapshots.
Crucially, the fit is performed in the SVD latent space,

$$
\tilde{\boldsymbol{\Phi}}\tilde{\mathbf{B}}, \tilde{\boldsymbol{\omega}} = \arg \min_{\tilde{\boldsymbol{\Phi}} \tilde{\mathbf{B}}, \tilde{\boldsymbol{\omega}}} || \boldsymbol{\Sigma}_k \mathbf{V}_k^{*} - \tilde{\boldsymbol{\Phi}} \tilde{\mathbf{B}} \mathbf{T}(\tilde{\boldsymbol{\omega}}) ||_F,
$$

with the resulting $(k \times k)$ latent modes projected back to physical space via $\boldsymbol{\Phi} = \mathbf{U}_k \tilde{\boldsymbol{\Phi}}$.
This encoder (truncated SVD) $\rightarrow$ processor (Optimized DMD) $\rightarrow$ decoder (orthogonal projection) framework allows `SVD-ROM` to fit DMD models on snapshot matrices that are far too large to be handled directly, and is the key contribution of this work.

In addition, it allows uncertainty quantification through bagging [@Sashidhar:2022].

## Software design

`SVD-ROM` is built on top of `Xarray` [@Hoyer:2017] and `Dask` [@Rocklin:2015], and operates throughout on chunked, Dask-backed `Xarray` `DataArray` objects.
This choice underpins both of the package's design goals.
Firstly, labelled dimensions and coordinates mean that users work with named physical axes (for example, 'time', 'latitude' and 'longitude') rather than raw matrix indices, and are preserved across decomposition, reconstruction and forecasting.
Secondly, the `Dask` backend provides the lazy, out-of-core and parallel execution that allows datasets larger than memory to be processed on a single machine, streaming from and to on-disk `Zarr` stores or `NetCDF` files.

All models share a common interface through an abstract `DecompositionModel` base class exposing the `fit` and `reconstruct` methods.
The computational core is the `TruncatedSVD` class, which offers a choice between a deterministic, communication-efficient QR factorization for tall-and-skinny matrices [@Benson:2013], and a randomized algorithm [@Halko:2011], trading accuracy for speed and relaxed chunking requirements on very large arrays.
`POD` extends `TruncatedSVD` with the appropriate energy normalization, mean removal and extended-POD functionality.
`OptDMD` consumes the pre-computed SVD factors directly and solves the resulting low-dimensional nonlinear least-squares problem via the variable-projection Optimized DMD algorithm of the `PyDMD` package, with optional parallelized bootstrap aggregation for uncertainty quantification [@Sashidhar:2022] and Hankel time-delay embedding [@Brunton:2017] for systems with insufficient rank.
The singular value decomposition is computed once and reused, meaning that a single decomposition can feed several downstream Optimized DMD models at no additional cost.

Throughout, users retain explicit control over what is materialized in memory: `compute_*` flags and companion methods make it possible to build a `Dask` computational graph and defer its evaluation, so that expensive quantities such as spatial modes or time coefficients are only computed when needed.
The package is written in pure Python with no compilation or MPI configuration required, is fully type annotated, and ships optional extras for weather and climate workflows (`weather_utils`).
An `SPOD` class is currently a placeholder, reflecting the intended extension of the same interface to further SVD-based methods in future releases.

## Example

`SVD-ROM` fits a DMD model to a 36 GB slice of the ERA5 reanalysis dataset [@Hersbach:2020] and forecasts 45 days beyond the training window in approximately 10 minutes on a 10-core, 32 GB MacBook M1 Pro.
The slice holds a single variable (temperature at the 500 hPa pressure level) on a `(time, latitude, longitude)` grid: 8,766 snapshots at 4-hourly resolution between 1 Jan 2016 and 31 Dec 2019, on a $0.25^\circ$ grid of $721 \times 1{,}440$ points, giving a $(1{,}038{,}240 \times 8{,}766)$ snapshot matrix.

```python
import xarray as xr

from svdrom import OptDMD, TruncatedSVD
from svdrom.preprocessing import StandardScaler, variable_spatial_stack

# Open the store lazily, remove the time mean, and stack latitude and longitude
# into the "samples" dimension of a tall-and-skinny snapshot matrix.
X = xr.open_dataarray("era5_slice.zarr", chunks="auto")
scaler = StandardScaler()
X = variable_spatial_stack(scaler(X), dims=("latitude", "longitude")).T

# Encode with a randomized truncated SVD, then fit Optimized DMD on the triplet.
svd = TruncatedSVD(
    n_components=40, algorithm="randomized", rechunk=True, compute_var_ratio=True
).fit(X, n_power_iter=2, n_oversamples=15)
dmd = OptDMD(n_modes=40, time_units="h").fit(svd.u, svd.s, svd.v)

# Forecast, keep the real part, restore the grid and the mean, and stream to disk.
forecast = dmd.forecast(forecast_span="45 D").real.unstack() + scaler.mean
forecast.to_zarr("forecast_45_days.zarr", mode="w")
```

Three `SVD-ROM` calls carry the whole workflow, and the data never leaves the labelled, chunked representation.
Only the two `fit` calls and the mean removal are eager; everything else builds a `Dask` graph, so the forecast is returned on the original latitude/longitude grid with its `time` coordinate extended past the training window, and streams to `Zarr` chunk by chunk without ever being held in memory whole.
Of the ten minutes, the randomized SVD accounts for roughly seven; mean removal, the DMD fit and the forecast take about one minute each.
The 40 retained components explain over 80% of the variance of $\mathbf{X}$, and the DMD fit operates on the resulting $(40 \times 8{,}766)$ latent triplet rather than on the full matrix, which is why it runs in-memory with a `NumPy` backend.
Because that triplet is computed once and reused, further DMD models — at different ranks, or bagged for uncertainty quantification — have a small additional computational cost.
The `demos/` directory contains extended versions of this workflow, including retrieval of the ERA5 slice and validation of the forecast against climatology.

## Research impact statement

`SVD-ROM` was developed as part of the Environmental Forecasting mission at the Alan Turing Institute, with the aim of making a computationally efficient and interpretable reduced order modeling and forecasting tool accessible to researchers and practitioners in fluid dynamics, weather, and climate.
A major motivation was to enable the analysis of large datasets that are otherwise intractable on standard computing resources.
`SVD-ROM` and related work has been presented at several conferences and workshops, including Climate Informatics 2026 in Lausanne [@Salvador:2026a], a PyData Meetup in London in February 2026 [@Salvador:2026b], and FOSDEM 2025 in Brussels [@Salvador:2025].
The package is under active development and used by researchers at The Alan Turing Institute to explore use cases in climate and weather forecasting.

## AI usage disclosure

Generative AI tools were used to assist with code development, to edit the documentation, and to review this manuscript.
All AI-assisted output was reviewed, tested, and approved by the authors, who take full responsibility for the content of the software and this paper.

## Acknowledgements

We acknowledge contributions from J. Nathan Kutz (Autodesk Research, UK) and Benet Eiximeno Franch (NVIDIA, Spain) for their advice on the design of a scalable DMD implementation, and from the `PyDMD` development team for their support in integrating Optimized DMD into `SVD-ROM`.
Authors DSJ, OS, LvZ, LF, PY and SH were supported by The Alan Turing Institute.

## References
