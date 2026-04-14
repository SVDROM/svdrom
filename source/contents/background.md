# Theoretical background

## Singular Value Decomposition

Despite their high dimensionality, datasets in fluid dynamics, weather, and climate often exhibit low-rank structure, where a small number of dominant patterns explain most of the variability.
For instance, at certain flow velocities, periodic vortex shedding will take place behind a circular cylinder in a cross-flow, which will dominate the dynamics.
If we consider temperature across the Earth's atmosphere, while there are many scales of variability, the temperature fluctuations will be dominated by the seasonal component with annual frequency.
Both of these examples correspond to spatio-temporal dynamical systems, where there are spatially coherent motions that oscillate over time.
[Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD)-based methods provide efficient and interpretable tools for linear Reduced Order Modeling (ROM) such systems.

### Representing the data as a spatio-temporal matrix

A typical way to represent these spatio-temporal systems is by arranging the data into a matrix $\mathbf{X}$, where rows represent different spatial locations and columns represent different points over time.
Each column therefore represents a snapshot of the system at time $t$.
Typically, we have many more spatial locations than we have temporal snapshots, so these matrices are typically *tall-and-skinny*.
This is especially the case with Computational Fluid Dynamics (CFD) datasets, particularly with high fidelity simulations where the computational meshes can contain millions or tens of millions of cells, while typically only a few hundred snapshots are extracted for transient analysis.
In weather and climate modeling, the situation is somewhat different because the time period over which the data is sampled is typically measured in years or decades, and the temporal resolution can be as small as 6-hourly.
As a result, while the spatial dimension typically still dominates (particularly for global weather models), we might end up with thousands or event tens of thousands of temporal snapshots.
In these situations we might classify the resulting matrix $\mathbf{X}$ as *moderately wide*, rather than strictly tall-and-skinny.

An illustration of what the spatio-temporal matrix would look like is given below.

```{image} ../media/spatio-temporal-matrix.png
:width: 400px
:align: center
```

Given a $(m \times n)$ matrix $\mathbf{X}$, its SVD is defined as:

$$
\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{*},
$$

where $\mathbf{U}$ is the $(m \times m)$ matrix of **left singular vectors**, $\mathbf{\Sigma}$ is the $(m \times n)$ rectangular diagonal matrix of **singular values**, $\mathbf{V}$ is the $(n \times n)$ matrix of **right singular vectors**, and $\mathbf{V}^{*}$ is the conjugate transpose of $\mathbf{V}$.
If $\mathbf{X}$ is tall-and-skinny, then $m >> n$.

To perform ROM, instead of computing the exact SVD by retaining all singular values, we compute a truncated SVD where we only keep the top $k$ singular values.
This allows to perform a rank $k$ approximation of $\mathbf{X}$:

$$
\mathbf{X}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_{k}^{*},
$$

where $\mathbf{U}_k$, $\mathbf{\Sigma}_k$ and $\mathbf{V}_{k}$ now have size $(m \times k)$, $(k \times k)$ and $(n \times k)$, respectively.
If $\mathbf{X}$ truly exhibits low-rank structure, then $k << n$ while $\mathbf{X}_k$ retains most of the variance of $\mathbf{X}$.

To illustrate the usefulness of low-rank approximations for spatio-temporal dynamical systems, consider the figure below which shows a snapshot of vortex shedding behind a cylinder in a cross-flow.
The flow field has been polluted artificially with uncorrelated white noise, which is why the image appears so pixelated.

```{image} ../media/cylinder-full-rank.png
:width: 400px
:align: center
```

We can perform a truncated SVD of the dataset with rank $k=2$, which results in the following reconstruction of the flow field:

```{image} ../media/cylinder-low-rank.png
:width: 400px
:align: center
```

The 2 retained modes represent the large-scale periodic vortex shedding, while all smaller scales (including the artificial white noise), are represented by the modes that we have discarded.

### Scalable SVD algorithms

SVD-ROM employs [Dask](https://docs.dask.org/en/stable/) for parallel, out-of-core computation.
The [Dask Array API](https://docs.dask.org/en/stable/array.html) enables working with arrays larger than memory by cutting them up into many small blocks and applying blocked algorithms coordinated with dynamic task scheduling [1].

A scalable implementation of the SVD is provided by Dask's [`svd_compressed()`](https://docs.dask.org/en/stable/generated/dask.array.linalg.svd_compressed.html).
This function enables the application of the SVD to arbitrarily large matrices, eliminating the tall-and-skinny requirement while preserving efficiency and accuracy [2].
In the first step, a random projection is applied to the input matrix $\mathbf{X}$, which may contain a considerably large number of columns ($n$).
This approach assumes that the input matrix is low-rank, and therefore most of the action of $\mathbf{X}$ occurs in a subspace.
The main idea is that this subspace can be identified through random sampling [3].
The moderately-wide input matrix $\mathbf{X}$ is then compressed by projecting it into this much smaller subspace, converting it into a tall-and-skinny matrix $\mathbf{Y}$ of approximately the same column space.
In the second step, a parallel and communication-efficient algorithm for direct QR factorization of tall-and-skinny matrices (TSQR) [4] is applied to $\mathbf{Y}$.
A major advantage of applying the direct QR factorization, compared to other approaches, is numerical stability.
The result of a randomized projection + TSQR is an approximate SVD of $\mathbf{X}$ that is massively scalable while retaining accuracy and stability.

## Dynamic Mode Decomposition

[Dynamic Mode Decomposition](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition) (DMD) extends the ROM framework of SVD to time-resolved data by extracting coherent spatio-temporal structures and their associated dynamics [5].
The SVD by itself does not extract temporal correlation from the data.
For instance, one could shuffle the temporal order of the columns in the spatio-temporal matrix $\mathbf{X}$ and would obtain the same modes and associated singular values.
SVD can be understood as only performing dimensionality reduction along the spatial direction.
DMD, on the other hand, connects the favorable aspects of the SVD for spatial dimensionality reduction and the Fast Fourier Transform (FFT) for temporal frequency identification.

Given a ($m \times n$) matrix $\mathbf{X}$, with snapshots organized into columns, DMD seeks a rank $k$ spatio-temporal decomposition of $\mathbf{X}$ of the following form:

$$
\mathbf{X} \approx \mathbf{\Phi} \mathbf{B} \mathbf{T}(\boldsymbol{\omega}),
$$

where $\mathbf{\Phi}$ is the $(m \times k)$ matrix of DMD modes, $\mathbf{B}$ is the ($k \times k$) diagonal matrix of mode amplitudes, and $\mathbf{T}(\boldsymbol{\omega})$ is the $(k \times n)$ matrix of temporal dynamics of the form $e^{\omega_j t}$, where the $j^{th}$ row contains the time evolution of the $j^{th}$ DMD mode governed by complex frequency $\omega_j$.

To compute the DMD modes and associated dynamics, the exact DMD algorithm [5] seeks the leading spectral decomposition of the best-fit linear operator $\mathbf{A}$ that advances $\mathbf{X}$ to its time-shifted version $\mathbf{X}'$:

$$
\mathbf{X}' = \mathbf{A} \mathbf{X}
$$

DMD is closely related to Koopman spectral theory.
While DMD is fundamentally a linear approximation, it can approximate the [Koopman operator](https://en.wikipedia.org/wiki/Composition_operator), which allows nonlinear dynamics to be represented in an infinite-dimensional linear framework [6].

The figure below, reproduced from [6], shows the result of applying DMD to a time-series of snapshots of a cylinder in a cross-flow.
The extracted DMD modes represent spatial patterns that are accompanied by corresponding temporal dynamics (with a frequency of oscillation and a growth or decay rate).
These dynamics can be extrapolated into the future, enabling the use of DMD for forecasting.

```{image} ../media/dmd-cylinder.png
:width: 500px
:align: center
```

### Scalable DMD algorithms

As discussed above, exact DMD (the original DMD implementation) seeks the leading spectral decomposition of the operator $\mathbf{A}$.
However, it is known to be strongly affected by the presence of noise, which is always present in real-world datasets.
Additionally, exact DMD requires that the snapshots in $\mathbf{X}$ are evenly sampled in time.
Optimized DMD (OptDMD) [7] is a non-linear optimization of DMD enabled by variable projection methods.
It avoids much of the bias of exact DMD, it is robust to noise and can handle snapshots that are unevenly sampled in time.
OptDMD solves the exponential fitting problem directly:

$$
\mathbf{\Phi}\mathbf{B}, \boldsymbol{\omega} = \arg \min_{\mathbf{\Phi} \mathbf{B}, \boldsymbol{\omega}} || \mathbf{X} - \mathbf{\Phi} \mathbf{B} \mathbf{T}(\boldsymbol{\omega}) ||
$$

When the snapshot matrix $\mathbf{X}$ is very large, instead of solving the exponential fitting problem directly on the data, one can perform a rank $k$ truncated SVD of $\mathbf{X}$ and solve the DMD fitting problem in the SVD latent space:

$$
\mathbf{\Phi}\mathbf{B}, \boldsymbol{\omega} = \arg \min_{\mathbf{\Phi} \mathbf{B}, \boldsymbol{\omega}} || \mathbf{\Sigma}_k \mathbf{V}_k^* - \mathbf{\Phi} \mathbf{B} \mathbf{T}(\boldsymbol{\omega}) ||
$$

The resulting DMD modes can then be projected back to the original space using the left singular vectors $\mathbf{U}_k$.

This can be viewed as an encoder (truncated SVD) $\rightarrow$ processor (OptDMD) $\rightarrow$ decoder (orthogonal projection) framework.
If we use a highly scalable SVD algorithm (randomization + TSQR), it allows to perform DMD on huge tall-and-skinny or moderately-wide snapshot matrices $\mathbf{X}$.
This is the approach that has been implemented in SVD-ROM.

## References

[1] Rocklin, M (2015). Dask: Parallel computation with blocked algorithms and task scheduling. Proceedings of the 14th Python in Science Conference, 126-132.

[2] Tepper, M., Sapiro, G. (2016). Compressed Nonnegative Matrix Factorization Is Fast and Accurate. IEEE Transactions on Signal Processing, 64(9), 2269-2283.

[3] Halko, N., Martinsson, P. G., Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review 53(2), 217–288.

[4] Benson, A. R., Gleich, D. F., Demmel, J. (2013). Direct QR factorizations for tall-and-skinny matrices in MapReduce architectures. 2013 IEEE International Conference on Big Data, 264-272.

[5] Schmid, P. J. (2022). Dynamic Mode Decomposition and Its Variants. Annual Review of Fluid Mechanics 54, 225-254.

[6] Kutz, J. N., Brunton, S. L., Brunton, B. W., Proctor, J. L. (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.

[7] Askham, T., & Kutz, J. N. (2018). Variable Projection Methods for an Optimized Dynamic Mode Decomposition. SIAM Journal on Applied Dynamical Systems, 17(1), 380-416.
