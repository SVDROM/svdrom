# Theoretical background

## Singular value decomposition for linear reduced order modeling

Despite their high dimensionality, datasets in fluid dynamics, weather, and climate often exhibit low-rank structure, where a small number of dominant patterns explain most of the variability.
For instance, at certain flow velocities, periodic vortex shedding will take place behind a circular cylinder in a cross-flow, which will dominate the dynamics.
If we consider temperature across the Earth's atmosphere, while there are many scales of variability, the temperature fluctuations will be dominated by the seasonal component with annual frequency.
Both of these examples correspond to spatio-temporal dynamical systems, where there are spatially coherent motions that oscillate over time.
[Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD)-based methods provide efficient and interpretable tools for linear Reduced Order Modeling (ROM) such systems.

## Representing the data as a spatio-temporal matrix

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

## Scalable SVD algorithms

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
The result of the second step is an approximate SVD of $\mathbf{X}$ that is massively scalable while retaining accuracy and stability.

## References

[1] Rocklin, M (2015). Dask: Parallel computation with blocked algorithms and task scheduling. Proceedings of the 14th Python in Science Conference, 126-132.

[2] Tepper, M., Sapiro, G. (2016). Compressed Nonnegative Matrix Factorization Is Fast and Accurate. IEEE Transactions on Signal Processing, 64(9), 2269-2283.

[3] Halko, N., Martinsson, P. G., Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review 53(2), 217–288.

[4] Benson, A. R., Gleich, D. F., Demmel, J. (2013). Direct QR factorizations for tall-and-skinny matrices in MapReduce architectures. 2013 IEEE International Conference on Big Data, 264-272.
