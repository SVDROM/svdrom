# Theoretical background

## Singular value decomposition for linear reduced order modeling

Despite their high dimensionality, datasets in fluid dynamics, weather, and climate often exhibit low-rank structure, where a small number of dominant patterns explain most of the variability.
For instance, at certain flow velocities, periodic vortex shedding will take place behind a circular cylinder in a cross-flow, which will dominate the dynamics.
If we consider temperature across the Earth's atmosphere, while there are many scales of variability, the temperature fluctuations will be dominated by the seasonal component with annual frequency.
Both of these examples correspond to spatio-temporal dynamical systems, where there are spatially coherent motions that oscillate over time.
Singular Value Decomposition (SVD)-based methods provide efficient and interpretable tools for linear dimensionality reduction in such systems.

## Representing the data as a spatio-temporal matrix

A typical way to represent these spatio-temporal systems is by arranging the data into a matrix $\mathbf{X}$, where rows represent different spatial locations and columns represent different points over time.
Each column therefore represents a snapshot of the system at time $t$.
Typically, we have many more spatial locations than we have temporal snapshots, so these matrices are typically *tall-and-skinny*.
This is especially the case with Computational Fluid Dynamics (CFD) datasets, particularly with high fidelity simulations where the computational meshes can contain millions or tens of millions of cells, while typically only a few hundred snapshots are extracted for transient analysis.
In weather and climate modeling, the situation is somewhat different because the time period over which the data is sampled is typically measured in years or decades, and the temporal resolution can be as small as 6-hourly.
As a result, while the spatial dimension typically still dominates (particularly for global weather models), we might end up with thousands or event tens of thousands of temporal snapshots.
In these situations we might classify the resulting matrix $\mathbf{X}$ as *moderately wide*, rather than strictly tall-and-skinny.

## Scalable algorithms

SVD-ROM enables the application of SVD to matrices larger than memory by making use of the randomized SVD [1].
When combined with a communication-efficient, parallel QR factorization for tall-and-skinny matrices (TSQR) [2], it enables scalable low-rank approximations of arbitrarily large, moderately wide matrices [3].

## References

[1] Halko, N., Martinsson, P. G., Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review 53(2), 217–288.

[2] Benson, A. R., Gleich, D. F., Demmel, J. (2013). Direct QR factorizations for tall-and-skinny matrices in MapReduce architectures. 2013 IEEE International Conference on Big Data, 264-272.

[3] Tepper, M., Sapiro, G. (2016). Compressed Nonnegative Matrix Factorization Is Fast and Accurate. IEEE Transactions on Signal Processing, 64(9), 2269-2283.
