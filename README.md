# SVD-ROM

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![Documentation Status][rtd-badge]][rtd-link]

A Python package for the application of Reduced Order Modeling (ROM) to large spatio-temporal datasets using the Singular Value Decomposition (SVD).
A spatio-temporal dataset is one that has both spatial and temporal dimensions, such as weather or climate data, or a numerical simulation of flow past a circular cylinder.
Despite the many degrees of freedom present in such datasets, they often exhibit low-dimensional structures that can be captured using dimensionality reduction techniques.

SVD-ROM is specifically designed for very large datasets that do not fit in memory.
It leverages out-of-core, parallel computing, as well as scalable and noise-robust versions of matrix decomposition algorithms such as the Randomized SVD and the Optimized Dynamic Mode Decomposition (DMD).
SVD-ROM provides a user-friendly API fully built on the Python ecosystem, abstracting away the complexity of distributed computing and memory management from the user.
It is built on top of [Dask](https://www.dask.org/), a Python library for parallel computing, which enables efficient handling of large-scale data processing on laptops, HPC clusters or the cloud with minimal code changes.

SVD-ROM is work in progress, and currently supports Principal Component Analysis (PCA), Proper Orthogonal Decomposition (POD) and Dynamic Mode Decomposition (DMD).
We will soon add support for other methods such as the Spectral Proper Orthogonal Decomposition (SPOD).

SVD-ROM is an open-source project that originated at [The Alan Turing Institute (London, UK)](https://www.turing.ac.uk/).


## Installation

Intall SVD-ROM from source:
```bash
git clone https://github.com/SVDROM/svdrom
cd svdrom
python -m pip install .
```

It is strongly recommended to install SVD-ROM using [uv](https://docs.astral.sh/uv/):
```bash
uv venv
source .venv/bin/activate
uv pip install .
```

## Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) from the `source/` directory and hosted on [Read the Docs][rtd-link].
See [source/contents/development.md](source/contents/development.md) for instructions on building it locally.

## Usage

The best way to get started is to have a look at the notebooks in the `demos/` folder.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/robertvava"><img src="https://avatars.githubusercontent.com/u/77018483?v=4?s=100" width="100px;" alt="Robert Vava"/><br /><sub><b>Robert Vava</b></sub></a><br /><a href="https://github.com/SVDROM/svdrom/pulls?q=is%3Apr+reviewed-by%3Arobertvava" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/SVDROM/svdrom/commits?author=robertvava" title="Code">💻</a> <a href="#maintenance-robertvava" title="Maintenance">🚧</a> <a href="https://github.com/SVDROM/svdrom/commits?author=robertvava" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dsj976"><img src="https://avatars.githubusercontent.com/u/57944311?v=4?s=100" width="100px;" alt="David Salvador Jasin"/><br /><sub><b>David Salvador Jasin</b></sub></a><br /><a href="https://github.com/SVDROM/svdrom/commits?author=dsj976" title="Code">💻</a> <a href="#maintenance-dsj976" title="Maintenance">🚧</a> <a href="https://github.com/SVDROM/svdrom/pulls?q=is%3Apr+reviewed-by%3Adsj976" title="Reviewed Pull Requests">👀</a> <a href="#infra-dsj976" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/SVDROM/svdrom/commits?author=dsj976" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/louisavz"><img src="https://avatars.githubusercontent.com/u/63079440?v=4?s=100" width="100px;" alt="louisavz"/><br /><sub><b>louisavz</b></sub></a><br /><a href="#research-louisavz" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LydiaFrance"><img src="https://avatars.githubusercontent.com/u/85945427?v=4?s=100" width="100px;" alt="Lydia France"/><br /><sub><b>Lydia France</b></sub></a><br /><a href="#research-LydiaFrance" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ots22"><img src="https://avatars.githubusercontent.com/u/5434836?v=4?s=100" width="100px;" alt="ots22"/><br /><sub><b>ots22</b></sub></a><br /><a href="#research-ots22" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.yatsyshin.com/"><img src="https://avatars.githubusercontent.com/u/20075514?v=4?s=100" width="100px;" alt="Peter Yatsyshin"/><br /><sub><b>Peter Yatsyshin</b></sub></a><br /><a href="#research-pyatsysh" title="Research">🔬</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/SVDROM/svdrom/workflows/CI/badge.svg
[actions-link]:             https://github.com/SVDROM/svdrom/actions
[pypi-link]:                https://pypi.org/project/svdrom/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/svdrom
[pypi-version]:             https://img.shields.io/pypi/v/svdrom
[rtd-badge]:                https://readthedocs.org/projects/svdrom/badge/?version=latest
[rtd-link]:                 https://svdrom.readthedocs.io/en/latest/
<!-- prettier-ignore-end -->
