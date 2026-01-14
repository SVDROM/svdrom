# SVD-ROM

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A Python package for the application of Reduced Order Modeling (ROM) to large datasets using the Singular Value Decomposition (SVD).
The backbone of SVD-ROM is the truncated SVD, which allows you to perform dimensionality reduction on huge arrays, and implement machine learning methods such as Principal Component Analysis (PCA), Proper Orthogonal Decomposition (POD), Spectral Proper Orthogonal Decomposition (sPOD), or Dynamic Mode Decomposition (DMD).
These methods have applications in fields such as fluid dynamics, combustion, finance, weather and climate modeling, neuroscience, or chemometrics, to name a few.
SVD-ROM is work in progress, and currently supports (or will soon support) PCA, POD and DMD.
Other methods will be implemented in the future.

## Installation

From source:
```bash
git clone https://github.com/dsj976/svdrom
cd svdrom
python -m pip install .
```

## Usage

The best way to get started is to have a look at the notebooks in the `demos` folder.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/dsj976/svdrom/workflows/CI/badge.svg
[actions-link]:             https://github.com/dsj976/svdrom/actions
[pypi-link]:                https://pypi.org/project/svdrom/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/svdrom
[pypi-version]:             https://img.shields.io/pypi/v/svdrom
[![All Contributors](https://img.shields.io/github/all-contributors/SVDROM/svdrom?color=ee8449&style=flat-square)](#contributors)
<!-- prettier-ignore-end -->
