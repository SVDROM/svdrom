# Installation

Install the package from source by cloning the GitHub repository:

```
git clone https://github.com/SVDROM/svdrom
cd svdrom
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Optional dependencies

You can install additional optional dependencies specified under `project.optional-dependencies` in `pyproject.toml`. The available extras are:

- `weather`: dependencies for weather and climate analysis (Cartopy, WeatherBench2, properscoring).
- `dev`: development and testing dependencies (pytest, pre-commit).
- `docs`: documentation build dependencies (Sphinx, MyST-Parser).

For example, to install dependencies for weather and climate analysis:

```
python -m pip install ".[weather]"
```
```

To install development dependencies:

```
python -m pip install ".[dev]"
```

## Using uv

It is highly recommended to use [uv](https://docs.astral.sh/uv/getting-started/features/#the-pip-interface) to install SVD-ROM.
