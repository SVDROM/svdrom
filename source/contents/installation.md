# Installation

It is strongly recommended to use [uv](https://docs.astral.sh/uv/) to install SVD-ROM. Please [install uv](https://docs.astral.sh/uv/getting-started/installation/) before proceeding.

Install the package from source by cloning the GitHub repository:

```bash
git clone https://github.com/SVDROM/svdrom
cd svdrom
uv venv
source .venv/bin/activate
uv pip install .
```

## Optional dependencies

You can install additional optional dependencies specified under `project.optional-dependencies` in `pyproject.toml`. The available extras are:

- `weather`: dependencies for weather and climate analysis (Cartopy, WeatherBench2, properscoring).
- `dev`: development and testing dependencies (pytest, pre-commit).
- `docs`: documentation build dependencies (Sphinx, MyST-Parser).

For example, to install dependencies for weather and climate analysis:

```bash
uv pip install ".[weather]"
```

To install development dependencies:

```bash
uv pip install ".[dev]"
```

You can combine multiple extras in a single command:

```bash
uv pip install ".[dev,weather]"
```
