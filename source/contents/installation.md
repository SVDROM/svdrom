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

You can install additional optional dependencies specified under `project.optional-dependencies` in `pyproject.toml`. For example, for dependencies related to weather analysis:

```
python -m pip install ".[weather]"
```

## Using uv

It is highly recommended to use [uv](https://docs.astral.sh/uv/getting-started/features/#the-pip-interface) to install SVD-ROM.
