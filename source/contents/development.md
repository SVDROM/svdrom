# Development

## Building the documentation

To build the Sphinx documentation, first install the documentation dependencies:

```bash
uv sync --extra docs
```

Then build the HTML documentation:

```bash
uv run make html
```

The built HTML documentation will be in `build/html/`. You can view available build targets with:

```bash
uv run make help
```

## Testing

Install the same extras used by the CPU test jobs and run the unit checks:

```bash
uv sync --extra dev --extra weather
uv run --no-sync pytest
```

For a GPU test environment, also select ``--extra gpu`` when syncing. The GPU
tests skip when CuPy is absent or no CUDA device is available.

## Coverage

Use pytest-cov to generate coverage reports:

```bash
uv run pytest --cov=svdrom
```

You can generate a HTML coverage report that you can open in your browser by running:

```bash
uv run coverage html
```

## Pre-commit

This project uses pre-commit for all style checking. Install pre-commit and run:

```bash
uv run pre-commit run --all-files
```

to check all files.
