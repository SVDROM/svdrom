# Development

## Building the documentation

To build the Sphinx documentation, first install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Then build the HTML documentation:

```bash
make html
```

The built HTML documentation will be in `build/html/`. You can view available build targets with:

```bash
make help
```

## Testing

Use pytest to run the unit checks:

```bash
pytest
```

## Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=svdrom
```

You can generate a HTML coverage report that you can open in your browser by running:

```bash
coverage html
```

## Pre-commit

This project uses pre-commit for all style checking. Install pre-commit and run:

```bash
pre-commit run --all-files
```

to check all files.
