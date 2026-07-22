See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Setting up a development environment manually

You can set up a development environment by running:

```zsh
python3 -m venv venv          # create a virtualenv called venv
source ./venv/bin/activate   # now `python` points to the virtualenv python
pip install -v -e ".[dev]"    # -v for verbose, -e for editable, [dev] for dev dependencies
```

Some tests exercise optional weather-scoring functionality. To run the full
test suite, install the optional `weather` dependency as well:

```zsh
pip install -v -e ".[dev,weather]"
```

# Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
pip install pre-commit # or brew install pre-commit on macOS
pre-commit install # this will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
pytest
```

Without the optional `weather`, tests that require those dependencies are skipped.

# Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=svdrom
```

You can generate a HTML coverage report that you can open in your browser by running:

```bash
coverage html
```

# Pre-commit

This project uses pre-commit for all style checking. Install pre-commit and run:

```bash
pre-commit run -a
```

to check all files.

# Testing GitHub Actions locally

You can run this repo's GitHub Actions workflows locally with
[`nektos/act`][nektos-act], via the `gh` CLI extension:

```bash
gh extension install nektos/gh-act
```

`act` runs workflows in Docker containers, so make sure Docker is installed
and running, and that your user can access the Docker socket (on Linux,
`sudo usermod -aG docker $USER`, then start a new session).

```bash
gh act --list          # show all jobs act can see across .github/workflows
gh act -j pre-commit   # run a specific job
gh act -j checks
gh act pull_request    # run all jobs for a given event
```

Notes:

- The `publish` job in `cd.yml` uses OIDC (`id-token: write`) to authenticate
  to PyPI via trusted publishing, which `act` cannot emulate locally. Only
  `dist` (the build/check job) is meaningful to run locally.
- The `release` event has no local equivalent, so testing jobs that trigger
  on it requires a fake event payload, e.g. `gh act release -e event.json -j dist`
  with `event.json` containing `{ "action": "published" }`.

[nektos-act]: https://nektosact.com/installation/gh.html

# Getting started

We are always looking for new contributors for this open source project.
Have a look at our open issues to find ways in which you can help.
Reach out to [David on LinkedIn](linkedin.com/in/david-salvador-jasin) for more info.
