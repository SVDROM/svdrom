See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Prerequisites

This project uses **[uv]** for dependency management and virtual-environment
handling. Please [install uv][uv-install] before proceeding.

[uv]: https://docs.astral.sh/uv/
[uv-install]: https://docs.astral.sh/uv/getting-started/installation/

# Setting up a development environment

You can set up a development environment by running:

```bash
uv sync --extra dev
```

This creates a `.venv` virtualenv, resolves dependencies, and installs the
project in editable mode with the `dev` extras (pytest, pytest-cov,
pre-commit, etc.).

Some tests exercise optional weather-scoring functionality. To run the full
test suite, include the `weather` extras as well:

```bash
uv sync --extra dev --extra weather
```

# Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks (`pre-commit` is already included in the `dev` extras):

```bash
uv run pre-commit install # this will install a pre-commit hook into the git repo
```

You can also/alternatively run `uv run pre-commit run` (changes only) or
`uv run pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
uv run pytest
```

Without the optional `weather` extras, tests that require those dependencies are
skipped.

# Coverage

Use pytest-cov to generate coverage reports:

```bash
uv run pytest --cov=svdrom
```

You can generate a HTML coverage report that you can open in your browser by running:

```bash
uv run coverage html
```

# Pre-commit

This project uses pre-commit for all style checking. Run:

```bash
uv run pre-commit run -a
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

# Dependency and security maintenance

Dependencies are kept current and free of known vulnerabilities by three pieces that work together.

1. **Dependabot** (`.github/dependabot.yml`) opens PRs every Monday at 05:00 UTC that bump outdated `uv` dependencies and GitHub Actions, whether or not they are vulnerable. Minor and patch bumps are grouped into one PR. Major `cryptography` bumps are ignored: `apache-beam` (pulled in by the `weather` extra via `weatherbench2`) caps it, and resolving a major bump downgrades `apache-beam` and `pyarrow` and brings back known vulnerabilities.
2. **Security audit** (`.github/workflows/security-audit.yml`) runs `uv audit` against `uv.lock` every Monday at 07:00 UTC, on manual dispatch, and on PRs that change `uv.lock`, `pyproject.toml` or the workflow itself. Advisories that can't be fixed yet because of an upstream cap are skipped with commented `--ignore` flags. Remove them once the cap is lifted.
3. **Devin Automation** ("Remediate failed uv audit on SVDROM/svdrom main") starts a [Devin](https://devin.ai) session when the audit job fails on `main`. Failures on PR branches don't trigger it. The session:
   - skips advisories already covered by an open PR, including Dependabot's;
   - bumps the affected packages in `uv.lock` only, without raising floors in `pyproject.toml`;
   - adds commented `--ignore` flags for advisories blocked by an upstream cap or with no fixed release;
   - checks the `uv.lock` diff for downgrades;
   - runs the tests, pre-commit and the audit, then opens a `devin/...` PR for review.

   It never merges, runs at most twice a week, and posts its final report to Slack, including when it can't open a PR.

Dependabot runs two hours before the audit so that its PR already exists and the automation doesn't duplicate it.

When reviewing dependency PRs, check the `uv.lock` diff for downgrades, not just the CI result. If you change the remediation rules above, update the automation's prompt to match.

# Getting started

We are always looking for new contributors for this open source project.
Have a look at our open issues to find ways in which you can help.
Reach out to [David on LinkedIn](linkedin.com/in/david-salvador-jasin) for more info.
