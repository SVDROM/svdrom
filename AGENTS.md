# AGENTS.md

## Setup Commands
- Clone the repository
- Use `uv` for package management
- Install dependencies: `uv sync --all-extras`
- Run tests: `uv run pytest`
- Run pre-commit checks: `uv run pre-commit`

## Code Style
- Follow PEP 8 for Python code
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Make sure pre-commit checks pass before committing

## Testing Guidelines
- Write unit tests for all new functions
- Aim for >80% code coverage
- Run tests before committing

## Project Structure
- `/src` - Main application code
- `/tests` - Test files
- `/demos` - Demo Jupyter notebooks

## Development Workflow
- Unless specified otherwise, create feature branches from `main`
- Use pull requests for code review
- Squash commits before merging
- Use the branch naming convention `<agent-name>/<ticket>-<description>`
