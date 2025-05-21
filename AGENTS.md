# AGENTS.md

This file defines coding and testing guidelines for contributors and AI agents working on this repository.

## Scope
These instructions apply to the entire repository.

## Coding style
- Prefer Python 3.9+ syntax with type hints.
- Use 4 spaces per indentation level.
- Keep lines under **120** characters when possible.
- Document functions and classes with concise English docstrings and comments, following the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Tests
- Always run the full test suite with `pytest -q` before committing changes.
- If dependencies are missing, run `python check_env.py --auto-install` to install them locally.
- Tests must pass or the failure should be explained in the pull request.

