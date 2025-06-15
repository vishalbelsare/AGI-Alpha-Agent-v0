# Contributing

Thank you for considering a contribution to this project. For the complete contributor guide and coding standards, see [AGENTS.md](AGENTS.md).

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

Please read **AGENTS.md** carefully before opening issues or submitting pull requests.

## Type Checking the Demo

Run mypy only on the demo package while iterating:

```bash
mypy --config-file mypy.ini alpha_factory_v1/demos/alpha_agi_insight_v1
```

Replace the path with your demo directory as needed. The configuration excludes
other modules so checks remain fast.

## Running Tests

Before running the test suite, ensure optional dependencies are installed. This
project relies on packages such as `openai-agents` and `google-adk` for the
integration tests.

```bash
python scripts/check_python_deps.py
python check_env.py --auto-install  # add --wheelhouse <dir> when offline
```

The environment check installs any missing packages from PyPI (or from your
wheelhouse when offline). Once it succeeds, execute the tests:

```bash
pytest -q
```
