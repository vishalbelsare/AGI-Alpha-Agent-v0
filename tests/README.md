# SPDX-License-Identifier: Apache-2.0
# 🧪 Root Test Suite

These integration tests expect the `alpha_factory_v1` package to be importable.

## Setup

1. Install the development requirements:
   ```bash
   pip install -r requirements-dev.txt
   ```
2. Install the demo extras:
   ```bash
   pip install -r requirements-demo.txt
   ```
3. Run `python check_env.py --auto-install` to ensure optional packages are installed. The command
   downloads dependencies from PyPI, so it **requires an internet connection or a local
   wheelhouse** supplied via `--wheelhouse <dir>`.
4. Set `PYTHONPATH=$(pwd)` or install the project in editable mode with `pip install -e .`.
5. Execute `pytest -q`.

### Offline install

Create a wheelhouse so the tests run without contacting PyPI. Build the wheels on
a machine with connectivity and copy the directory to the offline host. Include
`requirements.txt` and `requirements-dev.txt` (add the MuZero demo requirements if
needed):

```bash
mkdir -p wheels
pip wheel -r requirements.txt -w wheels
pip wheel -r alpha_factory_v1/demos/muzero_planning/requirements.txt -w wheels
pip wheel -r requirements-dev.txt -w wheels
```

Install and run the tests without contacting PyPI:

```bash
WHEELHOUSE=$(pwd)/wheels pip install --no-index --find-links "$WHEELHOUSE" -r requirements-dev.txt
WHEELHOUSE=$(pwd)/wheels python check_env.py --auto-install --wheelhouse "$WHEELHOUSE"
PYTHONPATH=$(pwd) WHEELHOUSE="$WHEELHOUSE" pytest -q
```

The `check_env.py` command will fail offline unless `--wheelhouse` is provided.
Ensure the `WHEELHOUSE` environment variable points to your wheel directory
before running `pytest`.

Missing optional dependencies often cause failures. Re-run the environment check or pass `--wheelhouse` to install them offline.

When running from the repository root without installation:

```bash
export PYTHONPATH=$(pwd)
python -m pytest -q tests
```

Alternatively install the package first:

```bash
pip install -e .
pytest -q
```
- Playwright test `test_umap_fallback.py` ensures the simulator uses random UMAP coordinates when Pyodide is blocked.
- The `test_bridge_online_mode` case in `test_meta_agentic_tree_search_demo.py` requires the `openai-agents` package. Set `OPENAI_API_KEY=dummy` and run:
```bash
OPENAI_API_KEY=dummy pytest tests/test_meta_agentic_tree_search_demo.py::test_bridge_online_mode
```
- The meta-agentic tree search tests also rely on `numpy` and `pyyaml`. These packages are included in `requirements-dev.txt`, so running `pip install -r requirements-dev.txt` will install them.
