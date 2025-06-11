# SPDX-License-Identifier: Apache-2.0
# 🧪 Root Test Suite

These integration tests expect the `alpha_factory_v1` package to be importable.

## Setup

1. Install the development requirements:
   ```bash
   pip install -r requirements-dev.txt
   ```
2. Install the demo extras **required for the full suite**:
   ```bash
   pip install -r requirements-demo.txt
   ```
   This also installs `openai>=1.82.0,<2.0` and `openai-agents>=0.0.16` for
   the OpenAI Agents tests.
   3. Verify the core dependencies are present:
   ```bash
   python scripts/check_python_deps.py
   ```
4. Ensure `numpy`, `pyyaml` and `pandas` are installed. They ship with
   `requirements-dev.txt` but might be missing in minimal setups.
5. Install any missing optional packages:
   ```bash
   python check_env.py --auto-install
   ```
   These commands download packages from PyPI, so ensure you have either
   internet connectivity or a wheelhouse available via `--wheelhouse <dir>`
   (or the `WHEELHOUSE` environment variable).
   The full suite exercises features that depend on optional packages such as
   `numpy`, `torch`, `pandas`, `prometheus_client`, `gymnasium`, `playwright`,
   `httpx`, `uvicorn`, `git` and `hypothesis`.
   
   The test suite automatically attempts to install missing packages at
   session start when `numpy` or `torch` are unavailable by invoking
   `check_env.py --auto-install`.  Set the `WHEELHOUSE` environment
   variable to point to a local wheel directory when running offline so
   these installs can succeed without contacting PyPI.
6. Set `PYTHONPATH=$(pwd)` or install the project in editable mode with `pip install -e .`.
7. Before running the tests, execute `python check_env.py --auto-install` once
   more (add `--wheelhouse <dir>` when offline), then run `pytest -q`.

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

## Troubleshooting

ImportErrors during test collection usually mean optional packages are missing.
Run:

```bash
python check_env.py --auto-install
```

Use `--wheelhouse <dir>` or set `WHEELHOUSE` when offline so packages
install from your local wheel cache.
