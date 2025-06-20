[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

# Offline Setup Reference

[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

This document summarises how to install the project without internet access.

## Build a wheelhouse
Run the helper script on a machine with connectivity:

```bash
./scripts/build_offline_wheels.sh
```

This collects wheels for all lock files inside a `wheels/` directory. Copy this
directory to the offline host.

Set `WHEELHOUSE=$(pwd)/wheels` before running `check_env.py` or the tests:

```bash
export WHEELHOUSE=$(pwd)/wheels
```

## Environment variables
Set these before running the helper scripts:

```bash
export WHEELHOUSE=/media/wheels
export AUTO_INSTALL_MISSING=1
```

`check_env.py` reads them to install packages from the wheelhouse when the network is unavailable.

### Prebuilt wheels for heavy dependencies
`numpy`, `PyYAML` and `pandas` ship as binary wheels on PyPI. These small wheels
can be bundled with the repository so the smoke tests run offline. Grab them
when constructing the wheelhouse so the installer does not attempt to compile
them from source:

```bash
pip wheel numpy pyyaml pandas -w /media/wheels
```

Include any other large dependencies, such as `torch` or `scipy`, by passing
their names to `pip wheel` or `pip download` with the versions pinned in
`requirements.lock`.

If the repository already contains a `wheels/` directory you can use it as the
wheelhouse directly:

```bash
export WHEELHOUSE="$(pwd)/wheels"
```

Run `check_env.py --auto-install --wheelhouse "$WHEELHOUSE"` to install from
this local cache.

## Verify packages
Use the scripts below to confirm all requirements are satisfied:

```bash
python scripts/check_python_deps.py
python check_env.py --auto-install --wheelhouse "$WHEELHOUSE"
```

Run `pytest -q` once the check succeeds.

See [tests/README.md](../tests/README.md#offline-install) and [AGENTS.md](../AGENTS.md#offline-setup) for the full instructions.

### Example: Business demo
The business demo works offline when a wheelhouse is provided. Assuming
the wheels live under `/media/wheels`:

```bash
export WHEELHOUSE=/media/wheels
export AUTO_INSTALL_MISSING=1
python alpha_factory_v1/demos/alpha_agi_business_v1/start_alpha_business.py \
  --wheelhouse "$WHEELHOUSE"
```
