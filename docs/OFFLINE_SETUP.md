# Offline Setup Reference

This document summarises how to install the project without internet access.

## Build a wheelhouse
Run these commands on a machine with connectivity:

```bash
mkdir -p /media/wheels
pip wheel -r requirements.txt -w /media/wheels
pip wheel -r requirements-dev.txt -w /media/wheels
```

Copy the directory to the offline host.

## Environment variables
Set these before running the helper scripts:

```bash
export WHEELHOUSE=/media/wheels
export AUTO_INSTALL_MISSING=1
```

`check_env.py` reads them to install packages from the wheelhouse when the network is unavailable.

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
