[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

# Offline Installation Quickstart

[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

This guide summarises how to install the project without internet access and run the Macro-Sentinel demo.

## 1. Build the wheelhouse
Run these commands on a machine with connectivity:

```bash
mkdir -p /media/wheels
pip wheel -r requirements.txt -w /media/wheels
pip wheel -r alpha_factory_v1/requirements-colab.txt -w /media/wheels
```

## 2. Create lock files
Compile reproducible requirements from the wheel cache:

```bash
pip-compile --generate-hashes --output-file requirements.lock requirements.txt
pip-compile --no-index --find-links /media/wheels --generate-hashes \
    --output-file alpha_factory_v1/requirements-colab.lock \
    alpha_factory_v1/requirements-colab.txt
```

## 3. Verify the environment
Use `check_env.py` to install any missing packages from the wheelhouse:

```bash
python check_env.py --auto-install --wheelhouse /media/wheels
```

## Environment variables
Set these variables when running offline so the helper scripts install
packages from the local wheel cache and Insight never attempts network
access:

| Variable | Purpose |
|----------|---------|
| `WHEELHOUSE` | Directory containing prebuilt wheels. |
| `AUTO_INSTALL_MISSING` | Set to `1` to automatically install missing packages. |
| `AGI_INSIGHT_OFFLINE` | Set to `1` to force local inference models. |
| `AGI_INSIGHT_BROADCAST` | Set to `0` to disable network broadcasting. |

## 4. Launch Macro-Sentinel
Start the demo with offline data feeds:

```bash
cd alpha_factory_v1/demos/macro_sentinel
LIVE_FEED=0 ./run_macro_demo.sh
```

See [docs/OFFLINE_SETUP.md](OFFLINE_SETUP.md) for additional details.

