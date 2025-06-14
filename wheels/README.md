# Offline Wheelhouse

This directory stores prebuilt wheels so the MuZero Planning demo and
unit tests can run without network access. Build the wheelhouse on a
machine with connectivity and copy it here:

```bash
mkdir -p wheels
pip wheel -r requirements.txt -w wheels
pip wheel -r alpha_factory_v1/demos/muzero_planning/requirements.txt -w wheels
pip wheel -r requirements-dev.txt -w wheels
```

Set `WHEELHOUSE=$(pwd)/wheels` before running the setup script or tests:

```bash
WHEELHOUSE=$(pwd)/wheels AUTO_INSTALL_MISSING=1 ./codex/setup.sh
WHEELHOUSE=$(pwd)/wheels AUTO_INSTALL_MISSING=1 \
  python check_env.py --auto-install --wheelhouse "$WHEELHOUSE"
```

`run_muzero_demo.sh` and `pytest` pick up the `WHEELHOUSE` environment
variable automatically when this directory exists.
