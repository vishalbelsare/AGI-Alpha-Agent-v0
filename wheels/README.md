This repository is a conceptual research prototype. References to "AGI" and "superintelligence" describe aspirational goals and do not indicate the presence of a real general intelligence. Use at your own risk. Nothing herein constitutes financial advice. MontrealAI and the maintainers accept no liability for losses incurred from using this software.

# Offline Wheelhouse

This directory stores prebuilt wheels so the MuZero Planning demo and
unit tests can run without network access. Build the wheelhouse on a
machine with connectivity and copy it here. The helper script
`tools/build_wheelhouse.sh` collects all required wheels from
`requirements.lock`, `requirements-dev.txt`, `requirements-demo.lock`
and each demo's `requirements.lock` file:

```bash
./tools/build_wheelhouse.sh
```

Set `WHEELHOUSE=$(pwd)/wheels` before running the setup script or tests:

```bash
WHEELHOUSE=$(pwd)/wheels AUTO_INSTALL_MISSING=1 ./codex/setup.sh
WHEELHOUSE=$(pwd)/wheels AUTO_INSTALL_MISSING=1 \
  python check_env.py --auto-install --wheelhouse "$WHEELHOUSE"
```

`run_muzero_demo.sh` and `pytest` pick up the `WHEELHOUSE` environment
variable automatically when this directory exists.
