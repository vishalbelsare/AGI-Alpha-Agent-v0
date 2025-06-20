#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Verify core Python packages are present before running the full environment
# check. ``check_python_deps.py`` now relies on ``python -m pip show`` for
# speed and reliability.
set -euo pipefail
python scripts/check_python_deps.py
python check_env.py --auto-install
