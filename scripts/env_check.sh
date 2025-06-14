#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
python scripts/check_python_deps.py
python check_env.py --auto-install
