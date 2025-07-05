#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install pre-commit and configure the git hook.
set -euo pipefail

PYTHON=${PYTHON:-python3}

wheel_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  wheel_opts+=(--no-index --find-links "$WHEELHOUSE")
fi

if ! command -v pre-commit >/dev/null; then
  "$PYTHON" -m pip install --quiet "${wheel_opts[@]}" pre-commit
fi

pre-commit install
