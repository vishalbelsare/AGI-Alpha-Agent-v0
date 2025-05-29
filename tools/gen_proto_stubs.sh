#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Generate protobuf stubs for Python and Go.
set -euo pipefail

PROTO="src/utils/a2a.proto"
PY_OUT="src/utils"
GO_OUT="tools/go_a2a_client"
PY_INCLUDE=$(python -c 'import pkg_resources,os;print(os.path.dirname(pkg_resources.resource_filename("google.protobuf","struct.proto")))')

protoc -I "$(dirname "$PROTO")" -I "$PY_INCLUDE" --python_out="$PY_OUT" "$PROTO"

if command -v protoc-gen-go >/dev/null; then
  mkdir -p "$GO_OUT"
  protoc -I "$(dirname "$PROTO")" -I "$PY_INCLUDE" --go_out="$GO_OUT" --go_opt=paths=source_relative "$PROTO"
else
  echo "protoc-gen-go not found; skipping Go stub generation" >&2
fi

