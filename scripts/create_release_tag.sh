#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create the v0.1.0-alpha annotated release tag.

set -euo pipefail

usage() {
    cat <<USAGE
Usage: $0 [commit]

Create the annotated tag 'v0.1.0-alpha' at the given commit.
If no commit is provided, HEAD is used.
USAGE
}

if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
    usage
    exit 0
fi

commit=${1:-HEAD}

git tag -a v0.1.0-alpha "$commit"
echo "Created tag v0.1.0-alpha at $commit"
