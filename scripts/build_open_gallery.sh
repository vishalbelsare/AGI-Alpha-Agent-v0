#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the demo gallery and open it in the default browser.
# This wrapper runs build_gallery_site.sh followed by the cross-platform
# open_gallery.py helper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/build_gallery_site.sh"
python "$SCRIPT_DIR/open_gallery.py"
