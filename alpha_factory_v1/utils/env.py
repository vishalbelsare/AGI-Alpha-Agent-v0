# SPDX-License-Identifier: Apache-2.0
"""Environment file helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
import logging


log = logging.getLogger(__name__)


def _load_env_file(path: str | os.PathLike[str]) -> Dict[str, str]:
    """Return key/value pairs from ``path``.

    Falls back to a minimal parser when :mod:`python_dotenv` is unavailable.
    """
    try:  # pragma: no cover - optional dependency
        from dotenv import dotenv_values

        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except Exception:  # noqa: BLE001 - any import/parsing error falls back
        pass

    data: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip().strip('"')
    return data


def _env_int(name: str, default: int) -> int:
    """Return ``int`` environment value or ``default`` if conversion fails."""

    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        log.warning("Invalid %s=%r, using default %s", name, val, default)
        return default
