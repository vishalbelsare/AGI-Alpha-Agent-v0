from pydantic import BaseSettings
from typing import Any, Dict, TypeAlias

SettingsConfigDict: TypeAlias = Dict[str, Any]
__all__ = ["BaseSettings", "SettingsConfigDict"]
