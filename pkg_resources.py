from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version


@dataclass(frozen=True)
class Distribution:
    version: str


def get_distribution(name: str) -> Distribution:
    """Compatibility shim for packages that still import pkg_resources."""
    return Distribution(version=version(name))
