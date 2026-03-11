"""Top-level package for the PyTorch implementation of SCOPE."""

from importlib.metadata import version as _pkg_version

from .config import SimulationConfig
from .spectral.fluspect import FluspectModel, LeafBioBatch, LeafOptics, OptiPar, SpectralGrids

__all__ = [
    "SimulationConfig",
    "FluspectModel",
    "LeafBioBatch",
    "LeafOptics",
    "OptiPar",
    "SpectralGrids",
]

try:  # pragma: no cover
    __version__ = _pkg_version("scope-torch")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"
