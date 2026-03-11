"""Top-level package for the PyTorch implementation of SCOPE."""

from importlib.metadata import version as _pkg_version

from .config import SimulationConfig
from .canopy.foursail import FourSAILModel, FourSAILResult, campbell_lidf
from .canopy.fluorescence import CanopyFluorescenceModel, CanopyFluorescenceResult
from .canopy.reflectance import CanopyReflectanceModel, CanopyReflectanceResult
from .spectral.fluspect import FluspectModel, LeafBioBatch, LeafOptics, OptiPar, SpectralGrids
from .spectral.loaders import FluspectResources, SoilSpectraLibrary, load_fluspect_resources, load_scope_filenames, load_soil_spectra
from .spectral.soil import BSMSoilParameters, SoilBSMModel, SoilEmpiricalParams

__all__ = [
    "SimulationConfig",
    "FluspectModel",
    "LeafBioBatch",
    "LeafOptics",
    "OptiPar",
    "SpectralGrids",
    "FluspectResources",
    "SoilSpectraLibrary",
    "BSMSoilParameters",
    "SoilBSMModel",
    "SoilEmpiricalParams",
    "load_fluspect_resources",
    "load_scope_filenames",
    "load_soil_spectra",
    "FourSAILModel",
    "FourSAILResult",
    "CanopyFluorescenceModel",
    "CanopyFluorescenceResult",
    "CanopyReflectanceModel",
    "CanopyReflectanceResult",
    "campbell_lidf",
]

try:  # pragma: no cover
    __version__ = _pkg_version("scope-torch")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"
